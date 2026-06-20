import importlib.util
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification


HAWQ_SCRIPT = Path("scripts/17_benchmark_gpt2_hawq.py")


def load_hawq_module():
    spec = importlib.util.spec_from_file_location("gpt2_hawq", HAWQ_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


hawq = load_hawq_module()
zq = hawq.zq


OUTPUT_JSON = zq.RESULTS_DIR / "gpt2_haq_inference_benchmark_bs8_ep3_wd001_warmup500.json"
POLICY_JSON = zq.RESULTS_DIR / "gpt2_haq_policy_bs8_ep3_wd001_warmup500.json"

CALIBRATION_SAMPLES = 512
CALIBRATION_BATCH_SIZE = 8

LOW_BITS = 4
MID_BITS = 6
HIGH_BITS = 8
ACTIVATION_BITS = 8

LOW_BIT_FRACTION = 0.35
MID_BIT_FRACTION = 0.35


def collect_layer_hardware_costs(model: nn.Module):
    """
    HAQ-style hardware cost proxy.

    Original HAQ uses direct hardware feedback. Here we use number of weights
    as a hardware pressure proxy because larger GPT2 Conv1D/Linear layers
    usually cost more memory bandwidth and matrix multiplication work.
    """
    costs = {}

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) or zq.is_gpt2_conv1d(module):
            costs[name] = module.weight.numel()

    return costs


def assign_haq_bits(sensitivity: dict[str, float], hardware_costs: dict[str, int]):
    """
    HAQ-style bit assignment.

    We compress layers aggressively when:
      - hardware cost is high
      - sensitivity is low

    benefit_score = normalized_cost / normalized_sensitivity

    Higher benefit_score -> better candidate for 4-bit.
    """
    layer_names = list(sensitivity.keys())

    sens_values = [sensitivity[n] for n in layer_names]
    cost_values = [hardware_costs[n] for n in layer_names]

    min_sens, max_sens = min(sens_values), max(sens_values)
    min_cost, max_cost = min(cost_values), max(cost_values)

    scores = {}

    for name in layer_names:
        sens_norm = (sensitivity[name] - min_sens) / max(max_sens - min_sens, 1e-12)
        cost_norm = (hardware_costs[name] - min_cost) / max(max_cost - min_cost, 1e-12)

        benefit = (cost_norm + 0.05) / (sens_norm + 0.05)

        # GPT2 classification head is called "score"; protect it.
        if name == "score" or "classifier" in name:
            benefit = -1.0

        scores[name] = benefit

    sorted_by_benefit = sorted(layer_names, key=lambda n: scores[n], reverse=True)

    n = len(sorted_by_benefit)
    n_low = math.floor(LOW_BIT_FRACTION * n)
    n_mid = math.floor(MID_BIT_FRACTION * n)

    low_set = set(sorted_by_benefit[:n_low])
    mid_set = set(sorted_by_benefit[n_low:n_low + n_mid])

    assignment = {}

    for name in layer_names:
        if name == "score" or "classifier" in name:
            assignment[name] = HIGH_BITS
        elif name in low_set:
            assignment[name] = LOW_BITS
        elif name in mid_set:
            assignment[name] = MID_BITS
        else:
            assignment[name] = HIGH_BITS

    ranked_policy = [
        {
            "layer": name,
            "hardware_cost_num_weights": hardware_costs[name],
            "sensitivity": sensitivity[name],
            "haq_hardware_benefit_score": scores[name],
            "assigned_weight_bits": assignment[name],
        }
        for name in sorted_by_benefit
    ]

    return assignment, ranked_policy


def main():
    zq.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    zq.clear_memory()

    print("=" * 100)
    print("GPT2 AG News HAQ-style benchmark")
    print("=" * 100)

    print("Device:", zq.DEVICE)
    if zq.DEVICE == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        print("Total VRAM GB:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 4))

    print("\nLoading tokenized dataset:", zq.DATA_DIR)
    dataset = zq.load_from_disk(str(zq.DATA_DIR))

    test_dataset = dataset["test"]
    calibration_dataset = dataset["train"].select(range(CALIBRATION_SAMPLES))

    test_loader = zq.make_dataloader(test_dataset, zq.BATCH_SIZE)
    calibration_loader = zq.DataLoader(
        calibration_dataset,
        batch_size=CALIBRATION_BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    print("Test samples:", len(test_dataset))
    print("Calibration samples:", len(calibration_dataset))

    print("\nLoading GPT2 baseline model:", zq.MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(str(zq.MODEL_DIR))

    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = zq.GPT2_PAD_TOKEN_ID

    model.config.use_cache = False
    model.to(zq.DEVICE)
    model.eval()

    baseline_storage_mb = zq.estimate_model_storage_mb(model)
    print(f"Baseline model storage estimate: {baseline_storage_mb:.2f} MB")

    sensitivity = hawq.estimate_hawq_sensitivity(model, calibration_loader, zq.DEVICE)
    hardware_costs = collect_layer_hardware_costs(model)
    assignment, ranked_policy = assign_haq_bits(sensitivity, hardware_costs)

    bit_counts = {}
    for bits in assignment.values():
        bit_counts[bits] = bit_counts.get(bits, 0) + 1

    policy_report = {
        "method": "HAQ",
        "model": "GPT2",
        "policy_description": "Hardware-aware mixed precision using weight count as hardware-cost proxy and gradient-square as sensitivity proxy.",
        "calibration_samples": CALIBRATION_SAMPLES,
        "low_bits": LOW_BITS,
        "mid_bits": MID_BITS,
        "high_bits": HIGH_BITS,
        "activation_bits": ACTIVATION_BITS,
        "low_bit_fraction": LOW_BIT_FRACTION,
        "mid_bit_fraction": MID_BIT_FRACTION,
        "bit_assignment_counts": bit_counts,
        "ranked_layers_high_to_low_hardware_benefit": ranked_policy,
    }

    with open(POLICY_JSON, "w") as f:
        json.dump(policy_report, f, indent=2)

    print("\nSaved HAQ policy report to:", POLICY_JSON)

    print("\nBit assignment counts:")
    print(bit_counts)

    print("\nTop 10 layers selected by HAQ hardware-benefit score:")
    for row in ranked_policy[:10]:
        print(
            f"  {row['layer']:70s} "
            f"cost={row['hardware_cost_num_weights']:9d} "
            f"sens={row['sensitivity']:.8e} "
            f"score={row['haq_hardware_benefit_score']:.6f} "
            f"bits={row['assigned_weight_bits']}"
        )

    print("\nMoving model to CPU before replacement...")
    model.to("cpu")
    zq.clear_memory()

    print("Applying GPT2 HAQ-style mixed precision quantization...")
    replaced_layers = hawq.replace_layers_with_hawq(model, assignment)

    print(f"Replaced layers: {len(replaced_layers)}")
    print("First 12 replaced layers:")
    for item in replaced_layers[:12]:
        print(f"  - {item['name']} ({item['type']}, {item['weight_bits']}-bit)")

    actual_storage_mb = zq.estimate_model_storage_mb(model)
    effective_storage_mb = hawq.estimate_effective_hawq_storage_mb(model)
    avg_bits = sum(assignment.values()) / len(assignment)

    print(f"Actual implementation storage estimate: {actual_storage_mb:.2f} MB")
    print(f"Effective packed HAQ storage estimate: {effective_storage_mb:.2f} MB")
    print(f"Average assigned weight bits: {avg_bits:.3f}")

    print("\nMoving HAQ GPT2 model to device...")
    model.to(zq.DEVICE)
    model.eval()

    zq.warmup_model(model, test_loader, zq.DEVICE, zq.WARMUP_BATCHES)

    result = zq.benchmark(model, test_loader, zq.DEVICE)

    result["method"] = "HAQ"
    result["method_description"] = "HAQ-style hardware-aware mixed precision for GPT2 Conv1D and Linear layers."
    result["model_dir"] = str(zq.MODEL_DIR)
    result["data_dir"] = str(zq.DATA_DIR)
    result["baseline_storage_mb_before_quantization"] = baseline_storage_mb
    result["actual_storage_mb_after_quantization"] = actual_storage_mb
    result["effective_packed_storage_mb_after_quantization"] = effective_storage_mb
    result["estimated_effective_storage_compression_ratio"] = baseline_storage_mb / max(effective_storage_mb, 1e-9)
    result["num_replaced_layers"] = len(replaced_layers)
    result["num_replaced_gpt2_conv1d_layers"] = sum(1 for x in replaced_layers if x["type"] == "GPT2Conv1D")
    result["num_replaced_linear_layers"] = sum(1 for x in replaced_layers if x["type"] == "Linear")
    result["bit_assignment_counts"] = bit_counts
    result["average_assigned_weight_bits"] = avg_bits
    result["calibration_samples"] = CALIBRATION_SAMPLES
    result["policy_report_path"] = str(POLICY_JSON)

    print("\nBenchmark result:")
    print(json.dumps(result, indent=2))

    with open(OUTPUT_JSON, "w") as f:
        json.dump(result, f, indent=2)

    print("\nSaved benchmark result to:", OUTPUT_JSON)

    del model
    zq.clear_memory()

    print("\nGPU cleanup called.")


if __name__ == "__main__":
    main()
