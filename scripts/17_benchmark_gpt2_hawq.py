import importlib.util
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification


ZEROQUANT_SCRIPT = Path("scripts/16_benchmark_gpt2_zeroquant.py")


def load_zeroquant_module():
    spec = importlib.util.spec_from_file_location("gpt2_zeroquant", ZEROQUANT_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


zq = load_zeroquant_module()


OUTPUT_JSON = zq.RESULTS_DIR / "gpt2_hawq_inference_benchmark_bs8_ep3_wd001_warmup500.json"
SENSITIVITY_JSON = zq.RESULTS_DIR / "gpt2_hawq_layer_sensitivity_bs8_ep3_wd001_warmup500.json"

CALIBRATION_SAMPLES = 512
CALIBRATION_BATCH_SIZE = 8

LOW_BITS = 4
MID_BITS = 6
HIGH_BITS = 8
ACTIVATION_BITS = 8


def get_parent_module(root: nn.Module, module_name: str):
    parts = module_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def get_quantizable_layer_names(model: nn.Module):
    names = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) or zq.is_gpt2_conv1d(module):
            names.append(name)

    return names


def estimate_hawq_sensitivity(model, calibration_loader, device):
    """
    HAWQ-style sensitivity estimation.

    Original HAWQ uses Hessian-based sensitivity.
    Here we use mean squared gradient as a practical empirical-Fisher proxy.

    Larger value = more sensitive = should receive higher precision.
    """
    print("\nEstimating GPT2 HAWQ-style layer sensitivity...")

    model.eval()
    model.zero_grad(set_to_none=True)

    layer_names = get_quantizable_layer_names(model)

    weight_param_to_layer = {
        f"{name}.weight": name
        for name in layer_names
    }

    sensitivity_sum = {name: 0.0 for name in layer_names}
    sensitivity_count = {name: 0 for name in layer_names}

    named_params = dict(model.named_parameters())

    for batch in tqdm(calibration_loader, desc="Calibration sensitivity"):
        batch = zq.move_batch_to_device(batch, device)

        model.zero_grad(set_to_none=True)

        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        for param_name, layer_name in weight_param_to_layer.items():
            param = named_params.get(param_name)

            if param is None or param.grad is None:
                continue

            score = param.grad.detach().float().pow(2).mean().item()
            sensitivity_sum[layer_name] += score
            sensitivity_count[layer_name] += 1

    sensitivity = {}
    for name in layer_names:
        sensitivity[name] = sensitivity_sum[name] / max(sensitivity_count[name], 1)

    model.zero_grad(set_to_none=True)
    return sensitivity


def assign_bits_from_sensitivity(sensitivity: dict[str, float]):
    """
    HAWQ-style mixed precision:
      top 30% most sensitive     -> 8-bit
      middle 40%                 -> 6-bit
      bottom 30% least sensitive -> 4-bit

    Classification head is forced to 8-bit.
    """
    layer_names = list(sensitivity.keys())

    sorted_layers = sorted(
        layer_names,
        key=lambda name: sensitivity[name],
        reverse=True,
    )

    n = len(sorted_layers)
    n_high = math.ceil(0.30 * n)
    n_low = math.floor(0.30 * n)

    high_set = set(sorted_layers[:n_high])
    low_set = set(sorted_layers[-n_low:]) if n_low > 0 else set()

    assignment = {}

    for name in sorted_layers:
        if "score" in name or "classifier" in name:
            assignment[name] = HIGH_BITS
        elif name in high_set:
            assignment[name] = HIGH_BITS
        elif name in low_set:
            assignment[name] = LOW_BITS
        else:
            assignment[name] = MID_BITS

    return assignment, sorted_layers


def replace_layers_with_hawq(model: nn.Module, assignment: dict[str, int]):
    replaced = []

    for layer_name, bits in assignment.items():
        parent, child_name = get_parent_module(model, layer_name)
        child = getattr(parent, child_name)

        if isinstance(child, nn.Linear):
            new_layer = zq.QuantizedLinear.from_float(
                child,
                weight_bits=bits,
                activation_bits=ACTIVATION_BITS,
            )
            layer_type = "Linear"

        elif zq.is_gpt2_conv1d(child):
            new_layer = zq.QuantizedGPT2Conv1D.from_float(
                child,
                weight_bits=bits,
                activation_bits=ACTIVATION_BITS,
            )
            layer_type = "GPT2Conv1D"

        else:
            raise TypeError(f"Unsupported layer at {layer_name}: {type(child)}")

        setattr(parent, child_name, new_layer)

        replaced.append(
            {
                "name": layer_name,
                "type": layer_type,
                "weight_bits": bits,
                "activation_bits": ACTIVATION_BITS,
            }
        )

    return replaced


def estimate_effective_hawq_storage_mb(model: nn.Module):
    """
    Theoretical packed storage.

    Implementation stores qweight as int8 for simplicity, but this estimates
    storage if 4/6/8-bit weights were actually packed.
    """
    total_bytes = 0.0
    quantized_prefixes = []

    for module_name, module in model.named_modules():
        if isinstance(module, (zq.QuantizedLinear, zq.QuantizedGPT2Conv1D)):
            quantized_prefixes.append(module_name)

            total_bytes += module.qweight.numel() * (module.weight_bits / 8.0)
            total_bytes += zq.tensor_nbytes(module.scale)

            if module.bias is not None:
                total_bytes += zq.tensor_nbytes(module.bias)

    for name, param in model.named_parameters():
        total_bytes += zq.tensor_nbytes(param)

    for name, buffer in model.named_buffers():
        belongs_to_quantized = any(
            name.startswith(prefix + ".")
            for prefix in quantized_prefixes
        )
        if not belongs_to_quantized:
            total_bytes += zq.tensor_nbytes(buffer)

    return total_bytes / 1024**2


def main():
    zq.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    zq.clear_memory()

    print("=" * 100)
    print("GPT2 AG News HAWQ-style benchmark")
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

    sensitivity = estimate_hawq_sensitivity(model, calibration_loader, zq.DEVICE)
    assignment, sorted_layers = assign_bits_from_sensitivity(sensitivity)

    bit_counts = {}
    for bits in assignment.values():
        bit_counts[bits] = bit_counts.get(bits, 0) + 1

    sensitivity_report = {
        "method": "HAWQ",
        "model": "GPT2",
        "sensitivity_proxy": "mean squared gradient of quantizable layer weight on calibration data",
        "calibration_samples": CALIBRATION_SAMPLES,
        "low_bits": LOW_BITS,
        "mid_bits": MID_BITS,
        "high_bits": HIGH_BITS,
        "activation_bits": ACTIVATION_BITS,
        "bit_assignment_counts": bit_counts,
        "layers_ranked_high_to_low_sensitivity": [
            {
                "layer": name,
                "sensitivity": sensitivity[name],
                "assigned_weight_bits": assignment[name],
            }
            for name in sorted_layers
        ],
    }

    with open(SENSITIVITY_JSON, "w") as f:
        json.dump(sensitivity_report, f, indent=2)

    print("\nSaved HAWQ sensitivity report to:", SENSITIVITY_JSON)

    print("\nBit assignment counts:")
    print(bit_counts)

    print("\nTop 10 most sensitive layers:")
    for name in sorted_layers[:10]:
        print(f"  {name:70s} sensitivity={sensitivity[name]:.8e} bits={assignment[name]}")

    print("\nMoving model to CPU before replacement...")
    model.to("cpu")
    zq.clear_memory()

    print("Applying GPT2 HAWQ-style mixed precision quantization...")
    replaced_layers = replace_layers_with_hawq(model, assignment)

    print(f"Replaced layers: {len(replaced_layers)}")
    print("First 12 replaced layers:")
    for item in replaced_layers[:12]:
        print(f"  - {item['name']} ({item['type']}, {item['weight_bits']}-bit)")

    actual_storage_mb = zq.estimate_model_storage_mb(model)
    effective_storage_mb = estimate_effective_hawq_storage_mb(model)

    avg_bits = sum(assignment.values()) / len(assignment)

    print(f"Actual implementation storage estimate: {actual_storage_mb:.2f} MB")
    print(f"Effective packed HAWQ storage estimate: {effective_storage_mb:.2f} MB")
    print(f"Average assigned weight bits: {avg_bits:.3f}")

    print("\nMoving HAWQ GPT2 model to device...")
    model.to(zq.DEVICE)
    model.eval()

    zq.warmup_model(model, test_loader, zq.DEVICE, zq.WARMUP_BATCHES)

    result = zq.benchmark(model, test_loader, zq.DEVICE)

    result["method"] = "HAWQ"
    result["method_description"] = "HAWQ-style sensitivity-aware mixed precision for GPT2 Conv1D and Linear layers."
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
    result["sensitivity_report_path"] = str(SENSITIVITY_JSON)

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
