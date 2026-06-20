import gc
import json
import math
import statistics
import threading
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import pynvml
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification


MODEL_DIR = Path("models_updated/distilbert_agnews_bs8_ep3_wd001_warmup500")
DATA_DIR = Path("data/tokenized/distilbert_agnews_maxlen128")
RESULTS_DIR = Path("results_updated")

OUTPUT_JSON = RESULTS_DIR / "distilbert_haq_inference_benchmark_bs8_ep3_wd001_warmup500.json"
POLICY_JSON = RESULTS_DIR / "distilbert_haq_policy_bs8_ep3_wd001_warmup500.json"

BATCH_SIZE = 8
CALIBRATION_BATCH_SIZE = 8
CALIBRATION_SAMPLES = 512

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

WARMUP_BATCHES = 10
POWER_SAMPLE_INTERVAL_SEC = 0.05

LOW_BITS = 4
MID_BITS = 6
HIGH_BITS = 8
ACTIVATION_BITS = 8

# HAQ-style policy target:
# use low precision on hardware-expensive but low-sensitivity layers.
LOW_BIT_FRACTION = 0.35
MID_BIT_FRACTION = 0.35
# remaining layers become HIGH_BITS


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def move_batch_to_device(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def make_dataloader(dataset, batch_size, shuffle=False):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )


def tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def estimate_actual_model_storage_mb(model: nn.Module) -> float:
    total_bytes = 0
    for param in model.parameters():
        total_bytes += tensor_nbytes(param)
    for buffer in model.buffers():
        total_bytes += tensor_nbytes(buffer)
    return total_bytes / 1024**2


def fake_quantize_activation(x: torch.Tensor, bits: int | None):
    if bits is None:
        return x

    qmax = (2 ** (bits - 1)) - 1
    max_abs = x.detach().abs().amax(dim=-1, keepdim=True)
    scale = torch.clamp(max_abs / qmax, min=1e-8)

    qx = torch.round(x / scale)
    qx = torch.clamp(qx, min=-qmax, max=qmax)

    return qx * scale


def symmetric_quantize_weight_per_output_channel(weight: torch.Tensor, bits: int):
    qmax = (2 ** (bits - 1)) - 1
    weight_fp32 = weight.detach().float().cpu()

    max_abs = weight_fp32.abs().amax(dim=1, keepdim=True)
    scale = torch.clamp(max_abs / qmax, min=1e-8)

    qweight = torch.round(weight_fp32 / scale)
    qweight = torch.clamp(qweight, min=-qmax, max=qmax).to(torch.int8)

    return qweight, scale


class HAQQuantizedLinear(nn.Module):
    """
    Mixed-precision Linear layer.

    Assigned bit-width controls the quantization grid.
    qweight is physically stored as int8 for implementation simplicity.
    Effective packed model size is computed separately.
    """

    def __init__(self, qweight, scale, bias, weight_bits, activation_bits):
        super().__init__()

        self.weight_bits = int(weight_bits)
        self.activation_bits = None if activation_bits is None else int(activation_bits)

        self.register_buffer("qweight", qweight.contiguous())
        self.register_buffer("scale", scale.contiguous())

        if bias is not None:
            self.register_buffer("bias", bias.detach().float().cpu().contiguous())
        else:
            self.bias = None

    @classmethod
    def from_float(cls, linear: nn.Linear, weight_bits: int, activation_bits: int | None):
        qweight, scale = symmetric_quantize_weight_per_output_channel(
            linear.weight,
            bits=weight_bits,
        )
        bias = linear.bias.detach().clone() if linear.bias is not None else None

        return cls(
            qweight=qweight,
            scale=scale,
            bias=bias,
            weight_bits=weight_bits,
            activation_bits=activation_bits,
        )

    def forward(self, x):
        x = fake_quantize_activation(x, self.activation_bits)

        weight = self.qweight.to(dtype=x.dtype) * self.scale.to(dtype=x.dtype)

        bias = None
        if self.bias is not None:
            bias = self.bias.to(dtype=x.dtype)

        return F.linear(x, weight, bias)


def get_parent_module(root: nn.Module, module_name: str):
    parts = module_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def get_linear_layer_names(model: nn.Module):
    return [
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
    ]


def estimate_sensitivity(model, calibration_loader, device):
    """
    Accuracy-sensitivity proxy.

    Original HAQ uses hardware feedback in the policy loop.
    We also need an accuracy-risk signal, so we use mean squared gradient
    of each Linear layer's weight over calibration data.

    Higher sensitivity means higher accuracy risk when quantized aggressively.
    """
    print("\nEstimating layer sensitivity for HAQ policy...")

    model.eval()
    model.zero_grad(set_to_none=True)

    linear_layer_names = get_linear_layer_names(model)
    weight_param_to_layer = {
        f"{name}.weight": name
        for name in linear_layer_names
    }

    sensitivity_sum = {name: 0.0 for name in linear_layer_names}
    sensitivity_count = {name: 0 for name in linear_layer_names}
    named_params = dict(model.named_parameters())

    for batch in tqdm(calibration_loader, desc="Calibration sensitivity"):
        batch = move_batch_to_device(batch, device)

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
    for name in linear_layer_names:
        sensitivity[name] = sensitivity_sum[name] / max(sensitivity_count[name], 1)

    model.zero_grad(set_to_none=True)
    return sensitivity


def collect_layer_hardware_costs(model: nn.Module):
    """
    Hardware cost proxy.

    HAQ uses direct hardware feedback such as latency/energy.
    On this server, we approximate per-layer hardware pressure using:
        number of weights = out_features * in_features

    Larger layers are more expensive in memory bandwidth and matmul cost.
    """
    costs = {}

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            costs[name] = module.weight.numel()

    return costs


def assign_haq_bits(sensitivity: dict[str, float], hardware_costs: dict[str, int]):
    """
    HAQ-style hardware-aware bit assignment.

    We want to reduce precision where hardware benefit is high and accuracy risk is low.

    Define:
        benefit_score = normalized_hardware_cost / normalized_sensitivity

    High benefit_score means:
        large layer + low sensitivity -> good candidate for lower bits

    Then:
        top LOW_BIT_FRACTION benefit layers -> 4-bit
        next MID_BIT_FRACTION benefit layers -> 6-bit
        remaining layers -> 8-bit

    Classifier/pre_classifier stay 8-bit.
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

        # Lower sensitivity should increase compression benefit.
        # Add 0.05 to avoid exploding scores.
        benefit = (cost_norm + 0.05) / (sens_norm + 0.05)

        # Force final classifier layers to be protected.
        if "classifier" in name or "pre_classifier" in name:
            benefit = -1.0

        scores[name] = benefit

    sorted_by_benefit = sorted(
        layer_names,
        key=lambda n: scores[n],
        reverse=True,
    )

    n = len(sorted_by_benefit)
    n_low = math.floor(LOW_BIT_FRACTION * n)
    n_mid = math.floor(MID_BIT_FRACTION * n)

    low_set = set(sorted_by_benefit[:n_low])
    mid_set = set(sorted_by_benefit[n_low:n_low + n_mid])

    assignment = {}

    for name in layer_names:
        if "classifier" in name or "pre_classifier" in name:
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


def replace_linear_layers_with_haq(model: nn.Module, assignment: dict[str, int]):
    replaced = []

    for layer_name, bits in assignment.items():
        parent, child_name = get_parent_module(model, layer_name)
        child = getattr(parent, child_name)

        if not isinstance(child, nn.Linear):
            raise TypeError(f"Expected nn.Linear at {layer_name}, got {type(child)}")

        qlinear = HAQQuantizedLinear.from_float(
            child,
            weight_bits=bits,
            activation_bits=ACTIVATION_BITS,
        )

        setattr(parent, child_name, qlinear)
        replaced.append(layer_name)

    return replaced


def estimate_effective_haq_storage_mb(model: nn.Module):
    """
    Theoretical packed storage estimate based on assigned weight bits.
    """
    total_bytes = 0.0
    quantized_prefixes = []

    for module_name, module in model.named_modules():
        if isinstance(module, HAQQuantizedLinear):
            quantized_prefixes.append(module_name)

            total_bytes += module.qweight.numel() * (module.weight_bits / 8.0)
            total_bytes += tensor_nbytes(module.scale)

            if module.bias is not None:
                total_bytes += tensor_nbytes(module.bias)

    for name, param in model.named_parameters():
        total_bytes += tensor_nbytes(param)

    for name, buffer in model.named_buffers():
        belongs_to_quantized = any(
            name.startswith(prefix + ".")
            for prefix in quantized_prefixes
        )
        if not belongs_to_quantized:
            total_bytes += tensor_nbytes(buffer)

    return total_bytes / 1024**2


class NvmlPowerMonitor:
    def __init__(self, gpu_index=0, interval_sec=0.05):
        self.gpu_index = gpu_index
        self.interval_sec = interval_sec
        self.samples = []
        self._stop_event = threading.Event()
        self._thread = None
        self._handle = None
        self.available = False
        self.error = None

    def start(self):
        try:
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
            self.available = True
        except Exception as exc:
            self.error = repr(exc)
            self.available = False
            return

        self.samples = []
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def _sample_loop(self):
        while not self._stop_event.is_set():
            try:
                now = time.perf_counter()
                power_w = pynvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0
                mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
                used_mem_gb = mem.used / 1024**3
                self.samples.append((now, power_w, used_mem_gb))
            except Exception as exc:
                self.error = repr(exc)
                break

            time.sleep(self.interval_sec)

    def stop(self):
        if self.available:
            self._stop_event.set()
            if self._thread is not None:
                self._thread.join(timeout=2.0)

            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

    def summarize(self):
        if not self.samples:
            return {
                "nvml_available": self.available,
                "nvml_error": self.error,
                "energy_joules": None,
                "avg_power_watts": None,
                "min_power_watts": None,
                "max_power_watts": None,
                "peak_nvml_memory_used_gb": None,
                "num_power_samples": 0,
            }

        energy_joules = 0.0

        for i in range(1, len(self.samples)):
            t0, p0, _ = self.samples[i - 1]
            t1, p1, _ = self.samples[i]
            dt = t1 - t0
            avg_p = (p0 + p1) / 2.0
            energy_joules += avg_p * dt

        powers = [p for _, p, _ in self.samples]
        memories = [m for _, _, m in self.samples]

        return {
            "nvml_available": self.available,
            "nvml_error": self.error,
            "energy_joules": energy_joules,
            "avg_power_watts": statistics.mean(powers),
            "min_power_watts": min(powers),
            "max_power_watts": max(powers),
            "peak_nvml_memory_used_gb": max(memories),
            "num_power_samples": len(self.samples),
        }


@torch.inference_mode()
def warmup_model(model, dataloader, device, num_batches):
    model.eval()

    if num_batches <= 0:
        return

    print(f"Running {num_batches} warmup batches...")

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= num_batches:
            break

        batch = move_batch_to_device(batch, device)
        _ = model(**batch)

    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.inference_mode()
def benchmark(model, dataloader, device):
    model.eval()

    all_preds = []
    all_labels = []

    total_loss = 0.0
    total_samples = 0

    batch_latencies_ms = []

    clear_memory()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    power_monitor = NvmlPowerMonitor(
        gpu_index=0,
        interval_sec=POWER_SAMPLE_INTERVAL_SEC,
    )

    print("Starting NVML power monitor...")
    power_monitor.start()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    benchmark_start = time.perf_counter()

    for batch in tqdm(dataloader, desc="Benchmarking HAQ inference"):
        batch = move_batch_to_device(batch, device)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        batch_start = time.perf_counter()

        outputs = model(**batch)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        batch_end = time.perf_counter()

        logits = outputs.logits
        loss = outputs.loss
        preds = torch.argmax(logits, dim=-1)

        batch_size = batch["labels"].size(0)
        total_samples += batch_size
        total_loss += loss.item() * batch_size

        all_preds.extend(preds.detach().cpu().numpy().tolist())
        all_labels.extend(batch["labels"].detach().cpu().numpy().tolist())

        batch_latency_ms = (batch_end - batch_start) * 1000
        batch_latencies_ms.append(batch_latency_ms)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    benchmark_end = time.perf_counter()

    power_monitor.stop()
    power_summary = power_monitor.summarize()

    elapsed_sec = benchmark_end - benchmark_start

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels,
        all_preds,
        average="weighted",
        zero_division=0,
    )

    result = {
        "model": "DistilBERT",
        "dataset": "AG News",
        "method": "HAQ",
        "method_description": "HAQ-style hardware-aware mixed-precision quantization: layer bits selected using hardware-cost and sensitivity signals.",
        "activation_bits": ACTIVATION_BITS,
        "device": device,
        "batch_size": BATCH_SIZE,
        "warmup_batches": WARMUP_BATCHES,
        "eval_samples": total_samples,
        "eval_loss": total_loss / max(total_samples, 1),
        "accuracy": accuracy,
        "precision_weighted": precision,
        "recall_weighted": recall,
        "f1_weighted": f1,
        "total_inference_time_sec": elapsed_sec,
        "latency_ms_per_sample": (elapsed_sec / max(total_samples, 1)) * 1000,
        "throughput_samples_per_sec": total_samples / max(elapsed_sec, 1e-9),
        "mean_batch_latency_ms": statistics.mean(batch_latencies_ms),
        "median_batch_latency_ms": statistics.median(batch_latencies_ms),
        "min_batch_latency_ms": min(batch_latencies_ms),
        "max_batch_latency_ms": max(batch_latencies_ms),
    }

    if torch.cuda.is_available():
        result["peak_torch_memory_allocated_gb"] = torch.cuda.max_memory_allocated() / 1024**3
        result["peak_torch_memory_reserved_gb"] = torch.cuda.max_memory_reserved() / 1024**3
        result["gpu_name"] = torch.cuda.get_device_name(0)

    result.update(power_summary)

    if result["energy_joules"] is not None:
        result["energy_joules_per_sample"] = result["energy_joules"] / max(total_samples, 1)
    else:
        result["energy_joules_per_sample"] = None

    return result


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    clear_memory()

    print("=" * 100)
    print("DistilBERT AG News HAQ-style benchmark")
    print("=" * 100)

    print("Device:", DEVICE)
    if DEVICE == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        print("Total VRAM GB:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 4))

    print("\nLoading tokenized dataset:", DATA_DIR)
    dataset = load_from_disk(str(DATA_DIR))

    test_dataset = dataset["test"]
    calibration_dataset = dataset["train"].select(range(CALIBRATION_SAMPLES))

    test_loader = make_dataloader(test_dataset, BATCH_SIZE, shuffle=False)
    calibration_loader = make_dataloader(
        calibration_dataset,
        CALIBRATION_BATCH_SIZE,
        shuffle=False,
    )

    print("Test samples:", len(test_dataset))
    print("Calibration samples:", len(calibration_dataset))

    print("\nLoading trained baseline model:", MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
    model.to(DEVICE)
    model.eval()

    baseline_storage_mb = estimate_actual_model_storage_mb(model)
    print(f"Baseline model storage estimate: {baseline_storage_mb:.2f} MB")

    sensitivity = estimate_sensitivity(model, calibration_loader, DEVICE)
    hardware_costs = collect_layer_hardware_costs(model)
    assignment, ranked_policy = assign_haq_bits(sensitivity, hardware_costs)

    bit_counts = {}
    for bits in assignment.values():
        bit_counts[bits] = bit_counts.get(bits, 0) + 1

    policy_report = {
        "method": "HAQ",
        "policy_description": "Hardware-aware mixed precision using layer weight count as hardware-cost proxy and gradient-square as sensitivity proxy.",
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
    clear_memory()

    print("Applying HAQ-style mixed precision quantization...")
    replaced_layers = replace_linear_layers_with_haq(model, assignment)
    print(f"Replaced Linear layers: {len(replaced_layers)}")

    actual_storage_mb = estimate_actual_model_storage_mb(model)
    effective_storage_mb = estimate_effective_haq_storage_mb(model)
    avg_bits = sum(assignment.values()) / len(assignment)

    print(f"Actual implementation storage estimate: {actual_storage_mb:.2f} MB")
    print(f"Effective packed HAQ storage estimate: {effective_storage_mb:.2f} MB")
    print(f"Average assigned weight bits: {avg_bits:.3f}")

    print("\nMoving HAQ model to device...")
    model.to(DEVICE)
    model.eval()

    warmup_model(model, test_loader, DEVICE, WARMUP_BATCHES)

    result = benchmark(model, test_loader, DEVICE)

    result["model_dir"] = str(MODEL_DIR)
    result["data_dir"] = str(DATA_DIR)
    result["baseline_storage_mb_before_quantization"] = baseline_storage_mb
    result["actual_storage_mb_after_quantization"] = actual_storage_mb
    result["effective_packed_storage_mb_after_quantization"] = effective_storage_mb
    result["estimated_effective_storage_compression_ratio"] = baseline_storage_mb / max(effective_storage_mb, 1e-9)
    result["num_replaced_linear_layers"] = len(replaced_layers)
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
    clear_memory()

    print("\nGPU cleanup called.")


if __name__ == "__main__":
    main()
