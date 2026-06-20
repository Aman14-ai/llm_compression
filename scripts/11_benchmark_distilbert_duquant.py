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

OUTPUT_JSON = RESULTS_DIR / "distilbert_duquant_inference_benchmark_bs8_ep3_wd001_warmup500.json"

BATCH_SIZE = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

WARMUP_BATCHES = 10
POWER_SAMPLE_INTERVAL_SEC = 0.05

WEIGHT_BITS = 8
ACTIVATION_BITS = 8

# DuQuant-style block transform size.
# DistilBERT hidden size is 768, divisible by 16.
BLOCK_SIZE = 16


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def move_batch_to_device(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def make_dataloader(dataset, batch_size):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )


def tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def estimate_model_storage_mb(model: nn.Module) -> float:
    total_bytes = 0
    for param in model.parameters():
        total_bytes += tensor_nbytes(param)
    for buffer in model.buffers():
        total_bytes += tensor_nbytes(buffer)
    return total_bytes / 1024**2


def symmetric_quantize_weight_per_output_channel(weight: torch.Tensor, bits: int):
    qmax = (2 ** (bits - 1)) - 1

    weight_fp32 = weight.detach().float().cpu()

    max_abs = weight_fp32.abs().amax(dim=1, keepdim=True)
    scale = torch.clamp(max_abs / qmax, min=1e-8)

    qweight = torch.round(weight_fp32 / scale)
    qweight = torch.clamp(qweight, min=-qmax, max=qmax).to(torch.int8)

    return qweight, scale


def make_hadamard(n: int) -> torch.Tensor:
    """
    Create normalized Hadamard matrix for n that is a power of 2.

    H @ H.T = I after normalization.
    """
    if n < 1 or (n & (n - 1)) != 0:
        raise ValueError(f"Hadamard size must be power of 2, got {n}")

    h = torch.tensor([[1.0]])
    while h.shape[0] < n:
        h = torch.cat(
            [
                torch.cat([h, h], dim=1),
                torch.cat([h, -h], dim=1),
            ],
            dim=0,
        )

    h = h / math.sqrt(n)
    return h


def zigzag_permutation_indices(n: int) -> torch.Tensor:
    """
    Zigzag-like permutation.

    It alternates low/high indices:
        0, n-1, 1, n-2, 2, n-3, ...

    Goal: distribute potentially concentrated outlier dimensions.
    """
    indices = []
    left = 0
    right = n - 1

    while left <= right:
        indices.append(left)
        if left != right:
            indices.append(right)
        left += 1
        right -= 1

    return torch.tensor(indices, dtype=torch.long)


def fake_quantize_activation(x: torch.Tensor, bits: int | None):
    if bits is None:
        return x

    qmax = (2 ** (bits - 1)) - 1

    max_abs = x.detach().abs().amax(dim=-1, keepdim=True)
    scale = torch.clamp(max_abs / qmax, min=1e-8)

    qx = torch.round(x / scale)
    qx = torch.clamp(qx, min=-qmax, max=qmax)

    return qx * scale


def duquant_transform_activation(x: torch.Tensor, block_size: int):
    """
    DuQuant-style dual transformation approximation.

    For activation tensor x with last dimension hidden:
      1. zigzag-like permutation over hidden dimension
      2. block-wise orthogonal Hadamard rotation

    This redistributes large activation channels across dimensions before
    quantization, reducing outlier concentration.
    """
    hidden = x.shape[-1]

    if hidden % block_size != 0:
        return x

    device = x.device
    dtype = x.dtype

    perm = zigzag_permutation_indices(hidden).to(device)
    x_perm = x.index_select(dim=-1, index=perm)

    h = make_hadamard(block_size).to(device=device, dtype=dtype)

    original_shape = x_perm.shape
    x_blocks = x_perm.reshape(*original_shape[:-1], hidden // block_size, block_size)

    # Apply block-wise rotation along the last dimension.
    x_rot = torch.matmul(x_blocks, h)

    return x_rot.reshape(original_shape)


def duquant_fake_quantize_activation(x: torch.Tensor, bits: int | None, block_size: int):
    x = duquant_transform_activation(x, block_size=block_size)
    x = fake_quantize_activation(x, bits=bits)
    return x


class DuQuantLinear(nn.Module):
    """
    DuQuant-style Linear layer.

    Stored weights:
        INT8 per-output-channel quantized weights.

    Activation path:
        zigzag permutation + block-wise orthogonal rotation
        then INT8 fake quantization.

    Note:
        This is a practical DuQuant-style simulation. Real DuQuant applies
        carefully designed transformations with equivalent model rewrites.
    """

    def __init__(self, qweight, scale, bias, weight_bits, activation_bits, block_size):
        super().__init__()

        self.weight_bits = int(weight_bits)
        self.activation_bits = None if activation_bits is None else int(activation_bits)
        self.block_size = int(block_size)

        self.register_buffer("qweight", qweight.contiguous())
        self.register_buffer("scale", scale.contiguous())

        if bias is not None:
            self.register_buffer("bias", bias.detach().float().cpu().contiguous())
        else:
            self.bias = None

    @classmethod
    def from_float(cls, linear: nn.Linear, weight_bits: int, activation_bits: int | None, block_size: int):
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
            block_size=block_size,
        )

    def forward(self, x):
        x = duquant_fake_quantize_activation(
            x,
            bits=self.activation_bits,
            block_size=self.block_size,
        )

        weight = self.qweight.to(dtype=x.dtype) * self.scale.to(dtype=x.dtype)

        bias = None
        if self.bias is not None:
            bias = self.bias.to(dtype=x.dtype)

        return F.linear(x, weight, bias)


def replace_linear_layers_with_duquant(module: nn.Module, prefix: str = ""):
    replaced = []

    for child_name, child in list(module.named_children()):
        full_name = f"{prefix}.{child_name}" if prefix else child_name

        if isinstance(child, nn.Linear):
            qlinear = DuQuantLinear.from_float(
                child,
                weight_bits=WEIGHT_BITS,
                activation_bits=ACTIVATION_BITS,
                block_size=BLOCK_SIZE,
            )
            setattr(module, child_name, qlinear)
            replaced.append(full_name)
        else:
            replaced.extend(
                replace_linear_layers_with_duquant(
                    child,
                    prefix=full_name,
                )
            )

    return replaced


def estimate_effective_duquant_storage_mb(model: nn.Module):
    total_bytes = 0.0
    quantized_prefixes = []

    for module_name, module in model.named_modules():
        if isinstance(module, DuQuantLinear):
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

    for batch in tqdm(dataloader, desc="Benchmarking DuQuant inference"):
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
        "method": "DuQuant",
        "method_description": "DuQuant-style outlier distribution using zigzag permutation and block-wise Hadamard rotation before INT8 activation quantization.",
        "weight_bits": WEIGHT_BITS,
        "activation_bits": ACTIVATION_BITS,
        "block_size": BLOCK_SIZE,
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
        "estimated_model_storage_mb": estimate_model_storage_mb(model),
        "effective_packed_storage_mb": estimate_effective_duquant_storage_mb(model),
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
    print("DistilBERT AG News DuQuant-style benchmark")
    print("=" * 100)

    print("Device:", DEVICE)
    if DEVICE == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        print("Total VRAM GB:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 4))

    print("\nLoading tokenized dataset:", DATA_DIR)
    dataset = load_from_disk(str(DATA_DIR))
    test_dataset = dataset["test"]
    test_loader = make_dataloader(test_dataset, BATCH_SIZE)

    print("Test samples:", len(test_dataset))

    print("\nLoading trained baseline model on CPU:", MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
    model.eval()

    baseline_storage_mb = estimate_model_storage_mb(model)
    print(f"Baseline model storage estimate before quantization: {baseline_storage_mb:.2f} MB")

    print("\nApplying DuQuant-style Linear quantization...")
    replaced_layers = replace_linear_layers_with_duquant(model)

    print(f"Replaced Linear layers: {len(replaced_layers)}")
    print("First 10 replaced layers:")
    for name in replaced_layers[:10]:
        print("  -", name)

    actual_storage_mb = estimate_model_storage_mb(model)
    effective_storage_mb = estimate_effective_duquant_storage_mb(model)

    print(f"Actual implementation storage estimate: {actual_storage_mb:.2f} MB")
    print(f"Effective packed DuQuant storage estimate: {effective_storage_mb:.2f} MB")

    compression_ratio = baseline_storage_mb / max(effective_storage_mb, 1e-9)
    print(f"Estimated effective storage compression ratio: {compression_ratio:.3f}x")

    clear_memory()

    print("\nMoving DuQuant model to device...")
    model.to(DEVICE)
    model.eval()

    warmup_model(model, test_loader, DEVICE, WARMUP_BATCHES)

    result = benchmark(model, test_loader, DEVICE)

    result["model_dir"] = str(MODEL_DIR)
    result["data_dir"] = str(DATA_DIR)
    result["baseline_storage_mb_before_quantization"] = baseline_storage_mb
    result["actual_storage_mb_after_quantization"] = actual_storage_mb
    result["effective_packed_storage_mb_after_quantization"] = effective_storage_mb
    result["estimated_effective_storage_compression_ratio"] = compression_ratio
    result["num_replaced_linear_layers"] = len(replaced_layers)

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
