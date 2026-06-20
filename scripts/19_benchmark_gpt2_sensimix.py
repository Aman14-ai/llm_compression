import importlib.util
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification


HAWQ_SCRIPT = Path("scripts/17_benchmark_gpt2_hawq.py")


def load_hawq_module():
    spec = importlib.util.spec_from_file_location("gpt2_hawq", HAWQ_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


hawq = load_hawq_module()
zq = hawq.zq


OUTPUT_JSON = zq.RESULTS_DIR / "gpt2_sensimix_inference_benchmark_bs8_ep3_wd001_warmup500.json"
SENSITIVITY_JSON = zq.RESULTS_DIR / "gpt2_sensimix_layer_sensitivity_bs8_ep3_wd001_warmup500.json"

CALIBRATION_SAMPLES = 512
CALIBRATION_BATCH_SIZE = 8

SENSITIVE_BITS = 8
INSENSITIVE_BITS = 1
ACTIVATION_BITS = 8
BINARY_FRACTION = 0.30


def binary_quantize_linear_weight_per_output_channel(weight: torch.Tensor):
    """
    nn.Linear weight shape:
        [out_features, in_features]

    Binary approximation:
        W_hat = scale * sign(W)
    where scale is per output channel.
    """
    w = weight.detach().float().cpu()

    scale = w.abs().mean(dim=1, keepdim=True)
    scale = torch.clamp(scale, min=1e-8)

    bweight = torch.where(
        w >= 0,
        torch.ones_like(w, dtype=torch.int8),
        -torch.ones_like(w, dtype=torch.int8),
    )

    return bweight, scale


def binary_quantize_conv1d_weight_per_output_channel(weight: torch.Tensor):
    """
    GPT2 Conv1D weight shape:
        [in_features, out_features]

    Output channels are columns, so scale is computed along dim=0.
    """
    w = weight.detach().float().cpu()

    scale = w.abs().mean(dim=0, keepdim=True)
    scale = torch.clamp(scale, min=1e-8)

    bweight = torch.where(
        w >= 0,
        torch.ones_like(w, dtype=torch.int8),
        -torch.ones_like(w, dtype=torch.int8),
    )

    return bweight, scale


class BinaryLinear(nn.Module):
    def __init__(self, bweight, scale, bias, activation_bits):
        super().__init__()

        self.weight_bits = 1
        self.activation_bits = None if activation_bits is None else int(activation_bits)

        self.register_buffer("bweight", bweight.contiguous())
        self.register_buffer("scale", scale.contiguous())

        if bias is not None:
            self.register_buffer("bias", bias.detach().float().cpu().contiguous())
        else:
            self.bias = None

    @classmethod
    def from_float(cls, linear: nn.Linear, activation_bits: int | None):
        bweight, scale = binary_quantize_linear_weight_per_output_channel(linear.weight)
        bias = linear.bias.detach().clone() if linear.bias is not None else None
        return cls(bweight, scale, bias, activation_bits)

    def forward(self, x):
        x = zq.fake_quantize_activation(x, self.activation_bits)

        weight = self.bweight.to(dtype=x.dtype) * self.scale.to(dtype=x.dtype)

        bias = None
        if self.bias is not None:
            bias = self.bias.to(dtype=x.dtype)

        return F.linear(x, weight, bias)


class BinaryGPT2Conv1D(nn.Module):
    def __init__(self, bweight, scale, bias, activation_bits):
        super().__init__()

        self.weight_bits = 1
        self.activation_bits = None if activation_bits is None else int(activation_bits)

        self.in_features = bweight.shape[0]
        self.out_features = bweight.shape[1]

        self.register_buffer("bweight", bweight.contiguous())
        self.register_buffer("scale", scale.contiguous())

        if bias is not None:
            self.register_buffer("bias", bias.detach().float().cpu().contiguous())
        else:
            self.bias = None

    @classmethod
    def from_float(cls, conv1d: nn.Module, activation_bits: int | None):
        bweight, scale = binary_quantize_conv1d_weight_per_output_channel(conv1d.weight)
        bias = conv1d.bias.detach().clone() if conv1d.bias is not None else None
        return cls(bweight, scale, bias, activation_bits)

    def forward(self, x):
        x = zq.fake_quantize_activation(x, self.activation_bits)

        weight = self.bweight.to(dtype=x.dtype) * self.scale.to(dtype=x.dtype)
        y = torch.matmul(x, weight)

        if self.bias is not None:
            y = y + self.bias.to(dtype=x.dtype)

        return y


def assign_sensimix_bits(sensitivity: dict[str, float]):
    """
    SensiMix-style assignment:
      least-sensitive 30% -> binary 1-bit
      remaining layers    -> INT8

    GPT2 classifier head "score" is protected at 8-bit.
    """
    layer_names = list(sensitivity.keys())

    sorted_layers = sorted(
        layer_names,
        key=lambda name: sensitivity[name],
        reverse=True,
    )

    n = len(sorted_layers)
    n_binary = math.floor(BINARY_FRACTION * n)

    binary_set = set(sorted_layers[-n_binary:]) if n_binary > 0 else set()

    assignment = {}

    for name in sorted_layers:
        if name == "score" or "classifier" in name:
            assignment[name] = 8
        elif name in binary_set:
            assignment[name] = 1
        else:
            assignment[name] = 8

    return assignment, sorted_layers


def replace_layers_with_sensimix(model: nn.Module, assignment: dict[str, int]):
    replaced = []

    for layer_name, bits in assignment.items():
        parent, child_name = hawq.get_parent_module(model, layer_name)
        child = getattr(parent, child_name)

        if isinstance(child, nn.Linear):
            if bits == 1:
                new_layer = BinaryLinear.from_float(
                    child,
                    activation_bits=ACTIVATION_BITS,
                )
            elif bits == 8:
                new_layer = zq.QuantizedLinear.from_float(
                    child,
                    weight_bits=8,
                    activation_bits=ACTIVATION_BITS,
                )
            else:
                raise ValueError(f"Unsupported SensiMix bits: {bits}")

            layer_type = "Linear"

        elif zq.is_gpt2_conv1d(child):
            if bits == 1:
                new_layer = BinaryGPT2Conv1D.from_float(
                    child,
                    activation_bits=ACTIVATION_BITS,
                )
            elif bits == 8:
                new_layer = zq.QuantizedGPT2Conv1D.from_float(
                    child,
                    weight_bits=8,
                    activation_bits=ACTIVATION_BITS,
                )
            else:
                raise ValueError(f"Unsupported SensiMix bits: {bits}")

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


def estimate_effective_sensimix_storage_mb(model: nn.Module):
    """
    Theoretical packed storage.

    Binary weights are stored as int8 in this practical implementation, but
    this function estimates true packed 1-bit storage.
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

        elif isinstance(module, (BinaryLinear, BinaryGPT2Conv1D)):
            quantized_prefixes.append(module_name)

            total_bytes += module.bweight.numel() * (module.weight_bits / 8.0)
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
    print("GPT2 AG News SensiMix-style benchmark")
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
    assignment, sorted_layers = assign_sensimix_bits(sensitivity)

    bit_counts = {}
    for bits in assignment.values():
        bit_counts[bits] = bit_counts.get(bits, 0) + 1

    sensitivity_report = {
        "method": "SensiMix",
        "model": "GPT2",
        "sensitivity_proxy": "mean squared gradient of quantizable layer weight on calibration data",
        "calibration_samples": CALIBRATION_SAMPLES,
        "binary_fraction": BINARY_FRACTION,
        "sensitive_bits": SENSITIVE_BITS,
        "insensitive_bits": INSENSITIVE_BITS,
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

    print("\nSaved SensiMix sensitivity report to:", SENSITIVITY_JSON)

    print("\nBit assignment counts:")
    print(bit_counts)

    print("\nBottom 10 least sensitive layers:")
    for name in sorted_layers[-10:]:
        print(f"  {name:70s} sensitivity={sensitivity[name]:.8e} bits={assignment[name]}")

    print("\nMoving model to CPU before replacement...")
    model.to("cpu")
    zq.clear_memory()

    print("Applying GPT2 SensiMix-style mixed precision quantization...")
    replaced_layers = replace_layers_with_sensimix(model, assignment)

    print(f"Replaced layers: {len(replaced_layers)}")
    print("First 12 replaced layers:")
    for item in replaced_layers[:12]:
        print(f"  - {item['name']} ({item['type']}, {item['weight_bits']}-bit)")

    actual_storage_mb = zq.estimate_model_storage_mb(model)
    effective_storage_mb = estimate_effective_sensimix_storage_mb(model)
    avg_bits = sum(assignment.values()) / len(assignment)

    print(f"Actual implementation storage estimate: {actual_storage_mb:.2f} MB")
    print(f"Effective packed SensiMix storage estimate: {effective_storage_mb:.2f} MB")
    print(f"Average assigned weight bits: {avg_bits:.3f}")

    print("\nMoving SensiMix GPT2 model to device...")
    model.to(zq.DEVICE)
    model.eval()

    zq.warmup_model(model, test_loader, zq.DEVICE, zq.WARMUP_BATCHES)

    result = zq.benchmark(model, test_loader, zq.DEVICE)

    result["method"] = "SensiMix"
    result["method_description"] = "SensiMix-style sensitivity-aware mixed precision for GPT2: sensitive layers INT8, least-sensitive layers binary 1-bit."
    result["binary_fraction"] = BINARY_FRACTION
    result["sensitive_bits"] = SENSITIVE_BITS
    result["insensitive_bits"] = INSENSITIVE_BITS
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
