import importlib.util
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification


ZEROQUANT_SCRIPT = Path("scripts/16_benchmark_gpt2_zeroquant.py")


def load_zeroquant_module():
    spec = importlib.util.spec_from_file_location("gpt2_zeroquant", ZEROQUANT_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


zq = load_zeroquant_module()


OUTPUT_JSON = zq.RESULTS_DIR / "gpt2_duquant_inference_benchmark_bs8_ep3_wd001_warmup500.json"

WEIGHT_BITS = 8
ACTIVATION_BITS = 8
BLOCK_SIZE = 16


def make_hadamard(n: int) -> torch.Tensor:
    """
    Create normalized Hadamard matrix.

    n must be power of 2.
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
    Zigzag-like permutation:
        0, n-1, 1, n-2, 2, n-3, ...

    This distributes dimensions that may contain concentrated outliers.
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


def duquant_transform_last_dim(x: torch.Tensor, perm: torch.Tensor, hadamard: torch.Tensor, block_size: int):
    """
    Apply DuQuant-style transform to last dimension:
      1. permutation
      2. block-wise orthogonal Hadamard rotation
    """
    hidden = x.shape[-1]

    if hidden % block_size != 0:
        return x

    x_perm = x.index_select(dim=-1, index=perm)

    original_shape = x_perm.shape
    x_blocks = x_perm.reshape(*original_shape[:-1], hidden // block_size, block_size)

    x_rot = torch.matmul(x_blocks, hadamard.to(dtype=x.dtype))

    return x_rot.reshape(original_shape)


def duquant_transform_weight_linear(weight: torch.Tensor, block_size: int):
    """
    nn.Linear:
        y = x W^T

    Activation transform:
        x' = xR

    Equivalent weight:
        W' = W R

    Since our transform works on the last dimension, applying it to W directly
    gives W R.
    """
    w = weight.detach().float().cpu()
    in_features = w.shape[1]

    if in_features % block_size != 0:
        return w

    perm = zigzag_permutation_indices(in_features)
    h = make_hadamard(block_size)

    return duquant_transform_last_dim(w, perm, h, block_size).contiguous()


def duquant_transform_weight_conv1d(weight: torch.Tensor, block_size: int):
    """
    GPT2 Conv1D:
        y = x W

    W shape:
        [in_features, out_features]

    Activation transform:
        x' = xR

    Equivalent weight:
        W' = R^T W

    We compute this efficiently as:
        W' = transform(W.T).T
    """
    w = weight.detach().float().cpu()
    in_features = w.shape[0]

    if in_features % block_size != 0:
        return w

    perm = zigzag_permutation_indices(in_features)
    h = make_hadamard(block_size)

    transformed_w_t = duquant_transform_last_dim(w.t(), perm, h, block_size)
    return transformed_w_t.t().contiguous()


class DuQuantLinear(nn.Module):
    def __init__(self, qweight, scale, bias, weight_bits, activation_bits, block_size, in_features):
        super().__init__()

        self.weight_bits = int(weight_bits)
        self.activation_bits = None if activation_bits is None else int(activation_bits)
        self.block_size = int(block_size)
        self.in_features = int(in_features)

        self.register_buffer("qweight", qweight.contiguous())
        self.register_buffer("scale", scale.contiguous())

        perm = zigzag_permutation_indices(in_features)
        h = make_hadamard(block_size)

        self.register_buffer("perm", perm.contiguous())
        self.register_buffer("hadamard", h.contiguous())

        if bias is not None:
            self.register_buffer("bias", bias.detach().float().cpu().contiguous())
        else:
            self.bias = None

    @classmethod
    def from_float(cls, linear: nn.Linear, weight_bits: int, activation_bits: int | None, block_size: int):
        transformed_weight = duquant_transform_weight_linear(
            linear.weight,
            block_size=block_size,
        )

        qweight, scale = zq.quantize_linear_weight_per_output_channel(
            transformed_weight,
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
            in_features=linear.weight.shape[1],
        )

    def forward(self, x):
        x = duquant_transform_last_dim(
            x,
            perm=self.perm,
            hadamard=self.hadamard,
            block_size=self.block_size,
        )

        x = zq.fake_quantize_activation(x, self.activation_bits)

        weight = self.qweight.to(dtype=x.dtype) * self.scale.to(dtype=x.dtype)

        bias = None
        if self.bias is not None:
            bias = self.bias.to(dtype=x.dtype)

        return F.linear(x, weight, bias)


class DuQuantGPT2Conv1D(nn.Module):
    def __init__(self, qweight, scale, bias, weight_bits, activation_bits, block_size, in_features, out_features):
        super().__init__()

        self.weight_bits = int(weight_bits)
        self.activation_bits = None if activation_bits is None else int(activation_bits)
        self.block_size = int(block_size)

        self.in_features = int(in_features)
        self.out_features = int(out_features)

        self.register_buffer("qweight", qweight.contiguous())
        self.register_buffer("scale", scale.contiguous())

        perm = zigzag_permutation_indices(in_features)
        h = make_hadamard(block_size)

        self.register_buffer("perm", perm.contiguous())
        self.register_buffer("hadamard", h.contiguous())

        if bias is not None:
            self.register_buffer("bias", bias.detach().float().cpu().contiguous())
        else:
            self.bias = None

    @classmethod
    def from_float(cls, conv1d: nn.Module, weight_bits: int, activation_bits: int | None, block_size: int):
        transformed_weight = duquant_transform_weight_conv1d(
            conv1d.weight,
            block_size=block_size,
        )

        qweight, scale = zq.quantize_conv1d_weight_per_output_channel(
            transformed_weight,
            bits=weight_bits,
        )

        bias = conv1d.bias.detach().clone() if conv1d.bias is not None else None

        return cls(
            qweight=qweight,
            scale=scale,
            bias=bias,
            weight_bits=weight_bits,
            activation_bits=activation_bits,
            block_size=block_size,
            in_features=conv1d.weight.shape[0],
            out_features=conv1d.weight.shape[1],
        )

    def forward(self, x):
        x = duquant_transform_last_dim(
            x,
            perm=self.perm,
            hadamard=self.hadamard,
            block_size=self.block_size,
        )

        x = zq.fake_quantize_activation(x, self.activation_bits)

        weight = self.qweight.to(dtype=x.dtype) * self.scale.to(dtype=x.dtype)

        y = torch.matmul(x, weight)

        if self.bias is not None:
            y = y + self.bias.to(dtype=x.dtype)

        return y


def replace_layers_with_duquant(module: nn.Module, prefix: str = ""):
    replaced = []

    for child_name, child in list(module.named_children()):
        full_name = f"{prefix}.{child_name}" if prefix else child_name

        if isinstance(child, nn.Linear):
            new_layer = DuQuantLinear.from_float(
                child,
                weight_bits=WEIGHT_BITS,
                activation_bits=ACTIVATION_BITS,
                block_size=BLOCK_SIZE,
            )

            setattr(module, child_name, new_layer)

            replaced.append(
                {
                    "name": full_name,
                    "type": "Linear",
                    "weight_bits": WEIGHT_BITS,
                    "activation_bits": ACTIVATION_BITS,
                }
            )

        elif zq.is_gpt2_conv1d(child):
            new_layer = DuQuantGPT2Conv1D.from_float(
                child,
                weight_bits=WEIGHT_BITS,
                activation_bits=ACTIVATION_BITS,
                block_size=BLOCK_SIZE,
            )

            setattr(module, child_name, new_layer)

            replaced.append(
                {
                    "name": full_name,
                    "type": "GPT2Conv1D",
                    "weight_bits": WEIGHT_BITS,
                    "activation_bits": ACTIVATION_BITS,
                }
            )

        else:
            replaced.extend(replace_layers_with_duquant(child, prefix=full_name))

    return replaced


def estimate_effective_duquant_storage_mb(model: nn.Module):
    total_bytes = 0.0
    quantized_prefixes = []

    for module_name, module in model.named_modules():
        if isinstance(module, (DuQuantLinear, DuQuantGPT2Conv1D)):
            quantized_prefixes.append(module_name)

            total_bytes += module.qweight.numel() * (module.weight_bits / 8.0)
            total_bytes += zq.tensor_nbytes(module.scale)

            if module.bias is not None:
                total_bytes += zq.tensor_nbytes(module.bias)

            # Include small transform buffers.
            total_bytes += zq.tensor_nbytes(module.perm)
            total_bytes += zq.tensor_nbytes(module.hadamard)

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
    print("GPT2 AG News DuQuant-style benchmark")
    print("=" * 100)

    print("Device:", zq.DEVICE)
    if zq.DEVICE == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        print("Total VRAM GB:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 4))

    print("\nLoading tokenized dataset:", zq.DATA_DIR)
    dataset = zq.load_from_disk(str(zq.DATA_DIR))

    test_dataset = dataset["test"]
    test_loader = zq.make_dataloader(test_dataset, zq.BATCH_SIZE)

    print("Test samples:", len(test_dataset))

    print("\nLoading GPT2 baseline model on CPU:", zq.MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(str(zq.MODEL_DIR))

    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = zq.GPT2_PAD_TOKEN_ID

    model.config.use_cache = False
    model.eval()

    baseline_storage_mb = zq.estimate_model_storage_mb(model)
    print(f"Baseline model storage estimate before quantization: {baseline_storage_mb:.2f} MB")

    print("\nApplying GPT2 DuQuant-style quantization...")
    replaced_layers = replace_layers_with_duquant(model)

    print(f"Replaced layers: {len(replaced_layers)}")
    print("First 12 replaced layers:")
    for item in replaced_layers[:12]:
        print(f"  - {item['name']} ({item['type']})")

    actual_storage_mb = zq.estimate_model_storage_mb(model)
    effective_storage_mb = estimate_effective_duquant_storage_mb(model)

    compression_ratio = baseline_storage_mb / max(effective_storage_mb, 1e-9)

    print(f"Actual implementation storage estimate: {actual_storage_mb:.2f} MB")
    print(f"Effective packed DuQuant storage estimate: {effective_storage_mb:.2f} MB")
    print(f"Estimated effective storage compression ratio: {compression_ratio:.3f}x")

    zq.clear_memory()

    print("\nMoving DuQuant GPT2 model to device...")
    model.to(zq.DEVICE)
    model.eval()

    zq.warmup_model(model, test_loader, zq.DEVICE, zq.WARMUP_BATCHES)

    result = zq.benchmark(model, test_loader, zq.DEVICE)

    result["method"] = "DuQuant"
    result["method_description"] = "DuQuant-style outlier redistribution for GPT2 using zigzag permutation and block-wise Hadamard rotation with equivalent weight transformation."
    result["weight_bits"] = WEIGHT_BITS
    result["activation_bits"] = ACTIVATION_BITS
    result["block_size"] = BLOCK_SIZE
    result["model_dir"] = str(zq.MODEL_DIR)
    result["data_dir"] = str(zq.DATA_DIR)
    result["baseline_storage_mb_before_quantization"] = baseline_storage_mb
    result["actual_storage_mb_after_quantization"] = actual_storage_mb
    result["effective_packed_storage_mb_after_quantization"] = effective_storage_mb
    result["estimated_effective_storage_compression_ratio"] = compression_ratio
    result["num_replaced_layers"] = len(replaced_layers)
    result["num_replaced_gpt2_conv1d_layers"] = sum(1 for x in replaced_layers if x["type"] == "GPT2Conv1D")
    result["num_replaced_linear_layers"] = sum(1 for x in replaced_layers if x["type"] == "Linear")

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
