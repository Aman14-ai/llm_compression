"""
Fixed DuQuant-style benchmark.

Why this file exists:
The first naive DuQuant attempt transformed activations but not weights.
That changes the function of each Linear layer and destroyed accuracy.

Correct equivalence:
    y = x W^T
    x' = x R
    W' = W R
    y = x' W'^T = x R (W R)^T = x R R^T W^T = x W^T

So here we reuse the previous benchmark infrastructure but patch DuQuantLinear
so it transforms weights consistently with the activation transform.
"""

import importlib.util
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


OLD_SCRIPT = Path("scripts/11_benchmark_distilbert_duquant.py")


def load_old_module():
    spec = importlib.util.spec_from_file_location("duquant_old", OLD_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


duq = load_old_module()


def build_duquant_rotation_matrix(in_features: int, block_size: int) -> torch.Tensor:
    """
    Build matrix R such that:
        transformed_x = x @ R

    We get R by applying the same DuQuant activation transform to identity:
        I @ R = R
    """
    eye = torch.eye(in_features, dtype=torch.float32)
    rotation = duq.duquant_transform_activation(eye, block_size=block_size)
    return rotation.contiguous()


def transform_weight_equivalently(weight: torch.Tensor, block_size: int) -> torch.Tensor:
    """
    Given Linear weight W with shape [out_features, in_features],
    and activation transform x' = xR,
    use W' = W R.

    Then:
        x' W'^T = x W^T
    """
    weight_fp32 = weight.detach().float().cpu()
    in_features = weight_fp32.shape[1]

    if in_features % block_size != 0:
        return weight_fp32

    rotation = build_duquant_rotation_matrix(in_features, block_size)
    transformed_weight = weight_fp32 @ rotation
    return transformed_weight.contiguous()


class FixedDuQuantLinear(nn.Module):
    """
    Corrected DuQuant-style Linear layer.

    Activation path:
        x -> xR -> INT8 fake quantization

    Weight path:
        W -> WR -> INT8 weight quantization

    This preserves the original Linear function before quantization noise.
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
        transformed_weight = transform_weight_equivalently(
            linear.weight,
            block_size=block_size,
        )

        qweight, scale = duq.symmetric_quantize_weight_per_output_channel(
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
        )

    def forward(self, x):
        x = duq.duquant_fake_quantize_activation(
            x,
            bits=self.activation_bits,
            block_size=self.block_size,
        )

        weight = self.qweight.to(dtype=x.dtype) * self.scale.to(dtype=x.dtype)

        bias = None
        if self.bias is not None:
            bias = self.bias.to(dtype=x.dtype)

        return F.linear(x, weight, bias)


def main():
    # Monkey-patch the old module so all its existing benchmark code uses
    # the corrected DuQuantLinear class.
    duq.DuQuantLinear = FixedDuQuantLinear

    # Save corrected result to the original expected final path.
    duq.OUTPUT_JSON = duq.RESULTS_DIR / "distilbert_duquant_inference_benchmark_bs8_ep3_wd001_warmup500.json"

    print("=" * 100)
    print("Running FIXED DuQuant benchmark")
    print("Previous naive result was invalid because activation was transformed without weight compensation.")
    print("=" * 100)

    duq.main()


if __name__ == "__main__":
    main()
