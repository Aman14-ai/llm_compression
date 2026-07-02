import argparse

import gc
import json
import math
import statistics
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pynvml
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MODEL_CONFIGS = {
    "bertbase": {
        "display_name": "BERT-Base",
        "hf_name": "google-bert/bert-base-uncased",
    },
    "distilbert": {
        "display_name": "DistilBERT",
        "hf_name": "distilbert/distilbert-base-uncased",
    },
    "mobilebert": {
        "display_name": "MobileBERT",
        "hf_name": "google/mobilebert-uncased",
    },
    "albert": {
        "display_name": "ALBERT",
        "hf_name": "albert/albert-base-v2",
    },
    "tinybert": {
        "display_name": "TinyBERT",
        "hf_name": "huawei-noah/TinyBERT_General_4L_312D",
    },
}

DATASET_CONFIGS = {
    "sst2": {
        "display_name": "SST2",
        "hf_config": "sst2",
        "text_fields": ("sentence",),
        "num_labels": 2,
    },
    "qnli": {
        "display_name": "QNLI",
        "hf_config": "qnli",
        "text_fields": ("question", "sentence"),
        "num_labels": 2,
    },
    "mrpc": {
        "display_name": "MRPC",
        "hf_config": "mrpc",
        "text_fields": ("sentence1", "sentence2"),
        "num_labels": 2,
    },
}

RESULTS_DIR = Path("results_updated")
MODELS_DIR = Path("models_updated")

DATASET_NAME = "nyu-mll/glue"

MAX_LENGTH = 128

BATCH_SIZE = 8
TRAIN_BATCH_SIZE = 8
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 500
NUM_EPOCHS = 3

CALIBRATION_SAMPLES = 512
CALIBRATION_BATCH_SIZE = 8


OUTPUT_JSON = RESULTS_DIR / "bertbase_sst2_sensiquant_without_kd.json"
OUTPUT_CSV = RESULTS_DIR / "bertbase_sst2_sensiquant_without_kd_table.csv"

MODEL_NAME = "google-bert/bert-base-uncased"
DATASET_NAME = "nyu-mll/glue"
DATASET_CONFIG = "sst2"

MAX_LENGTH = 128
NUM_LABELS = 2

BATCH_SIZE = 32
TRAIN_BATCH_SIZE = 32
LEARNING_RATE = 2e-5
NUM_EPOCHS = 1

CALIBRATION_SAMPLES = 512
CALIBRATION_BATCH_SIZE = 16

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

WARMUP_BATCHES = 10
POWER_SAMPLE_INTERVAL_SEC = 0.05

# SensiQuant without KD:
# no teacher model, no distillation loss.
# We use sensitivity-aware mixed precision:
# sensitive Linear layers -> INT8
# less-sensitive Linear layers -> INT4
HIGH_BITS = 8
LOW_BITS = 4
ACTIVATION_BITS = 8
LOW_BIT_FRACTION = 0.35


def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def move_batch_to_device(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def tokenize_dataset():
    print("Loading SST2 dataset...")
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def preprocess(batch):
        return tokenizer(
            batch["sentence"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )

    tokenized = dataset.map(preprocess, batched=True)
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )

    return tokenized, tokenizer


def make_dataloader(dataset, batch_size, shuffle=False):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )


def tensor_nbytes(tensor):
    return tensor.numel() * tensor.element_size()


def estimate_model_storage_mb(model):
    total = 0
    for p in model.parameters():
        total += tensor_nbytes(p)
    for b in model.buffers():
        total += tensor_nbytes(b)
    return total / 1024**2


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()

    all_preds = []
    all_labels = []
    total_loss = 0.0
    total_samples = 0

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.perf_counter()

    for batch in tqdm(dataloader, desc="Evaluating"):
        batch = move_batch_to_device(batch, device)
        outputs = model(**batch)

        loss = outputs.loss
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)

        bs = batch["labels"].size(0)
        total_samples += bs
        total_loss += loss.item() * bs

        all_preds.extend(preds.detach().cpu().tolist())
        all_labels.extend(batch["labels"].detach().cpu().tolist())

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end = time.perf_counter()
    elapsed = end - start

    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels,
        all_preds,
        average="weighted",
        zero_division=0,
    )

    return {
        "eval_loss": total_loss / max(total_samples, 1),
        "accuracy": acc,
        "precision_weighted": precision,
        "recall_weighted": recall,
        "f1_weighted": f1,
        "eval_samples": total_samples,
        "eval_time_sec": elapsed,
        "latency_ms_per_sample": elapsed / max(total_samples, 1) * 1000,
        "throughput_samples_per_sec": total_samples / max(elapsed, 1e-9),
    }


def train_or_load_model(tokenized, tokenizer):
    if MODEL_DIR.exists():
        print("Loading existing fine-tuned BERT-Base model:", MODEL_DIR)
        model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
        return model

    print("Fine-tuning BERT-Base on SST2...")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    train_loader = make_dataloader(tokenized["train"], TRAIN_BATCH_SIZE, shuffle=True)
    val_loader = make_dataloader(tokenized["validation"], BATCH_SIZE, shuffle=False)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
    )
    model.to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    for epoch in range(NUM_EPOCHS):
        model.train()
        progress = tqdm(train_loader, desc=f"Training epoch {epoch + 1}/{NUM_EPOCHS}")

        for batch in progress:
            batch = move_batch_to_device(batch, DEVICE)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(**batch)
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            progress.set_postfix({"loss": f"{loss.item():.4f}"})

        metrics = evaluate(model, val_loader, DEVICE)
        print("Validation after epoch:", json.dumps(metrics, indent=2))

    print("Saving fine-tuned BERT-Base model:", MODEL_DIR)
    model.save_pretrained(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))

    del optimizer
    clear_memory()

    model.to("cpu")
    return model


def fake_quantize_activation(x, bits):
    if bits is None:
        return x

    qmax = (2 ** (bits - 1)) - 1
    max_abs = x.detach().abs().amax(dim=-1, keepdim=True)
    scale = torch.clamp(max_abs / qmax, min=1e-8)

    qx = torch.round(x / scale)
    qx = torch.clamp(qx, min=-qmax, max=qmax)
    return qx * scale


def quantize_weight_per_output_channel(weight, bits):
    qmax = (2 ** (bits - 1)) - 1
    w = weight.detach().float().cpu()

    max_abs = w.abs().amax(dim=1, keepdim=True)
    scale = torch.clamp(max_abs / qmax, min=1e-8)

    qweight = torch.round(w / scale)
    qweight = torch.clamp(qweight, min=-qmax, max=qmax).to(torch.int8)

    return qweight, scale


class SensiQuantLinear(nn.Module):
    def __init__(self, qweight, scale, bias, weight_bits, activation_bits):
        super().__init__()

        self.weight_bits = int(weight_bits)
        self.activation_bits = int(activation_bits)

        self.register_buffer("qweight", qweight.contiguous())
        self.register_buffer("scale", scale.contiguous())

        if bias is not None:
            self.register_buffer("bias", bias.detach().float().cpu().contiguous())
        else:
            self.bias = None

    @classmethod
    def from_float(cls, linear, weight_bits, activation_bits):
        qweight, scale = quantize_weight_per_output_channel(linear.weight, weight_bits)
        bias = linear.bias.detach().clone() if linear.bias is not None else None

        return cls(qweight, scale, bias, weight_bits, activation_bits)

    def forward(self, x):
        x = fake_quantize_activation(x, self.activation_bits)

        weight = self.qweight.to(dtype=x.dtype) * self.scale.to(dtype=x.dtype)

        bias = None
        if self.bias is not None:
            bias = self.bias.to(dtype=x.dtype)

        return F.linear(x, weight, bias)


def get_linear_layer_names(model):
    return [
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
    ]


def get_parent_module(root, module_name):
    parts = module_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def estimate_sensitivity(model, calibration_loader):
    print("Estimating sensitivity for SensiQuant without KD...")

    model.to(DEVICE)
    model.eval()
    model.zero_grad(set_to_none=True)

    linear_names = get_linear_layer_names(model)
    named_params = dict(model.named_parameters())

    sensitivity_sum = {name: 0.0 for name in linear_names}
    sensitivity_count = {name: 0 for name in linear_names}

    for batch in tqdm(calibration_loader, desc="Calibration"):
        batch = move_batch_to_device(batch, DEVICE)

        model.zero_grad(set_to_none=True)

        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        for name in linear_names:
            param = named_params.get(f"{name}.weight")
            if param is None or param.grad is None:
                continue

            score = param.grad.detach().float().pow(2).mean().item()
            sensitivity_sum[name] += score
            sensitivity_count[name] += 1

    sensitivity = {}
    for name in linear_names:
        sensitivity[name] = sensitivity_sum[name] / max(sensitivity_count[name], 1)

    model.zero_grad(set_to_none=True)
    return sensitivity


def assign_sensiquant_bits(sensitivity):
    sorted_layers = sorted(
        sensitivity.keys(),
        key=lambda name: sensitivity[name],
        reverse=True,
    )

    n = len(sorted_layers)
    n_low = math.floor(LOW_BIT_FRACTION * n)
    low_set = set(sorted_layers[-n_low:]) if n_low > 0 else set()

    assignment = {}

    for name in sorted_layers:
        if "classifier" in name or "pooler" in name:
            assignment[name] = HIGH_BITS
        elif name in low_set:
            assignment[name] = LOW_BITS
        else:
            assignment[name] = HIGH_BITS

    return assignment, sorted_layers


def replace_linear_layers(model, assignment):
    replaced = []

    for layer_name, bits in assignment.items():
        parent, child_name = get_parent_module(model, layer_name)
        child = getattr(parent, child_name)

        if not isinstance(child, nn.Linear):
            continue

        new_layer = SensiQuantLinear.from_float(
            child,
            weight_bits=bits,
            activation_bits=ACTIVATION_BITS,
        )

        setattr(parent, child_name, new_layer)
        replaced.append({"name": layer_name, "bits": bits})

    return replaced


def estimate_effective_storage_mb(model):
    total_bytes = 0.0
    quantized_prefixes = []

    for module_name, module in model.named_modules():
        if isinstance(module, SensiQuantLinear):
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
                "energy_joules": None,
                "avg_power_watts": None,
                "peak_nvml_memory_used_gb": None,
                "num_power_samples": 0,
            }

        energy = 0.0
        for i in range(1, len(self.samples)):
            t0, p0, _ = self.samples[i - 1]
            t1, p1, _ = self.samples[i]
            energy += ((p0 + p1) / 2.0) * (t1 - t0)

        powers = [p for _, p, _ in self.samples]
        memories = [m for _, _, m in self.samples]

        return {
            "energy_joules": energy,
            "avg_power_watts": statistics.mean(powers),
            "peak_nvml_memory_used_gb": max(memories),
            "num_power_samples": len(self.samples),
        }


@torch.inference_mode()
def benchmark_quantized(model, dataloader):
    model.eval()

    all_preds = []
    all_labels = []
    total_loss = 0.0
    total_samples = 0
    batch_latencies = []

    clear_memory()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    print(f"Running {WARMUP_BATCHES} warmup batches...")
    for idx, batch in enumerate(dataloader):
        if idx >= WARMUP_BATCHES:
            break
        batch = move_batch_to_device(batch, DEVICE)
        _ = model(**batch)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    monitor = NvmlPowerMonitor(interval_sec=POWER_SAMPLE_INTERVAL_SEC)
    monitor.start()

    start = time.perf_counter()

    for batch in tqdm(dataloader, desc="Benchmarking SensiQuant without KD"):
        batch = move_batch_to_device(batch, DEVICE)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        bstart = time.perf_counter()
        outputs = model(**batch)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        bend = time.perf_counter()

        logits = outputs.logits
        loss = outputs.loss
        preds = torch.argmax(logits, dim=-1)

        bs = batch["labels"].size(0)
        total_samples += bs
        total_loss += loss.item() * bs

        all_preds.extend(preds.detach().cpu().tolist())
        all_labels.extend(batch["labels"].detach().cpu().tolist())

        batch_latencies.append((bend - bstart) * 1000)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end = time.perf_counter()

    monitor.stop()
    power = monitor.summarize()

    elapsed = end - start

    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels,
        all_preds,
        average="weighted",
        zero_division=0,
    )

    result = {
        "model": "BERT-Base",
        "dataset": "SST2",
        "method": "SensiQuant without Knowledge Distillation",
        "weight_bits_sensitive_layers": HIGH_BITS,
        "weight_bits_less_sensitive_layers": LOW_BITS,
        "activation_bits": ACTIVATION_BITS,
        "eval_samples": total_samples,
        "eval_loss": total_loss / max(total_samples, 1),
        "accuracy": acc,
        "precision_weighted": precision,
        "recall_weighted": recall,
        "f1_weighted": f1,
        "total_inference_time_sec": elapsed,
        "latency_ms_per_sample": elapsed / max(total_samples, 1) * 1000,
        "throughput_samples_per_sec": total_samples / max(elapsed, 1e-9),
        "mean_batch_latency_ms": statistics.mean(batch_latencies),
        "median_batch_latency_ms": statistics.median(batch_latencies),
    }

    if torch.cuda.is_available():
        result["peak_torch_memory_allocated_gb"] = torch.cuda.max_memory_allocated() / 1024**3
        result["peak_torch_memory_reserved_gb"] = torch.cuda.max_memory_reserved() / 1024**3
        result["gpu_name"] = torch.cuda.get_device_name(0)

    result.update(power)

    return result


def pct(x):
    return round(float(x) * 100, 4)


def main():
    seed_everything(42)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("BERT-Base SST2 SensiQuant without Knowledge Distillation")
    print("=" * 100)

    print("Device:", DEVICE)
    if DEVICE == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    tokenized, tokenizer = tokenize_dataset()

    val_loader = make_dataloader(tokenized["validation"], BATCH_SIZE, shuffle=False)
    calibration_dataset = tokenized["train"].select(range(CALIBRATION_SAMPLES))
    calibration_loader = make_dataloader(calibration_dataset, CALIBRATION_BATCH_SIZE, shuffle=False)

    model = train_or_load_model(tokenized, tokenizer)

    baseline_storage = estimate_model_storage_mb(model)
    print("Baseline storage MB:", baseline_storage)

    sensitivity = estimate_sensitivity(model, calibration_loader)
    assignment, sorted_layers = assign_sensiquant_bits(sensitivity)

    print("Top sensitive layers:")
    for name in sorted_layers[:5]:
        print(name, sensitivity[name], assignment[name])

    print("Bottom sensitive layers:")
    for name in sorted_layers[-5:]:
        print(name, sensitivity[name], assignment[name])

    model.to("cpu")
    clear_memory()

    replaced = replace_linear_layers(model, assignment)
    actual_storage = estimate_model_storage_mb(model)
    effective_storage = estimate_effective_storage_mb(model)

    bit_counts = {}
    for x in replaced:
        bit_counts[x["bits"]] = bit_counts.get(x["bits"], 0) + 1

    print("Replaced Linear layers:", len(replaced))
    print("Bit counts:", bit_counts)
    print("Actual storage MB:", actual_storage)
    print("Effective packed storage MB:", effective_storage)

    model.to(DEVICE)
    model.eval()

    result = benchmark_quantized(model, val_loader)

    result["baseline_storage_mb_before_quantization"] = baseline_storage
    result["actual_storage_mb_after_quantization"] = actual_storage
    result["effective_packed_storage_mb_after_quantization"] = effective_storage
    result["compression_ratio"] = baseline_storage / max(effective_storage, 1e-9)
    result["num_replaced_linear_layers"] = len(replaced)
    result["bit_assignment_counts"] = bit_counts

    with open(OUTPUT_JSON, "w") as f:
        json.dump(result, f, indent=2)

    row = {
        "Model Name": "BERT-Base",
        "Accuracy": pct(result["accuracy"]),
        "GPU Memory": round(float(result["peak_nvml_memory_used_gb"]), 4),
        "Latency": round(float(result["latency_ms_per_sample"]), 4),
        "Precision": pct(result["precision_weighted"]),
        "Recall": pct(result["recall_weighted"]),
        "F1 Score": pct(result["f1_weighted"]),
        "Throughput": round(float(result["throughput_samples_per_sec"]), 4),
        "Energy": round(float(result["energy_joules"]), 4),
    }

    import pandas as pd
    df = pd.DataFrame([row])
    df.to_csv(OUTPUT_CSV, index=False)

    print("\nFinal table row:")
    print(df.to_string(index=False))

    print("\nSaved JSON:", OUTPUT_JSON)
    print("Saved CSV:", OUTPUT_CSV)

    del model
    clear_memory()


if __name__ == "__main__":
    main()
