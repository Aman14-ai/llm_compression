import argparse
import gc
import json
import math
import statistics
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import pynvml
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


# ======================================================================================
# Experiment framework
# ======================================================================================

EXP_TAG = "bs8_ep3_wd001_warmup500"

RESULTS_DIR = Path("results_updated")
MODELS_DIR = Path("models_updated")

DATASET_NAME = "nyu-mll/glue"
MAX_LENGTH = 128

BATCH_SIZE = 8
TRAIN_BATCH_SIZE = 8
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 500
NUM_EPOCHS = 1

CALIBRATION_SAMPLES = 512
CALIBRATION_BATCH_SIZE = 8

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

WARMUP_BATCHES = 10
POWER_SAMPLE_INTERVAL_SEC = 0.05

# SensiQuant without KD
HIGH_BITS = 8
LOW_BITS = 4
ACTIVATION_BITS = 8
LOW_BIT_FRACTION = 0.35


# ======================================================================================
# Model and dataset configs
# ======================================================================================

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
        "problem_type": "single_label_classification",
        "train_split": "train",
        "eval_split": "validation",
    },
    "qnli": {
        "display_name": "QNLI",
        "hf_config": "qnli",
        "text_fields": ("question", "sentence"),
        "num_labels": 2,
        "problem_type": "single_label_classification",
        "train_split": "train",
        "eval_split": "validation",
    },
    "mrpc": {
        "display_name": "MRPC",
        "hf_config": "mrpc",
        "text_fields": ("sentence1", "sentence2"),
        "num_labels": 2,
        "problem_type": "single_label_classification",
        "train_split": "train",
        "eval_split": "validation",
    },
    "rte": {
        "display_name": "RTE",
        "hf_config": "rte",
        "text_fields": ("sentence1", "sentence2"),
        "num_labels": 2,
        "problem_type": "single_label_classification",
        "train_split": "train",
        "eval_split": "validation",
    },
    "wnli": {
        "display_name": "WNLI",
        "hf_config": "wnli",
        "text_fields": ("sentence1", "sentence2"),
        "num_labels": 2,
        "problem_type": "single_label_classification",
        "train_split": "train",
        "eval_split": "validation",
    },
    "mnli": {
        "display_name": "MNLI",
        "hf_config": "mnli",
        "text_fields": ("premise", "hypothesis"),
        "num_labels": 3,
        "problem_type": "single_label_classification",
        "train_split": "train",
        "eval_split": "validation_matched",
    },
    "cola": {
        "display_name": "COLA",
        "hf_config": "cola",
        "text_fields": ("sentence",),
        "num_labels": 2,
        "problem_type": "single_label_classification",
        "train_split": "train",
        "eval_split": "validation",
    },
    "stsb": {
        "display_name": "STS-B",
        "hf_config": "stsb",
        "text_fields": ("sentence1", "sentence2"),
        "num_labels": 1,
        "problem_type": "regression",
        "train_split": "train",
        "eval_split": "validation",
    },
}


# ======================================================================================
# Basic utilities
# ======================================================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_key",
        default="all",
        choices=["all"] + list(MODEL_CONFIGS.keys()),
        help="Model key: bertbase, distilbert, mobilebert, albert, tinybert, or all",
    )

    parser.add_argument(
        "--dataset_key",
        default="all",
        choices=["all"] + list(DATASET_CONFIGS.keys()),
        help="Dataset key: sst2, qnli, mrpc, rte, wnli, mnli, cola, stsb, or all",
    )

    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip runs whose final JSON already exists.",
    )

    parser.add_argument(
        "--make_report_only",
        action="store_true",
        help="Only build combined report from existing JSON files.",
    )

    return parser.parse_args()


def selected_keys(value, config_dict):
    if value == "all":
        return list(config_dict.keys())
    return [value]


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


def pct(x):
    return round(float(x) * 100, 4)


# ======================================================================================
# Dataset tokenization
# ======================================================================================

def tokenize_dataset(model_name, dataset_cfg):
    hf_config = dataset_cfg["hf_config"]
    text_fields = dataset_cfg["text_fields"]

    print(f"\nLoading GLUE dataset: {hf_config}")
    dataset = load_dataset(DATASET_NAME, hf_config)

    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.sep_token is not None:
            tokenizer.pad_token = tokenizer.sep_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    def preprocess(batch):
        if len(text_fields) == 1:
            return tokenizer(
                batch[text_fields[0]],
                truncation=True,
                padding="max_length",
                max_length=MAX_LENGTH,
            )

        return tokenizer(
            batch[text_fields[0]],
            batch[text_fields[1]],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )

    tokenized = dataset.map(preprocess, batched=True)

    if "label" in tokenized[dataset_cfg["train_split"]].column_names:
        tokenized = tokenized.rename_column("label", "labels")

    columns = ["input_ids", "attention_mask", "labels"]

    train_columns = tokenized[dataset_cfg["train_split"]].column_names
    if "token_type_ids" in train_columns:
        columns.append("token_type_ids")

    tokenized.set_format(type="torch", columns=columns)

    return tokenized, tokenizer


# ======================================================================================
# Metrics
# ======================================================================================

def compute_metrics_from_predictions(pred_values, label_values, dataset_key):
    pred_values = np.asarray(pred_values)
    label_values = np.asarray(label_values)

    # STS-B is regression in GLUE.
    # Since your sheet has Accuracy/Precision/Recall/F1 columns,
    # we binarize scores at 2.5 only for table compatibility.
    if dataset_key == "stsb":
        pred_classes = (pred_values >= 2.5).astype(int)
        label_classes = (label_values >= 2.5).astype(int)
    else:
        pred_classes = pred_values.astype(int)
        label_classes = label_values.astype(int)

    accuracy = accuracy_score(label_classes, pred_classes)

    precision, recall, f1, _ = precision_recall_fscore_support(
        label_classes,
        pred_classes,
        average="weighted",
        zero_division=0,
    )

    return accuracy, precision, recall, f1


@torch.no_grad()
def evaluate(model, dataloader, device, dataset_key):
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

        bs = batch["labels"].size(0)
        total_loss += loss.item() * bs
        total_samples += bs

        if dataset_key == "stsb":
            preds = logits.squeeze(-1)
            all_preds.extend(preds.detach().cpu().float().numpy().tolist())
            all_labels.extend(batch["labels"].detach().cpu().float().numpy().tolist())
        else:
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.detach().cpu().numpy().tolist())
            all_labels.extend(batch["labels"].detach().cpu().numpy().tolist())

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    accuracy, precision, recall, f1 = compute_metrics_from_predictions(
        all_preds,
        all_labels,
        dataset_key,
    )

    return {
        "eval_loss": total_loss / max(total_samples, 1),
        "accuracy": accuracy,
        "precision_weighted": precision,
        "recall_weighted": recall,
        "f1_weighted": f1,
        "eval_samples": total_samples,
        "eval_time_sec": elapsed,
        "latency_ms_per_sample": elapsed / max(total_samples, 1) * 1000,
        "throughput_samples_per_sec": total_samples / max(elapsed, 1e-9),
    }


# ======================================================================================
# Training
# ======================================================================================

def train_or_load_model(
    tokenized,
    tokenizer,
    model_name,
    model_dir,
    dataset_key,
    dataset_cfg,
):
    if (model_dir / "config.json").exists():
        print("\nLoading existing fine-tuned model:", model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        return model

    print("\nFine-tuning model...")
    print("Model:", model_name)
    print("Saving to:", model_dir)

    model_dir.mkdir(parents=True, exist_ok=True)

    train_split = dataset_cfg["train_split"]
    eval_split = dataset_cfg["eval_split"]

    train_loader = make_dataloader(
        tokenized[train_split],
        TRAIN_BATCH_SIZE,
        shuffle=True,
    )

    val_loader = make_dataloader(
        tokenized[eval_split],
        BATCH_SIZE,
        shuffle=False,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=dataset_cfg["num_labels"],
        problem_type=dataset_cfg["problem_type"],
        ignore_mismatched_sizes=True,
    )

    if tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    if len(tokenizer) > model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(len(tokenizer))

    model.to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    num_training_steps = len(train_loader) * NUM_EPOCHS

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=num_training_steps,
    )

    print("Optimizer: AdamW")
    print("Train batch size:", TRAIN_BATCH_SIZE)
    print("Eval batch size:", BATCH_SIZE)
    print("Epochs:", NUM_EPOCHS)
    print("Learning rate:", LEARNING_RATE)
    print("Weight decay:", WEIGHT_DECAY)
    print("Warmup steps:", WARMUP_STEPS)
    print("Total training steps:", num_training_steps)

    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    train_start = time.perf_counter()
    global_step = 0

    for epoch in range(NUM_EPOCHS):
        model.train()

        epoch_loss = 0.0
        epoch_samples = 0

        progress = tqdm(
            train_loader,
            desc=f"Training epoch {epoch + 1}/{NUM_EPOCHS}",
        )

        for batch in progress:
            batch = move_batch_to_device(batch, DEVICE)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(**batch)
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            bs = batch["labels"].size(0)
            epoch_loss += loss.item() * bs
            epoch_samples += bs
            global_step += 1

            progress.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = epoch_loss / max(epoch_samples, 1)
        print(f"\nEpoch {epoch + 1} average training loss: {avg_loss:.6f}")

        metrics = evaluate(model, val_loader, DEVICE, dataset_key)
        print("\nValidation after epoch:")
        print(json.dumps(metrics, indent=2))

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    train_time_sec = time.perf_counter() - train_start

    print("\nSaving fine-tuned model:", model_dir)
    model.save_pretrained(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))

    train_metadata = {
        "train_time_sec": train_time_sec,
        "global_steps": global_step,
        "train_batch_size": TRAIN_BATCH_SIZE,
        "eval_batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "warmup_steps": WARMUP_STEPS,
        "optimizer": "AdamW",
        "scheduler": "linear_schedule_with_warmup",
        "num_epochs": NUM_EPOCHS,
    }

    with open(model_dir / "training_metadata.json", "w") as f:
        json.dump(train_metadata, f, indent=2)

    del optimizer
    clear_memory()

    model.to("cpu")
    return model


# ======================================================================================
# SensiQuant layers
# ======================================================================================

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
        qweight, scale = quantize_weight_per_output_channel(
            linear.weight,
            weight_bits,
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
    print("\nEstimating SensiQuant sensitivity...")

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
        # Keep task heads safer at high precision.
        if "classifier" in name or "score" in name or "pooler" in name:
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

    for _, param in model.named_parameters():
        total_bytes += tensor_nbytes(param)

    for name, buffer in model.named_buffers():
        belongs_to_quantized = any(
            name.startswith(prefix + ".")
            for prefix in quantized_prefixes
        )

        if not belongs_to_quantized:
            total_bytes += tensor_nbytes(buffer)

    return total_bytes / 1024**2


# ======================================================================================
# Power monitor
# ======================================================================================

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
            "nvml_available": self.available,
            "nvml_error": self.error,
            "energy_joules": energy,
            "avg_power_watts": statistics.mean(powers),
            "peak_nvml_memory_used_gb": max(memories),
            "num_power_samples": len(self.samples),
        }


# ======================================================================================
# Quantized benchmark
# ======================================================================================

@torch.inference_mode()
def benchmark_quantized(
    model,
    dataloader,
    model_display_name,
    dataset_display_name,
    dataset_key,
    model_dir,
):
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

    print(f"\nRunning {WARMUP_BATCHES} warmup batches...")

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

        loss = outputs.loss
        logits = outputs.logits

        bs = batch["labels"].size(0)
        total_samples += bs
        total_loss += loss.item() * bs

        if dataset_key == "stsb":
            preds = logits.squeeze(-1)
            all_preds.extend(preds.detach().cpu().float().numpy().tolist())
            all_labels.extend(batch["labels"].detach().cpu().float().numpy().tolist())
        else:
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.detach().cpu().numpy().tolist())
            all_labels.extend(batch["labels"].detach().cpu().numpy().tolist())

        batch_latencies.append((bend - bstart) * 1000)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    monitor.stop()
    power = monitor.summarize()

    accuracy, precision, recall, f1 = compute_metrics_from_predictions(
        all_preds,
        all_labels,
        dataset_key,
    )

    result = {
        "model": model_display_name,
        "dataset": dataset_display_name,
        "method": "SensiQuant without Knowledge Distillation",
        "experiment_tag": EXP_TAG,
        "training_batch_size": TRAIN_BATCH_SIZE,
        "training_epochs": NUM_EPOCHS,
        "training_optimizer": "AdamW",
        "training_weight_decay": WEIGHT_DECAY,
        "training_warmup_steps": WARMUP_STEPS,
        "training_learning_rate": LEARNING_RATE,
        "model_dir": str(model_dir),
        "weight_bits_sensitive_layers": HIGH_BITS,
        "weight_bits_less_sensitive_layers": LOW_BITS,
        "activation_bits": ACTIVATION_BITS,
        "batch_size": BATCH_SIZE,
        "warmup_batches": WARMUP_BATCHES,
        "eval_samples": total_samples,
        "eval_loss": total_loss / max(total_samples, 1),
        "accuracy": accuracy,
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

    if result["energy_joules"] is not None:
        result["energy_joules_per_sample"] = result["energy_joules"] / max(total_samples, 1)
    else:
        result["energy_joules_per_sample"] = None

    return result


# ======================================================================================
# Run one row
# ======================================================================================

def make_run_paths(model_key, dataset_key):
    run_tag = f"{model_key}_{dataset_key}_sensiquant_without_kd_{EXP_TAG}"

    model_dir = MODELS_DIR / f"{model_key}_{dataset_key}_full_{EXP_TAG}"

    output_json = RESULTS_DIR / f"{run_tag}.json"
    output_csv = RESULTS_DIR / f"{run_tag}_table.csv"

    return run_tag, model_dir, output_json, output_csv


def run_one(model_key, dataset_key, skip_existing=False):
    seed_everything(42)
    clear_memory()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model_cfg = MODEL_CONFIGS[model_key]
    dataset_cfg = DATASET_CONFIGS[dataset_key]

    model_display_name = model_cfg["display_name"]
    dataset_display_name = dataset_cfg["display_name"]
    model_name = model_cfg["hf_name"]

    run_tag, model_dir, output_json, output_csv = make_run_paths(
        model_key,
        dataset_key,
    )

    if skip_existing and output_json.exists():
        print(f"\nSkipping existing run: {run_tag}")
        return

    print("=" * 100)
    print("SensiQuant without Knowledge Distillation")
    print("Model:", model_display_name)
    print("Dataset:", dataset_display_name)
    print("Run tag:", run_tag)
    print("=" * 100)

    print("Device:", DEVICE)

    if DEVICE == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        print(
            "Total VRAM GB:",
            round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 4),
        )

    tokenized, tokenizer = tokenize_dataset(model_name, dataset_cfg)

    eval_split = dataset_cfg["eval_split"]
    train_split = dataset_cfg["train_split"]

    val_loader = make_dataloader(
        tokenized[eval_split],
        BATCH_SIZE,
        shuffle=False,
    )

    calibration_count = min(CALIBRATION_SAMPLES, len(tokenized[train_split]))

    calibration_dataset = tokenized[train_split].select(range(calibration_count))

    calibration_loader = make_dataloader(
        calibration_dataset,
        CALIBRATION_BATCH_SIZE,
        shuffle=False,
    )

    model = train_or_load_model(
        tokenized=tokenized,
        tokenizer=tokenizer,
        model_name=model_name,
        model_dir=model_dir,
        dataset_key=dataset_key,
        dataset_cfg=dataset_cfg,
    )

    baseline_storage = estimate_model_storage_mb(model)
    print("\nBaseline storage MB:", baseline_storage)

    sensitivity = estimate_sensitivity(model, calibration_loader)
    assignment, sorted_layers = assign_sensiquant_bits(sensitivity)

    print("\nTop sensitive layers:")
    for name in sorted_layers[:5]:
        print(name, sensitivity[name], assignment[name])

    print("\nBottom sensitive layers:")
    for name in sorted_layers[-5:]:
        print(name, sensitivity[name], assignment[name])

    model.to("cpu")
    clear_memory()

    replaced = replace_linear_layers(model, assignment)

    actual_storage = estimate_model_storage_mb(model)
    effective_storage = estimate_effective_storage_mb(model)

    bit_counts = {}
    for item in replaced:
        bit_counts[str(item["bits"])] = bit_counts.get(str(item["bits"]), 0) + 1

    print("\nReplaced Linear layers:", len(replaced))
    print("Bit counts:", bit_counts)
    print("Actual storage MB:", actual_storage)
    print("Effective packed storage MB:", effective_storage)

    model.to(DEVICE)
    model.eval()

    result = benchmark_quantized(
        model=model,
        dataloader=val_loader,
        model_display_name=model_display_name,
        dataset_display_name=dataset_display_name,
        dataset_key=dataset_key,
        model_dir=model_dir,
    )

    result["baseline_storage_mb_before_quantization"] = baseline_storage
    result["actual_storage_mb_after_quantization"] = actual_storage
    result["effective_packed_storage_mb_after_quantization"] = effective_storage
    result["compression_ratio"] = baseline_storage / max(effective_storage, 1e-9)
    result["num_replaced_linear_layers"] = len(replaced)
    result["bit_assignment_counts"] = bit_counts
    result["calibration_samples"] = calibration_count

    with open(output_json, "w") as f:
        json.dump(result, f, indent=2)

    row = {
        "Dataset": dataset_display_name,
        "Model Name": model_display_name,
        "Accuracy": pct(result["accuracy"]),
        "GPU Memory": round(float(result["peak_nvml_memory_used_gb"]), 4),
        "Latency": round(float(result["latency_ms_per_sample"]), 4),
        "Precision": pct(result["precision_weighted"]),
        "Recall": pct(result["recall_weighted"]),
        "F1 Score": pct(result["f1_weighted"]),
        "Throughput": round(float(result["throughput_samples_per_sec"]), 4),
        "Energy": round(float(result["energy_joules"]), 4),
    }

    df = pd.DataFrame([row])
    df.to_csv(output_csv, index=False)

    print("\nFinal table row:")
    print(df.to_string(index=False))

    print("\nSaved JSON:", output_json)
    print("Saved CSV:", output_csv)

    del model
    clear_memory()


# ======================================================================================
# Build combined report
# ======================================================================================

def build_combined_report():
    rows = []

    for dataset_key, dataset_cfg in DATASET_CONFIGS.items():
        for model_key, model_cfg in MODEL_CONFIGS.items():
            _, _, output_json, _ = make_run_paths(model_key, dataset_key)

            if not output_json.exists():
                print("Missing:", output_json)
                continue

            with open(output_json, "r") as f:
                result = json.load(f)

            rows.append({
                "Dataset": dataset_cfg["display_name"],
                "Model Name": model_cfg["display_name"],
                "Accuracy": pct(result["accuracy"]),
                "GPU Memory": round(float(result["peak_nvml_memory_used_gb"]), 4),
                "Latency": round(float(result["latency_ms_per_sample"]), 4),
                "Precision": pct(result["precision_weighted"]),
                "Recall": pct(result["recall_weighted"]),
                "F1 Score": pct(result["f1_weighted"]),
                "Throughput": round(float(result["throughput_samples_per_sec"]), 4),
                "Energy": round(float(result["energy_joules"]), 4),
            })

    if not rows:
        print("No result JSON files found. Run experiments first.")
        return

    df = pd.DataFrame(rows)

    output_csv = RESULTS_DIR / f"sensiquant_without_kd_all_glue_report_{EXP_TAG}.csv"
    output_xlsx = RESULTS_DIR / f"sensiquant_without_kd_all_glue_report_{EXP_TAG}.xlsx"

    df.to_csv(output_csv, index=False)
    df.to_excel(output_xlsx, index=False)

    print("\nCombined report:")
    print(df.to_string(index=False))

    print("\nSaved combined CSV:", output_csv)
    print("Saved combined Excel:", output_xlsx)


# ======================================================================================
# Main
# ======================================================================================

def main():
    args = parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if args.make_report_only:
        build_combined_report()
        return

    model_keys = selected_keys(args.model_key, MODEL_CONFIGS)
    dataset_keys = selected_keys(args.dataset_key, DATASET_CONFIGS)

    for dataset_key in dataset_keys:
        for model_key in model_keys:
            run_one(
                model_key=model_key,
                dataset_key=dataset_key,
                skip_existing=args.skip_existing,
            )

    build_combined_report()


if __name__ == "__main__":
    main()