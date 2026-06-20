import gc
import json
import statistics
import threading
import time
from pathlib import Path

import numpy as np
import torch
import pynvml
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification


MODEL_DIR = Path("models_updated/gpt2_agnews_bs8_ep3_wd001_warmup500")
DATA_DIR = Path("data/tokenized/gpt2_agnews_maxlen128")
RESULTS_DIR = Path("results_updated")

OUTPUT_JSON = RESULTS_DIR / "gpt2_baseline_inference_benchmark_bs8_ep3_wd001_warmup500.json"

BATCH_SIZE = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

WARMUP_BATCHES = 10

# GPU power sampling interval for energy estimation.
POWER_SAMPLE_INTERVAL_SEC = 0.05


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


class NvmlPowerMonitor:
    """
    Samples NVIDIA GPU power and memory in a background thread.

    Energy approximation:
        E = integral(P dt)

    Power is sampled from NVML in watts.
    Time is measured in seconds.
    Energy is reported in joules.
    """

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

        # Trapezoidal integration over power samples
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

    for batch in tqdm(dataloader, desc="Benchmarking inference"):
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
        "model": "GPT2",
        "dataset": "AG News",
        "method": "Baseline_Internal_Not_For_Final_Table",
        "model_dir": str(MODEL_DIR),
        "data_dir": str(DATA_DIR),
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

    print("=" * 100)
    print("GPT2 AG News inference benchmark - bs8 ep3 wd0.01 warmup500")
    print("=" * 100)
    print("Device:", DEVICE)

    if DEVICE == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        print("Total VRAM GB:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 4))

    print("\nLoading tokenized dataset:", DATA_DIR)
    dataset = load_from_disk(str(DATA_DIR))
    test_dataset = dataset["test"]
    print("Test samples:", len(test_dataset))

    dataloader = make_dataloader(test_dataset, BATCH_SIZE)

    print("\nLoading model:", MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))

    # GPT2 needs pad_token_id for sequence classification batches.
    # 50256 is GPT2's EOS token id, used as PAD in our tokenized dataset.
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = 50256

    model.config.use_cache = False
    model.to(DEVICE)
    model.eval()

    # Warmup uses same dataloader, but warmup batches are not counted.
    warmup_model(model, dataloader, DEVICE, WARMUP_BATCHES)

    result = benchmark(model, dataloader, DEVICE)

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
