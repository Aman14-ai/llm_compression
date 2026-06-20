import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


DATA_DIR = Path("data/tokenized/gpt2_agnews_maxlen128")
MODEL_NAME = "openai-community/gpt2"

OUTPUT_DIR = Path("models/gpt2_agnews_sanity")
RESULTS_DIR = Path("results")

NUM_LABELS = 4
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 1

MAX_TRAIN_SAMPLES = 5000
MAX_EVAL_SAMPLES = 1000

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clear_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def make_dataloader(split, batch_size, shuffle):
    return DataLoader(
        split,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )


def move_batch_to_device(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


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
        total_loss += loss.item() * bs
        total_samples += bs

        all_preds.extend(preds.detach().cpu().numpy().tolist())
        all_labels.extend(batch["labels"].detach().cpu().numpy().tolist())

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end = time.perf_counter()
    elapsed = end - start

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels,
        all_preds,
        average="weighted",
        zero_division=0,
    )

    return {
        "eval_loss": total_loss / max(total_samples, 1),
        "accuracy": accuracy,
        "precision_weighted": precision,
        "recall_weighted": recall,
        "f1_weighted": f1,
        "eval_samples": total_samples,
        "eval_time_sec": elapsed,
        "throughput_samples_per_sec": total_samples / max(elapsed, 1e-9),
        "latency_ms_per_sample": (elapsed / max(total_samples, 1)) * 1000,
    }


def main():
    seed_everything(42)
    clear_gpu_memory()

    print("=" * 100)
    print("GPT2 sanity training on AG News")
    print("=" * 100)

    print("Device:", DEVICE)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("Total VRAM GB:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 4))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\nLoading tokenized dataset:", DATA_DIR)
    dataset = load_from_disk(str(DATA_DIR))

    train_dataset = dataset["train"].select(range(MAX_TRAIN_SAMPLES))
    eval_dataset = dataset["test"].select(range(MAX_EVAL_SAMPLES))

    print("Train samples:", len(train_dataset))
    print("Eval samples:", len(eval_dataset))

    train_loader = make_dataloader(train_dataset, BATCH_SIZE, shuffle=True)
    eval_loader = make_dataloader(eval_dataset, BATCH_SIZE, shuffle=False)

    print("\nLoading tokenizer:", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("pad_token:", tokenizer.pad_token)
    print("pad_token_id:", tokenizer.pad_token_id)

    print("\nLoading GPT2 sequence classification model:", MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        pad_token_id=tokenizer.pad_token_id,
    )

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    model.to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    global_step = 0

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    print("\nStarting GPT2 sanity training...")
    train_start = time.perf_counter()

    for epoch in range(NUM_EPOCHS):
        model.train()

        epoch_loss = 0.0
        epoch_samples = 0

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

            bs = batch["labels"].size(0)
            epoch_loss += loss.item() * bs
            epoch_samples += bs
            global_step += 1

            progress.set_postfix({"loss": f"{loss.item():.4f}"})

        print(f"\nEpoch {epoch + 1} average training loss: {epoch_loss / max(epoch_samples, 1):.6f}")

        metrics = evaluate(model, eval_loader, DEVICE)
        print("\nEvaluation metrics after epoch:")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"{k}: {v:.6f}")
            else:
                print(f"{k}: {v}")

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    train_time_sec = time.perf_counter() - train_start

    print("\nFinal evaluation...")
    final_metrics = evaluate(model, eval_loader, DEVICE)

    final_metrics["train_time_sec"] = train_time_sec
    final_metrics["global_steps"] = global_step
    final_metrics["model_name"] = MODEL_NAME
    final_metrics["dataset"] = "fancyzhx/ag_news"
    final_metrics["train_samples"] = len(train_dataset)
    final_metrics["batch_size"] = BATCH_SIZE
    final_metrics["learning_rate"] = LEARNING_RATE
    final_metrics["num_epochs"] = NUM_EPOCHS

    if torch.cuda.is_available():
        final_metrics["max_gpu_memory_allocated_gb"] = round(torch.cuda.max_memory_allocated() / 1024**3, 4)
        final_metrics["max_gpu_memory_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 1024**3, 4)

    print("\nSaving GPT2 sanity model to:", OUTPUT_DIR)
    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    result_path = RESULTS_DIR / "gpt2_sanity_baseline_metrics.json"
    with open(result_path, "w") as f:
        json.dump(final_metrics, f, indent=2)

    print("\nSaved metrics to:", result_path)
    print("\nFinal metrics:")
    print(json.dumps(final_metrics, indent=2))

    del model
    del optimizer
    clear_gpu_memory()
    print("\nGPU cleanup called. Process will now exit.")


if __name__ == "__main__":
    main()
