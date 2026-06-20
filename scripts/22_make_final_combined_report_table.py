import json
from pathlib import Path

import pandas as pd


RESULTS_DIR = Path("results")

ROWS = [
    # DistilBERT order
    ("DistilBERT", "AG News", "HAQ", RESULTS_DIR / "distilbert_haq_inference_benchmark.json"),
    ("DistilBERT", "AG News", "HAWQ", RESULTS_DIR / "distilbert_hawq_inference_benchmark.json"),
    ("DistilBERT", "AG News", "Sensimix", RESULTS_DIR / "distilbert_sensimix_inference_benchmark.json"),
    ("DistilBERT", "AG News", "Zeroquant", RESULTS_DIR / "distilbert_zeroquant_inference_benchmark.json"),
    ("DistilBERT", "AG News", "Duquant", RESULTS_DIR / "distilbert_duquant_inference_benchmark.json"),

    # GPT2 order
    ("GPT2", "AG News", "HAQ", RESULTS_DIR / "gpt2_haq_inference_benchmark.json"),
    ("GPT2", "AG News", "HAWQ", RESULTS_DIR / "gpt2_hawq_inference_benchmark.json"),
    ("GPT2", "AG News", "Sensimix", RESULTS_DIR / "gpt2_sensimix_inference_benchmark.json"),
    ("GPT2", "AG News", "Duquant", RESULTS_DIR / "gpt2_duquant_inference_benchmark.json"),
    ("GPT2", "AG News", "Zeroquant", RESULTS_DIR / "gpt2_zeroquant_inference_benchmark.json"),
]

OUTPUT_CSV = RESULTS_DIR / "final_agnews_compression_report_table.csv"
OUTPUT_XLSX = RESULTS_DIR / "final_agnews_compression_report_table.xlsx"


def pct(x):
    return round(float(x) * 100, 4)


def main():
    rows = []

    for model_name, dataset_name, method_name, path in ROWS:
        if not path.exists():
            raise FileNotFoundError(f"Missing result file: {path}")

        with open(path, "r") as f:
            data = json.load(f)

        row = {
            "Model Name": model_name,
            "Dataset": dataset_name,
            "Method Name": method_name,
            "Accuracy": pct(data["accuracy"]),
            "GPU Memory": round(float(data["peak_nvml_memory_used_gb"]), 4),
            "Latency": round(float(data["latency_ms_per_sample"]), 4),
            "Precision": pct(data["precision_weighted"]),
            "Recall": pct(data["recall_weighted"]),
            "F1 Score": pct(data["f1_weighted"]),
            "Throughput": round(float(data["throughput_samples_per_sec"]), 4),
            "Energy": round(float(data["energy_joules"]), 4),
        }

        rows.append(row)

    df = pd.DataFrame(rows)

    df.to_csv(OUTPUT_CSV, index=False)
    df.to_excel(OUTPUT_XLSX, index=False)

    print("\nFinal AG News compression report table:")
    print(df.to_string(index=False))

    print("\nSaved CSV:", OUTPUT_CSV)
    print("Saved Excel:", OUTPUT_XLSX)

    print("\nUnits:")
    print("Accuracy, Precision, Recall, F1 Score: percentage (%)")
    print("GPU Memory: GB, peak NVML GPU memory")
    print("Latency: milliseconds per sample")
    print("Throughput: samples per second")
    print("Energy: joules for full AG News test inference")


if __name__ == "__main__":
    main()
