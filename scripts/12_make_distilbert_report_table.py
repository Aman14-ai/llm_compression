import json
from pathlib import Path

import pandas as pd


RESULTS_DIR = Path("results")

FILES = {
    "HAQ": RESULTS_DIR / "distilbert_haq_inference_benchmark.json",
    "HAWQ": RESULTS_DIR / "distilbert_hawq_inference_benchmark.json",
    "Sensimix": RESULTS_DIR / "distilbert_sensimix_inference_benchmark.json",
    "Zeroquant": RESULTS_DIR / "distilbert_zeroquant_inference_benchmark.json",
    "Duquant": RESULTS_DIR / "distilbert_duquant_inference_benchmark.json",
}

OUTPUT_CSV = RESULTS_DIR / "distilbert_agnews_report_table.csv"
OUTPUT_XLSX = RESULTS_DIR / "distilbert_agnews_report_table.xlsx"


def pct(x):
    return round(float(x) * 100, 4)


def main():
    rows = []

    for method_name, path in FILES.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing result file for {method_name}: {path}")

        with open(path, "r") as f:
            data = json.load(f)

        row = {
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

    print("\nDistilBERT AG News report table:")
    print(df.to_string(index=False))

    print("\nSaved CSV:", OUTPUT_CSV)
    print("Saved Excel:", OUTPUT_XLSX)

    print("\nUnits used:")
    print("Accuracy, Precision, Recall, F1 Score: percentage (%)")
    print("GPU Memory: GB, using peak NVML GPU memory")
    print("Latency: milliseconds per sample")
    print("Throughput: samples per second")
    print("Energy: total joules for full AG News test inference")


if __name__ == "__main__":
    main()
