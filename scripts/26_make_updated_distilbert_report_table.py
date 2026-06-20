import json
from pathlib import Path

import pandas as pd


RESULTS_DIR = Path("results_updated")

FILES = {
    "HAQ": RESULTS_DIR / "distilbert_haq_inference_benchmark_bs8_ep3_wd001_warmup500.json",
    "HAWQ": RESULTS_DIR / "distilbert_hawq_inference_benchmark_bs8_ep3_wd001_warmup500.json",
    "Sensimix": RESULTS_DIR / "distilbert_sensimix_inference_benchmark_bs8_ep3_wd001_warmup500.json",
    "Zeroquant": RESULTS_DIR / "distilbert_zeroquant_inference_benchmark_bs8_ep3_wd001_warmup500.json",
    "Duquant": RESULTS_DIR / "distilbert_duquant_inference_benchmark_bs8_ep3_wd001_warmup500.json",
}

OUTPUT_CSV = RESULTS_DIR / "updated_distilbert_agnews_report_table_bs8_ep3_wd001_warmup500.csv"
OUTPUT_XLSX = RESULTS_DIR / "updated_distilbert_agnews_report_table_bs8_ep3_wd001_warmup500.xlsx"


def pct(x):
    return round(float(x) * 100, 4)


def main():
    rows = []

    for method_name, path in FILES.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing result file for {method_name}: {path}")

        with open(path, "r") as f:
            data = json.load(f)

        rows.append({
            "Model Name": "DistilBERT",
            "Dataset": "AG News",
            "Method Name": method_name,
            "Accuracy": pct(data["accuracy"]),
            "GPU Memory": round(float(data["peak_nvml_memory_used_gb"]), 4),
            "Latency": round(float(data["latency_ms_per_sample"]), 4),
            "Precision": pct(data["precision_weighted"]),
            "Recall": pct(data["recall_weighted"]),
            "F1 Score": pct(data["f1_weighted"]),
            "Throughput": round(float(data["throughput_samples_per_sec"]), 4),
            "Energy": round(float(data["energy_joules"]), 4),
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)
    df.to_excel(OUTPUT_XLSX, index=False)

    print("\nUpdated DistilBERT AG News table:")
    print(df.to_string(index=False))

    print("\nSaved CSV:", OUTPUT_CSV)
    print("Saved Excel:", OUTPUT_XLSX)


if __name__ == "__main__":
    main()
