import json
from pathlib import Path

import pandas as pd


RESULTS_DIR = Path("results_updated")
EXP_TAG = "bs8_ep3_wd001_warmup500"

FILES = {
    "HAQ": RESULTS_DIR / f"gpt2_haq_inference_benchmark_{EXP_TAG}.json",
    "HAWQ": RESULTS_DIR / f"gpt2_hawq_inference_benchmark_{EXP_TAG}.json",
    "Sensimix": RESULTS_DIR / f"gpt2_sensimix_inference_benchmark_{EXP_TAG}.json",
    "Duquant": RESULTS_DIR / f"gpt2_duquant_inference_benchmark_{EXP_TAG}.json",
    "Zeroquant": RESULTS_DIR / f"gpt2_zeroquant_inference_benchmark_{EXP_TAG}.json",
}

OUTPUT_CSV = RESULTS_DIR / f"updated_gpt2_agnews_report_table_{EXP_TAG}.csv"
OUTPUT_XLSX = RESULTS_DIR / f"updated_gpt2_agnews_report_table_{EXP_TAG}.xlsx"


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
            "Model Name": "GPT2",
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
        }

        rows.append(row)

    df = pd.DataFrame(rows)

    df.to_csv(OUTPUT_CSV, index=False)
    df.to_excel(OUTPUT_XLSX, index=False)

    print("\nUpdated GPT2 AG News report table:")
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