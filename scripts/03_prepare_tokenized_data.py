from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer


DATASET_NAME = "fancyzhx/ag_news"
MAX_LENGTH = 128

MODEL_CONFIGS = {
    "distilbert": "distilbert/distilbert-base-uncased",
    "gpt2": "openai-community/gpt2",
}


def tokenize_and_save(model_key: str, model_name: str):
    print("=" * 100)
    print(f"Preparing data for: {model_key}")
    print(f"Model checkpoint: {model_name}")
    print("=" * 100)

    dataset = load_dataset(DATASET_NAME)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        print("Tokenizer has no pad_token. Setting pad_token = eos_token.")
        tokenizer.pad_token = tokenizer.eos_token

    label_names = dataset["train"].features["label"].names
    print("Label names:", label_names)

    def tokenize_batch(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )

    print("Tokenizing dataset...")
    tokenized = dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=["text"],
        desc=f"Tokenizing for {model_key}",
    )

    tokenized = tokenized.rename_column("label", "labels")

    tokenized.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )

    out_dir = Path("data") / "tokenized" / f"{model_key}_agnews_maxlen{MAX_LENGTH}"
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving tokenized dataset to: {out_dir}")
    tokenized.save_to_disk(str(out_dir))

    print("\nSaved dataset:")
    print(tokenized)

    print("\nOne tokenized training example:")
    example = tokenized["train"][0]
    print("input_ids shape:", example["input_ids"].shape)
    print("attention_mask shape:", example["attention_mask"].shape)
    print("labels:", example["labels"])

    print(f"\nDone for {model_key}.")


def main():
    for model_key, model_name in MODEL_CONFIGS.items():
        tokenize_and_save(model_key, model_name)


if __name__ == "__main__":
    main()
