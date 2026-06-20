from datasets import load_dataset
from transformers import AutoTokenizer


DATASET_NAME = "fancyzhx/ag_news"

MODEL_CONFIGS = {
    "distilbert": "distilbert/distilbert-base-uncased",
    "gpt2": "openai-community/gpt2",
}


def inspect_tokenizer(model_key: str, model_name: str, sample_text: str):
    print("=" * 100)
    print(f"MODEL KEY: {model_key}")
    print(f"MODEL NAME: {model_name}")
    print("=" * 100)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # GPT2 has no pad token by default. For classification batching, use EOS as PAD.
    if tokenizer.pad_token is None:
        print("Tokenizer has no pad_token. Setting pad_token = eos_token.")
        tokenizer.pad_token = tokenizer.eos_token

    print("Tokenizer class:", tokenizer.__class__.__name__)
    print("Vocab size:", tokenizer.vocab_size)
    print("Pad token:", tokenizer.pad_token)
    print("Pad token id:", tokenizer.pad_token_id)
    print("EOS token:", tokenizer.eos_token)
    print("EOS token id:", tokenizer.eos_token_id)
    print("Model max length:", tokenizer.model_max_length)

    # also create attention mask to indicate which tokens are padding and which are not
    encoded = tokenizer(
        sample_text,
        padding="max_length",
        truncation=True,
        max_length=64, # Short texts get padded to 64 and long texts get truncated to 64
        return_tensors=None,
    )
    
    print("\nOriginal text:")
    print(sample_text)

    print("\nEncoded keys:")
    print(encoded.keys())

    print("\ninput_ids length:", len(encoded["input_ids"]))
    print("attention_mask length:", len(encoded["attention_mask"]))

    print("\nFirst 25 input_ids:")
    print(encoded["input_ids"][:25])

    print("\nFirst 25 attention_mask values:")
    print(encoded["attention_mask"][:25])

    tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][:25])
    print("\nFirst 25 tokens:")
    print(tokens)

    decoded = tokenizer.decode(encoded["input_ids"], skip_special_tokens=True)
    print("\n\nOriginal text:", sample_text)
    print("\nDecoded text:", decoded)


def main():
    print("Loading dataset:", DATASET_NAME)
    dataset = load_dataset(DATASET_NAME)

    sample = dataset["train"][0]
    sample_text = sample["text"]
    sample_label = sample["label"]
    label_names = dataset["train"].features["label"].names

    print("\nSample text:")
    print(sample_text)
    print("\nSample label id:", sample_label)
    print("Sample label name:", label_names[sample_label])

    print(MODEL_CONFIGS.items());

    for model_key, model_name in MODEL_CONFIGS.items():
        inspect_tokenizer(model_key, model_name, sample_text)


if __name__ == "__main__":
    main()
