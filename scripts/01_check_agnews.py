from datasets import load_dataset


def main():
    print("Loading AG News dataset...")
    dataset = load_dataset("fancyzhx/ag_news")

    print("\nDataset object:")
    print(dataset)

    print("\nTrain split:")
    print(dataset["train"])

    print("\nTest split:")
    print(dataset["test"])

    print("\nFeatures:")
    print(dataset["train"].features)

    label_feature = dataset["train"].features["label"]
    print("\nLabel feature:")
    print(label_feature)

    if hasattr(label_feature, "names"):
        print("\nLabel names:")
        for idx, name in enumerate(label_feature.names):
            print(f"{idx}: {name}")

    print("\nFirst 3 training examples:")
    for i in range(3):
        row = dataset["train"][i]
        print("-" * 80)
        print("Text:", row["text"])
        print("Label id:", row["label"])
        if hasattr(label_feature, "names"):
            print("Label name:", label_feature.names[row["label"]])


if __name__ == "__main__":
    main()
