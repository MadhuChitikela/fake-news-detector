import pandas as pd
import os

def download_liar_dataset():
    """
    Downloads and prepares the LIAR fake news dataset
    Converts to binary: Real (1) vs Fake (0)
    """
    print("📥 Downloading LIAR dataset...")

    # Download directly from GitHub mirror
    train_url = "https://raw.githubusercontent.com/thiagorainmaker77/liar_dataset/master/train.tsv"
    test_url  = "https://raw.githubusercontent.com/thiagorainmaker77/liar_dataset/master/test.tsv"
    val_url   = "https://raw.githubusercontent.com/thiagorainmaker77/liar_dataset/master/valid.tsv"

    cols = ["id","label","statement","subject","speaker","job",
            "state","party","barely_true","false","half_true",
            "mostly_true","pants_fire","context"]

    train_df = pd.read_csv(train_url, sep="\t", header=None, names=cols)
    test_df  = pd.read_csv(test_url,  sep="\t", header=None, names=cols)
    val_df   = pd.read_csv(val_url,   sep="\t", header=None, names=cols)

    print(f"✅ Train: {len(train_df)} | Test: {len(test_df)} | Val: {len(val_df)}")

    # Convert to binary labels
    # Real = mostly-true, true, half-true
    # Fake = false, barely-true, pants-fire
    def to_binary(label):
        if label in ["true", "mostly-true", "half-true"]:
            return 1  # Real
        else:
            return 0  # Fake

    for df in [train_df, test_df, val_df]:
        df["binary_label"] = df["label"].apply(to_binary)

    # Save cleaned data
    os.makedirs("data", exist_ok=True)
    train_df[["statement","binary_label"]].to_csv("data/train.csv", index=False)
    test_df[["statement","binary_label"]].to_csv("data/test.csv",   index=False)
    val_df[["statement","binary_label"]].to_csv("data/val.csv",     index=False)

    print("✅ Dataset saved to data/ folder!")
    print(f"   Train samples : {len(train_df)}")
    print(f"   Test samples  : {len(test_df)}")
    print(f"   Val samples   : {len(val_df)}")

    # Show label distribution
    print("\n📊 Label Distribution (Train):")
    print(train_df["binary_label"].value_counts())
    print("   0 = Fake | 1 = Real")


if __name__ == "__main__":
    download_liar_dataset()
