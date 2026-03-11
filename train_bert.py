import pandas as pd
import numpy as np
import torch
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
import os

# ── Config ──────────────────────────────────────────────────────
MODEL_NAME  = "bert-base-uncased"
OUTPUT_DIR  = "saved_model"
MAX_LENGTH  = 128
BATCH_SIZE  = 16
EPOCHS      = 3


def load_data():
    fake_path = "data/Fake.csv"
    true_path = "data/True.csv"
    
    if os.path.exists(fake_path) and os.path.exists(true_path):
        print(f"📊 Found Kaggle dataset ({fake_path} and {true_path}). Loading and merging...")
        fake_df = pd.read_csv(fake_path)
        true_df = pd.read_csv(true_path)
        
        fake_df["binary_label"] = 0
        true_df["binary_label"] = 1
        
        # Use title + text for more context
        fake_df["statement"] = fake_df["title"] + " " + fake_df["text"]
        true_df["statement"] = true_df["title"] + " " + true_df["text"]
        
        df = pd.concat([fake_df, true_df], ignore_index=True)
        df = df[["statement", "binary_label"]].dropna()
        
        # Shuffle
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Split 80/10/10
        n = len(df)
        train_df = df.iloc[:int(n*0.8)]
        val_df   = df.iloc[int(n*0.8):int(n*0.9)]
        test_df  = df.iloc[int(n*0.9):]
        
        print(f"✅ Kaggle data loaded: {len(df)} samples total.")
        return train_df, test_df, val_df
    
    # Fallback to LIAR dataset
    print("📋 Kaggle data not found in root. Falling back to LIAR data in data/ folder...")
    train = pd.read_csv("data/train.csv").dropna()
    test  = pd.read_csv("data/test.csv").dropna()
    val   = pd.read_csv("data/val.csv").dropna()
    return train, test, val


def tokenize_data(df, tokenizer):
    dataset = Dataset.from_pandas(df)

    def tokenize(batch):
        return tokenizer(
            batch["statement"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH
        )

    tokenized = dataset.map(tokenize, batched=True)
    tokenized = tokenized.rename_column("binary_label", "labels")
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    return tokenized


def compute_metrics(pred):
    labels = pred.label_ids
    preds  = pred.predictions.argmax(-1)
    acc    = accuracy_score(labels, preds)
    f1     = f1_score(labels, preds, average="weighted")
    return {"accuracy": acc, "f1": f1}


def train():
    print("🔄 Loading data...")
    train_df, test_df, val_df = load_data()

    print("🔄 Loading BERT tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    print("🔄 Tokenizing data...")
    train_data = tokenize_data(train_df, tokenizer)
    val_data   = tokenize_data(val_df,   tokenizer)
    test_data  = tokenize_data(test_df,  tokenizer)

    print("🔄 Loading BERT model...")
    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_dir="logs",
        logging_steps=100,
        fp16=torch.cuda.is_available(),  # Enable mixed precision if GPU available
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        compute_metrics=compute_metrics,
    )

    print("🚀 Training BERT model...")
    device = "GPU" if torch.cuda.is_available() else "CPU"
    print(f"   Training on {device}...")
    if device == "CPU":
        print("   This takes 20-30 minutes on CPU")
        print("   Go grab a coffee! ☕")
    else:
        print("   GPU detected! Training will be much faster. ⚡")
    trainer.train()

    print("🔄 Evaluating on test set...")
    results = trainer.evaluate(test_data)
    print(f"\n✅ Test Accuracy : {results['eval_accuracy']:.4f}")
    print(f"✅ Test F1 Score : {results['eval_f1']:.4f}")

    print("💾 Saving model...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"✅ Model saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    train()
