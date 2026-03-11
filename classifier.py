import os
import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification

# ── Load saved BERT model ────────────────────────────────────────
MODEL_PATH = "saved_model"

def load_model():
    """Load fine-tuned BERT model and tokenizer"""
    print("🔄 Loading BERT model...")
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
    model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"✅ Model loaded on {device}")
    return model, tokenizer, device


# Load once at startup
model, tokenizer, device = load_model()


def classify_text(text: str) -> dict:
    """
    Classifies news text as REAL or FAKE.
    Returns label, confidence score, and risk level.
    """
    # Tokenize input
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    ).to(device)

    # Get prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    # 0 = FAKE, 1 = REAL
    predicted_class = int(np.argmax(probs))
    confidence = float(np.max(probs)) * 100

    label = "REAL" if predicted_class == 1 else "FAKE"
    trust_score = float(probs[1]) * 100  # probability of being REAL

    # Risk level
    if trust_score >= 75:
        risk = "LOW RISK"
        color = "green"
    elif trust_score >= 45:
        risk = "MEDIUM RISK"
        color = "orange"
    else:
        risk = "HIGH RISK"
        color = "red"

    return {
        "label":       label,
        "trust_score": round(trust_score, 2),
        "confidence":  round(confidence, 2),
        "risk_level":  risk,
        "color":       color,
        "fake_prob":   round(float(probs[0]) * 100, 2),
        "real_prob":   round(float(probs[1]) * 100, 2),
    }


# ── Test it ──────────────────────────────────────────────────────
if __name__ == "__main__":
    test_texts = [
        "Scientists discover new vaccine that prevents cancer completely",
        "NASA confirms moon is made of cheese after secret mission",
    ]
    for text in test_texts:
        result = classify_text(text)
        print(f"\nText: {text[:60]}...")
        print(f"Label: {result['label']}")
        print(f"Trust Score: {result['trust_score']}%")
        print(f"Risk: {result['risk_level']}")
