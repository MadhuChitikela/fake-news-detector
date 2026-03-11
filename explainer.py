import shap
import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from classifier import model, tokenizer, device


def get_word_importance(text: str) -> list:
    """
    Returns list of (word, importance_score) pairs.
    Positive = supports REAL, Negative = supports FAKE
    """
    words = text.split()
    if len(words) > 100:
        words = words[:100]
        text = " ".join(words)

    scores = []

    # Baseline — empty string score
    baseline_inputs = tokenizer(
        "", return_tensors="pt",
        truncation=True, max_length=512, padding=True
    ).to(device)

    with torch.no_grad():
        baseline_out = model(**baseline_inputs)
        baseline_prob = torch.softmax(baseline_out.logits, dim=1)[0][1].item()

    # Score each word by removing it
    for i, word in enumerate(words):
        masked = words.copy()
        masked[i] = ""
        masked_text = " ".join(masked).strip()

        inputs = tokenizer(
            masked_text, return_tensors="pt",
            truncation=True, max_length=512, padding=True
        ).to(device)

        with torch.no_grad():
            out = model(**inputs)
            prob = torch.softmax(out.logits, dim=1)[0][1].item()

        # Importance = how much removing this word changes prediction
        importance = baseline_prob - prob
        scores.append((word, round(importance, 4)))

    # Sort by absolute importance
    scores.sort(key=lambda x: abs(x[1]), reverse=True)
    return scores[:20]


def get_highlighted_sentences(text: str, fake_probability: float) -> list:
    """
    Returns sentences with risk scores for UI highlighting.
    """
    sentences = [s.strip() for s in text.replace("!", ".").replace("?", ".").split(".") if s.strip()]
    results = []

    for sentence in sentences:
        if len(sentence) < 10:
            continue

        inputs = tokenizer(
            sentence, return_tensors="pt",
            truncation=True, max_length=512, padding=True
        ).to(device)

        with torch.no_grad():
            out = model(**inputs)
            probs = torch.softmax(out.logits, dim=1)[0]
            fake_prob = probs[0].item()

        # Risk color
        if fake_prob > 0.7:
            risk = "high"
            color = "#ff4466"
        elif fake_prob > 0.4:
            risk = "medium"
            color = "#ffaa00"
        else:
            risk = "low"
            color = "#00ff88"

        results.append({
            "sentence":  sentence,
            "fake_prob": round(fake_prob * 100, 1),
            "risk":      risk,
            "color":     color
        })

    return results


# ── Test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    test = "Scientists confirm aliens landed on Earth last Tuesday near Area 51."
    print("🔍 Word Importance:")
    scores = get_word_importance(test)
    for word, score in scores[:5]:
        print(f"  {word}: {score}")

    print("\n🎨 Sentence Highlighting:")
    sentences = get_highlighted_sentences(test, 0.8)
    for s in sentences:
        print(f"  [{s['risk'].upper()}] {s['sentence'][:50]}")
