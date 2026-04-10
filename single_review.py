import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForSequenceClassification
from collections import Counter, defaultdict

# 1. Setup the Device and Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SENTIMENT_LABEL_NAMES = ["positive", "negative", "neutral", "conflict"]
SENTIMENT_SCORE_MAP   = {"positive": 1.0, "negative": -1.0, "neutral": 0.0, "conflict": -0.5}
ATE_LABEL_NAMES       = ["O", "B-ASP", "I-ASP"]
SCORE_VEC             = np.array([SENTIMENT_SCORE_MAP[l] for l in SENTIMENT_LABEL_NAMES])

# Define the paths to your local trained models
MODEL_REGISTRY = {
    "BERT-base":       {"ate": "./bert_ate_model_final",     "sent": "./bert_sentiment_model_final"},
    "DeBERTa-v3-base": {"ate": "./deberta_ate_model_final",  "sent": "./deberta_sentiment_model_final"},
    "RoBERTa-base":    {"ate": "./roberta_ate_model_final",  "sent": "./roberta_sentiment_model_final"},
    "ELECTRA-base":    {"ate": "./electra_ate_model_final",  "sent": "./electra_sentiment_model_final"}
}

# 2. Globally load the Ensemble Models (Run once on startup)
print("⏳ Loading Ensemble Models into memory...")
ensemble_models = {}
for name, dirs in MODEL_REGISTRY.items():
    try:
        tok_ate  = AutoTokenizer.from_pretrained(dirs["ate"])
        mdl_ate  = AutoModelForTokenClassification.from_pretrained(dirs["ate"]).to(DEVICE)
        tok_sent = AutoTokenizer.from_pretrained(dirs["sent"])
        mdl_sent = AutoModelForSequenceClassification.from_pretrained(dirs["sent"]).to(DEVICE)
        
        mdl_ate.eval()
        mdl_sent.eval()
        
        ensemble_models[name] = {
            "tok_ate": tok_ate, "mdl_ate": mdl_ate,
            "tok_sent": tok_sent, "mdl_sent": mdl_sent
        }
        print(f"Loaded {name}")
    except Exception as e:
        print(f"Could not load {name}: {e}")

# ---------------------------------------------------------------------------
# Core Logic Functions (Ported from Streamlit)
# ---------------------------------------------------------------------------

def extract_aspects(text, ate_model, ate_tok):
    words = text.split()
    if not words: return []
    enc = ate_tok(words, is_split_into_words=True, max_length=128,
                  padding="max_length", truncation=True, return_tensors="pt")
    with torch.no_grad():
        out   = ate_model(input_ids=enc["input_ids"].to(DEVICE),
                          attention_mask=enc["attention_mask"].to(DEVICE))
        preds = torch.argmax(out.logits.squeeze(0), dim=-1).cpu().numpy()
    word_preds = {}
    for ti, wi in enumerate(enc.word_ids()):
        if wi is not None and wi not in word_preds:
            word_preds[wi] = preds[ti]
    aspects, current = [], []
    for wi, word in enumerate(words):
        label = ATE_LABEL_NAMES[word_preds.get(wi, 0)]
        if label == "B-ASP":
            if current: aspects.append(" ".join(current))
            current = [word]
        elif label == "I-ASP" and current:
            current.append(word)
        else:
            if current: aspects.append(" ".join(current))
            current = []
    if current: aspects.append(" ".join(current))
    return aspects

def classify_sentiment(text, aspect, sent_model, sent_tok):
    enc = sent_tok(f"{aspect} [SEP] {text}", max_length=128,
                   padding="max_length", truncation=True, return_tensors="pt")
    with torch.no_grad():
        out   = sent_model(input_ids=enc["input_ids"].to(DEVICE),
                           attention_mask=enc["attention_mask"].to(DEVICE))
        probs = torch.softmax(out.logits, dim=1).squeeze().cpu().numpy()
    pred_idx = int(np.argmax(probs))
    return SENTIMENT_LABEL_NAMES[pred_idx], float(probs[pred_idx]), probs

def compute_vendor_profile(results_list, review_count, use_norm=True):
    asp_scores = defaultdict(list)
    for r in results_list:
        # Simplified normalization for single review module
        key = r["aspect"].strip().lower()
        asp_scores[key].append(r["weighted_score"])
    if not asp_scores: return None
    total      = sum(len(v) for v in asp_scores.values())
    asp_means  = {a: float(np.mean(v)) for a, v in asp_scores.items()}
    raw_score  = sum(asp_means[a] * (len(asp_scores[a]) / total) for a in asp_means)
    profile    = {}
    for a in asp_means:
        profile[a] = {"score": asp_means[a], "count": len(asp_scores[a])}
    return {"raw_score": round(raw_score, 4), "review_count": review_count,
            "aspect_profile": profile}

# ---------------------------------------------------------------------------
# Pipeline Wrappers
# ---------------------------------------------------------------------------

def run_ensemble_pipeline(text, ensemble_models, min_votes=2):
    if not ensemble_models: return []
    votes = Counter()
    for key, m in ensemble_models.items():
        for span in extract_aspects(text, m["mdl_ate"], m["tok_ate"]):
            votes[span.lower().strip()] += 1
    retained = [sp for sp, cnt in votes.items() if cnt >= min(min_votes, len(ensemble_models))]
    if not retained: return []
    results = []
    for aspect in retained:
        prob_stack = []
        for key, m in ensemble_models.items():
            _, _, probs = classify_sentiment(text, aspect, m["mdl_sent"], m["tok_sent"])
            prob_stack.append(probs)
        mp = np.mean(prob_stack, axis=0)
        pred_idx = int(np.argmax(mp))
        results.append({
            "aspect": aspect, "sentiment": SENTIMENT_LABEL_NAMES[pred_idx],
            "confidence": float(mp[pred_idx]),
            "weighted_score": float(np.dot(mp, SCORE_VEC)), "probs": mp,
        })
    return results

def score_single_review(review_text):
    results = run_ensemble_pipeline(review_text, ensemble_models)
    if not results:
        return {
            "message": "No aspect terms detected.",
            "raw_score": 0.0,
            "star_rating": 3.0,
            "aspects": []
        }
    profile = compute_vendor_profile(results, review_count=1)
    raw_score = profile["raw_score"]
    star_rating = round(float(np.clip((raw_score + 1) / 2 * 5, 0, 5)), 2)
    return {
        "raw_score": raw_score,
        "star_rating": star_rating,
        "aspect_breakdown": profile["aspect_profile"],
        "detailed_results": results
    }