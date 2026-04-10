"""
ABSA Streamlit Application - Multi-Model Edition
Supports: BERT, DeBERTa-v3, RoBERTa, ELECTRA, and Ensemble pipeline.
Elo-based vendor ranking replaces the simple relative star scaling.

Run:
    streamlit run absa_streamlit_app.py

Train models first via the individual training notebooks.
"""

import io
import os
import warnings
from collections import defaultdict, Counter

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForSequenceClassification

import absa_how_it_works

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SENTIMENT_LABEL_NAMES = ["positive", "negative", "neutral", "conflict"]
SENTIMENT_SCORE_MAP   = {"positive": 1.0, "negative": -1.0, "neutral": 0.0, "conflict": -0.5}
ATE_LABEL_NAMES       = ["O", "B-ASP", "I-ASP"]
SENT_PALETTE          = ["#2ecc71", "#e74c3c", "#3498db", "#f39c12"]
SCORE_VEC             = np.array([SENTIMENT_SCORE_MAP[l] for l in SENTIMENT_LABEL_NAMES])

# All model directory pairs
MODEL_REGISTRY = {
    "BERT-base":       {"ate": "./bert_ate_model_final",     "sent": "./bert_sentiment_model_final",
                        "checkpoint": "bert-base-uncased"},
    "DeBERTa-v3-base": {"ate": "./deberta_ate_model_final",  "sent": "./deberta_sentiment_model_final",
                        "checkpoint": "microsoft/deberta-v3-base"},
    "RoBERTa-base":    {"ate": "./roberta_ate_model_final",  "sent": "./roberta_sentiment_model_final",
                        "checkpoint": "roberta-base"},
    "ELECTRA-base":    {"ate": "./electra_ate_model_final",  "sent": "./electra_sentiment_model_final",
                        "checkpoint": "google/electra-base-discriminator"},
    # Legacy fallback paths (original single-model training)
    "BERT (legacy)":   {"ate": "./ate_model_final",          "sent": "./sentiment_model_final",
                        "checkpoint": "bert-base-uncased"},
}

ASPECT_NORM_MAP = {
    "food": "quality", "taste": "quality", "flavour": "quality", "flavor": "quality",
    "portion": "quality", "meal": "quality", "dish": "quality", "menu": "quality",
    "ingredient": "quality", "freshness": "quality", "cuisine": "quality",
    "photos": "quality", "photo": "quality", "image": "quality", "images": "quality",
    "picture": "quality", "pictures": "quality", "editing": "quality",
    "video": "quality", "footage": "quality", "recording": "quality",
    "sound": "quality", "audio": "quality", "music": "quality", "playlist": "quality",
    "performance": "quality", "decoration": "quality", "decor": "quality",
    "cake": "quality", "dessert": "quality", "catering": "quality",
    "product": "quality", "build": "quality", "quality": "quality",
    "setup": "quality", "design": "quality", "output": "quality",
    "lighting": "quality", "equipment": "quality", "gear": "quality",
    "flowers": "quality", "arrangement": "quality", "backdrop": "quality",
    "entertainment": "quality", "show": "quality", "act": "quality",
    "service": "service", "staff": "service", "waiter": "service",
    "waitress": "service", "server": "service", "host": "service",
    "bartender": "service", "chef": "service", "cook": "service",
    "photographer": "service", "videographer": "service",
    "dj": "service", "emcee": "service", "mc": "service",
    "crew": "service", "team": "service", "worker": "service",
    "guard": "service", "security": "service", "officer": "service",
    "bouncer": "service", "personnel": "service",
    "planner": "service", "coordinator": "service", "organiser": "service",
    "organizer": "service", "florist": "service", "baker": "service",
    "caterer": "service", "driver": "service", "chauffeur": "service",
    "support": "service", "communication": "service", "responsiveness": "service",
    "helpfulness": "service", "attitude": "service", "professionalism": "service",
    "behaviour": "service", "behavior": "service", "manner": "service",
    "courtesy": "service", "friendliness": "service", "politeness": "service",
    "price": "value", "prices": "value", "cost": "value", "value": "value",
    "pricing": "value", "fee": "value", "fees": "value", "charge": "value",
    "charges": "value", "rate": "value", "rates": "value", "quote": "value",
    "package": "value", "deal": "value", "expensive": "value",
    "cheap": "value", "affordable": "value", "budget": "value",
    "reliability": "reliability", "punctuality": "reliability", "timing": "reliability",
    "delivery": "reliability", "wait": "reliability", "waiting": "reliability",
    "delay": "reliability", "delays": "reliability", "time": "reliability",
    "speed": "reliability", "turnaround": "reliability", "schedule": "reliability",
    "deadline": "reliability", "promptness": "reliability",
    "ambiance": "ambiance", "ambience": "ambiance", "atmosphere": "ambiance",
    "venue": "ambiance", "location": "ambiance", "place": "ambiance",
    "space": "ambiance", "cleanliness": "ambiance", "noise": "ambiance",
    "environment": "ambiance", "setting": "ambiance", "hall": "ambiance",
    "room": "ambiance", "area": "ambiance", "facility": "ambiance",
    "parking": "ambiance", "access": "ambiance", "accessibility": "ambiance",
    "experience": "experience", "overall": "experience", "visit": "experience",
    "recommendation": "experience", "event": "experience", "occasion": "experience",
    "celebration": "experience", "party": "experience", "wedding": "experience",
    "function": "experience",
}

# ---------------------------------------------------------------------------
# Page config and styles
# ---------------------------------------------------------------------------

st.set_page_config(page_title="ABSA", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.stApp { background-color: #0d0d0d; color: #e8e8e8; }
[data-testid="stSidebar"] { background-color: #0f0f0f; border-right: 1px solid #1e1e1e; }
[data-testid="stSidebar"] * { color: #bbbbbb !important; }
.absa-header { border-bottom: 1px solid #1e1e1e; padding: 2rem 0 1.4rem 0; margin-bottom: 1.8rem; }
.absa-title { font-family: 'IBM Plex Mono', monospace; font-size: 2.2rem; font-weight: 600;
              letter-spacing: -0.03em; color: #ffffff; margin: 0; line-height: 1; }
.absa-subtitle { font-size: 0.8rem; color: #444; font-weight: 300; margin-top: 0.4rem;
                 letter-spacing: 0.1em; text-transform: uppercase; }
.stat-row { display: flex; gap: 0.7rem; margin-bottom: 1.4rem; flex-wrap: wrap; }
.stat-card { flex: 1; min-width: 100px; background: #111; border: 1px solid #1e1e1e;
             border-radius: 4px; padding: 0.9rem 1.1rem; }
.stat-label { font-size: 0.64rem; text-transform: uppercase; letter-spacing: 0.1em;
              color: #3a3a3a; margin-bottom: 0.25rem; font-family: 'IBM Plex Mono', monospace; }
.stat-value { font-size: 1.55rem; font-weight: 600; color: #ffffff;
              font-family: 'IBM Plex Mono', monospace; line-height: 1; }
.sec { font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem; text-transform: uppercase;
       letter-spacing: 0.13em; color: #3a3a3a; border-bottom: 1px solid #161616;
       padding-bottom: 0.35rem; margin: 1.6rem 0 0.8rem 0; }
.info-box { background: #0d1620; border-left: 3px solid #3498db; padding: 0.75rem 1rem;
            font-size: 0.84rem; color: #777; margin-bottom: 0.9rem; border-radius: 0 3px 3px 0; }
.warn-box { background: #160f00; border-left: 3px solid #f39c12; padding: 0.75rem 1rem;
            font-size: 0.84rem; color: #777; margin-bottom: 0.9rem; border-radius: 0 3px 3px 0; }
.ok-box   { background: #051209; border-left: 3px solid #2ecc71; padding: 0.75rem 1rem;
            font-size: 0.84rem; color: #777; margin-bottom: 0.9rem; border-radius: 0 3px 3px 0; }
.err-box  { background: #160505; border-left: 3px solid #e74c3c; padding: 0.75rem 1rem;
            font-size: 0.84rem; color: #777; margin-bottom: 0.9rem; border-radius: 0 3px 3px 0; }
.asp-grid { display: flex; flex-wrap: wrap; gap: 0.6rem; margin-bottom: 1.1rem; }
.asp-card { background: #111; border: 1px solid #1e1e1e; border-radius: 4px;
            padding: 0.8rem 1.1rem; min-width: 150px; flex: 1; max-width: 210px; }
.asp-name { font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; text-transform: uppercase;
            letter-spacing: 0.08em; color: #555; margin-bottom: 0.25rem; }
.asp-sent { font-family: 'IBM Plex Mono', monospace; font-size: 1.2rem; font-weight: 600; line-height: 1; }
.asp-meta { font-size: 0.68rem; color: #2a2a2a; margin-top: 0.2rem; font-family: 'IBM Plex Mono', monospace; }
.asp-bar  { background: #161616; height: 3px; border-radius: 2px; margin-top: 0.45rem; overflow: hidden; }
.asp-fill { height: 100%; border-radius: 2px; }
.no-asp   { background: #0f0f0f; border: 1px dashed #222; border-radius: 4px; padding: 1.3rem;
            text-align: center; color: #3a3a3a; font-family: 'IBM Plex Mono', monospace;
            font-size: 0.8rem; margin: 0.7rem 0; }
.leaderboard-wrap { overflow-x: auto; margin-bottom: 1.4rem; }
.lb-table { width: 100%; border-collapse: collapse; background: #0f0f0f; }
.lb-th { padding: 0.5rem 0.85rem; text-align: left; color: #3a3a3a; font-size: 0.62rem;
         text-transform: uppercase; letter-spacing: 0.1em; border-bottom: 1px solid #161616;
         font-family: 'IBM Plex Mono', monospace; }
.lb-td { padding: 0.45rem 0.85rem; border-bottom: 1px solid #131313;
         font-family: 'IBM Plex Mono', monospace; font-size: 0.82rem; }
[data-testid="stTabs"] button { font-family: 'IBM Plex Mono', monospace; font-size: 0.73rem;
                                letter-spacing: 0.08em; text-transform: uppercase; }
.stButton > button { font-family: 'IBM Plex Mono', monospace; font-size: 0.78rem;
                     letter-spacing: 0.06em; text-transform: uppercase; border-radius: 3px; }
.stTextArea textarea, .stTextInput input {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.86rem;
    background-color: #0f0f0f !important; color: #e0e0e0 !important;
    border: 1px solid #1e1e1e !important; }
#MainMenu, footer { visibility: hidden; }
</style>""", unsafe_allow_html=True)

plt.rcParams.update({
    "figure.facecolor": "#111111", "axes.facecolor": "#111111",
    "axes.edgecolor": "#1e1e1e", "axes.labelcolor": "#666",
    "xtick.color": "#444", "ytick.color": "#444", "text.color": "#bbbbbb",
    "grid.color": "#161616", "grid.linestyle": "--",
    "font.family": "monospace", "figure.dpi": 110,
})

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown("""
<div class="absa-header">
    <div class="absa-title">ABSA</div>
    <div class="absa-subtitle">Aspect-Based Sentiment Analysis &nbsp;/&nbsp; Multi-Model BERT Pipeline</div>
</div>""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def sec(text):
    st.markdown(f'<div class="sec">{text}</div>', unsafe_allow_html=True)

def info(text):
    st.markdown(f'<div class="info-box">{text}</div>', unsafe_allow_html=True)

def warn(text):
    st.markdown(f'<div class="warn-box">{text}</div>', unsafe_allow_html=True)

def ok(text):
    st.markdown(f'<div class="ok-box">{text}</div>', unsafe_allow_html=True)

def err(text):
    st.markdown(f'<div class="err-box">{text}</div>', unsafe_allow_html=True)

def stat_row(*cards):
    inner = ""
    for card in cards:
        label, value = card[0], card[1]
        color = card[2] if len(card) > 2 else "#ffffff"
        inner += (f'<div class="stat-card"><div class="stat-label">{label}</div>'
                  f'<div class="stat-value" style="color:{color};">{value}</div></div>')
    st.markdown(f'<div class="stat-row">{inner}</div>', unsafe_allow_html=True)


def aspect_cards(results):
    if not results:
        st.markdown('<div class="no-asp">No aspect terms detected. See How It Works tab.</div>',
                    unsafe_allow_html=True)
        return
    cards = ""
    for r in results:
        score = r["weighted_score"]
        conf  = r["confidence"]
        sent  = r["sentiment"]
        bp    = int((score + 1) / 2 * 100)
        bc    = "#2ecc71" if score >= 0 else "#e74c3c"
        sc    = {"positive": "#2ecc71", "negative": "#e74c3c",
                 "neutral": "#3498db", "conflict": "#f39c12"}.get(sent, "#fff")
        cards += (f'<div class="asp-card"><div class="asp-name">{r["aspect"]}</div>'
                  f'<div class="asp-sent" style="color:{sc};">{sent}</div>'
                  f'<div class="asp-meta">{score:+.3f} | conf {conf:.2f}</div>'
                  f'<div class="asp-bar"><div class="asp-fill" style="width:{bp}%;background:{bc};"></div></div>'
                  f'</div>')
    st.markdown(f'<div class="asp-grid">{cards}</div>', unsafe_allow_html=True)


def stars_display(rating, size="1.2rem"):
    filled = max(0, min(5, int(round(rating))))
    empty  = 5 - filled
    s  = f'<span style="color:#f39c12;font-size:{size};">' + "&#9733;" * filled + "</span>"
    s += f'<span style="color:#222;font-size:{size};">' + "&#9733;" * empty + "</span>"
    return s


def rating_badge(vendor_name, star_rating, review_count):
    bar_pct = int(star_rating / 5 * 100)
    bar_col = "#2ecc71" if star_rating >= 3.5 else ("#f39c12" if star_rating >= 2.5 else "#e74c3c")
    st.markdown(f"""
    <div style="background:#111;border:1px solid #1e1e1e;border-radius:5px;
                padding:1.3rem 1.8rem;display:inline-block;min-width:200px;margin-bottom:0.9rem;">
        <div style="font-family:IBM Plex Mono,monospace;font-size:0.64rem;text-transform:uppercase;
                    letter-spacing:0.1em;color:#3a3a3a;margin-bottom:0.3rem;">Star Rating</div>
        <div style="font-family:IBM Plex Mono,monospace;font-size:2.8rem;font-weight:600;
                    color:#ffffff;line-height:1;">{star_rating:.2f}</div>
        <div style="margin:0.35rem 0;">{stars_display(star_rating, "1.3rem")}</div>
        <div style="background:#161616;height:3px;border-radius:2px;overflow:hidden;margin:0.4rem 0;">
            <div style="width:{bar_pct}%;background:{bar_col};height:100%;border-radius:2px;"></div>
        </div>
        <div style="font-family:IBM Plex Mono,monospace;font-size:0.64rem;color:#3a3a3a;margin-top:0.25rem;">
            {review_count} review{"s" if review_count != 1 else ""} analysed
        </div>
    </div>""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Model loading (cached)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading model...")
def load_single_model(ate_path, sent_path):
    tok_ate  = AutoTokenizer.from_pretrained(ate_path)
    mdl_ate  = AutoModelForTokenClassification.from_pretrained(ate_path).to(DEVICE)
    tok_sent = AutoTokenizer.from_pretrained(sent_path)
    mdl_sent = AutoModelForSequenceClassification.from_pretrained(sent_path).to(DEVICE)
    mdl_ate.eval(); mdl_sent.eval()
    return tok_ate, mdl_ate, tok_sent, mdl_sent


def available_models():
    avail = {}
    for name, dirs in MODEL_REGISTRY.items():
        if os.path.isdir(dirs["ate"]) and os.path.isdir(dirs["sent"]):
            avail[name] = dirs
    return avail


def models_ready():
    return "active_ate_model" in st.session_state


def get_models():
    return (st.session_state["active_ate_model"], st.session_state["active_ate_tok"],
            st.session_state["active_sent_model"], st.session_state["active_sent_tok"])


# ---------------------------------------------------------------------------
# Inference helpers
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


def run_pipeline(text, ate_model, ate_tok, sent_model, sent_tok):
    aspects = extract_aspects(text, ate_model, ate_tok)
    results = []
    for asp in aspects:
        label, conf, probs = classify_sentiment(text, asp, sent_model, sent_tok)
        results.append({
            "aspect": asp, "sentiment": label, "confidence": conf,
            "weighted_score": float(np.dot(probs, SCORE_VEC)), "probs": probs,
        })
    return results


def run_ensemble_pipeline(text, ensemble_models, min_votes=2):
    """
    Multi-model ensemble: ATE voting + ASC probability fusion.
    ensemble_models: {key: {"tok_ate":..,"mdl_ate":..,"tok_sent":..,"mdl_sent":..}}
    """
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


# ---------------------------------------------------------------------------
# Rating and Elo ranker
# ---------------------------------------------------------------------------

def normalise_aspect(term):
    return ASPECT_NORM_MAP.get(term.strip().lower(), term.strip().lower())


def compute_vendor_profile(results_list, review_count, use_norm=True):
    asp_scores = defaultdict(list)
    for r in results_list:
        key = normalise_aspect(r["aspect"]) if use_norm else r["aspect"].lower()
        asp_scores[key].append(r["weighted_score"])
    if not asp_scores: return None
    total      = sum(len(v) for v in asp_scores.values())
    asp_means  = {a: float(np.mean(v)) for a, v in asp_scores.items()}
    raw_score  = sum(asp_means[a] * (len(asp_scores[a]) / total) for a in asp_means)
    profile    = {}
    for a in asp_means:
        sents = [r["sentiment"] for r in results_list
                 if (normalise_aspect(r["aspect"]) if use_norm else r["aspect"].lower()) == a]
        dominant = max(set(sents), key=sents.count) if sents else "neutral"
        profile[a] = {"score": asp_means[a], "count": len(asp_scores[a]),
                      "weight": len(asp_scores[a]) / total, "dominant": dominant}
    return {"raw_score": round(raw_score, 4), "review_count": review_count,
            "aspect_profile": profile}


class SentimentEloRanker:
    """Elo-based vendor ranking. Reference: Elo (1978)."""
    def __init__(self, k=32, initial=1500, bayes_m=10):
        self.k = k; self.initial = initial; self.bayes_m = bayes_m
        self.ratings = {}; self.raw_scores = {}; self.rev_counts = {}

    def add_vendor(self, name, profile_dict):
        if not profile_dict: return
        if name not in self.ratings:
            self.ratings[name] = self.initial
        self.rev_counts[name]  = profile_dict["review_count"]
        raw   = profile_dict["raw_score"]
        v     = profile_dict["review_count"]
        m     = self.bayes_m
        gm    = np.mean(list(self.raw_scores.values())) if self.raw_scores else 0.0
        self.raw_scores[name] = (v * raw + m * gm) / (v + m)

    def _update(self, rounds=5):
        vendors = list(self.ratings.keys())
        for _ in range(rounds):
            for i in range(len(vendors)):
                for j in range(len(vendors)):
                    if i == j: continue
                    vi, vj = vendors[i], vendors[j]
                    ri, rj = self.ratings[vi], self.ratings[vj]
                    si = (self.raw_scores[vi] + 1) / 2
                    sj = (self.raw_scores[vj] + 1) / 2
                    actual   = si / (si + sj + 1e-9)
                    expected = 1.0 / (1.0 + 10 ** ((rj - ri) / 400))
                    self.ratings[vi] = ri + self.k * (actual - expected)

    def get_rankings(self):
        if not self.ratings: return pd.DataFrame()
        self._update()
        names    = list(self.ratings.keys())
        elo_arr  = np.array([self.ratings[n] for n in names])
        raw_arr  = np.array([self.raw_scores[n] for n in names])
        mn, mx   = elo_arr.min(), elo_arr.max()
        stars    = (np.full(len(names), 3.0) if mx == mn
                    else 1.0 + (elo_arr - mn) / (mx - mn) * 4.0)
        se       = np.abs(raw_arr) * 0.08 + 0.05
        df = pd.DataFrame({
            "vendor":      names,
            "elo_rating":  np.round(elo_arr, 1),
            "raw_score":   np.round(raw_arr, 4),
            "star_rating": np.round(np.clip(stars, 1, 5), 2),
            "ci_lo":       np.round(np.clip(stars - 1.96*se, 1, 5), 2),
            "ci_hi":       np.round(np.clip(stars + 1.96*se, 1, 5), 2),
            "reviews":     [self.rev_counts[n] for n in names],
        }).sort_values("elo_rating", ascending=False).reset_index(drop=True)
        df.insert(0, "rank", range(1, len(df)+1))
        return df


def assign_relative_ratings(vendor_profiles):
    names  = list(vendor_profiles.keys())
    scores = [vendor_profiles[n]["raw_score"] for n in names]
    mn, mx = min(scores), max(scores)
    result = {}
    for name in names:
        s    = vendor_profiles[name]["raw_score"]
        star = 3.0 if mx == mn else round(float(np.clip(1.0 + (s-mn)/(mx-mn)*4.0, 1, 5)), 2)
        result[name] = {**vendor_profiles[name], "star_rating": star}
    return result


# ---------------------------------------------------------------------------
# Chart helpers (kept identical to original app)
# ---------------------------------------------------------------------------

def fig_prob_bars(probs, title=""):
    fig, ax = plt.subplots(figsize=(4.5, 2.6))
    bars = ax.barh(SENTIMENT_LABEL_NAMES, probs, color=SENT_PALETTE, height=0.5)
    ax.set_xlim(0, 1.15); ax.set_xlabel("Probability", fontsize=8)
    if title: ax.set_title(title, fontsize=8.5, pad=5)
    ax.spines[["top","right"]].set_visible(False); ax.tick_params(labelsize=8)
    for bar, val in zip(bars, probs):
        ax.text(val+0.01, bar.get_y()+bar.get_height()/2, f"{val:.2f}", va="center", fontsize=7.5)
    plt.tight_layout(); return fig


def fig_aspect_scores(profile, max_show=18):
    items  = sorted(profile.items(), key=lambda x: -x[1]["count"])[:max_show]
    labels = [a for a, _ in items]
    scores = [profile[a]["score"] for a in labels]
    colors = ["#2ecc71" if s >= 0 else "#e74c3c" for s in scores]
    h = max(3.0, len(labels)*0.38)
    fig, ax = plt.subplots(figsize=(5.5, h))
    ax.barh(labels, scores, color=colors, height=0.55)
    ax.axvline(0, color="#2a2a2a", linewidth=0.8)
    ax.set_xlim(-1.15, 1.15); ax.set_xlabel("Sentiment Score", fontsize=8)
    ax.set_title("Aspect Scores", fontsize=9, pad=6)
    ax.spines[["top","right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=7.5); ax.tick_params(axis="x", labelsize=7.5)
    for i, (label, val) in enumerate(zip(labels, scores)):
        xp = val+0.03 if val >= 0 else val-0.03
        ax.text(xp, i, f"{val:+.2f}", va="center",
                ha="left" if val >= 0 else "right", fontsize=7)
    plt.tight_layout(); return fig


def fig_aspect_pie(profile, max_slices=9):
    items = sorted(profile.items(), key=lambda x: -x[1]["weight"])
    if len(items) > max_slices:
        top   = items[:max_slices]
        rest  = sum(v["weight"] for _, v in items[max_slices:])
        labels  = [a for a, _ in top] + ["other"]
        weights = [v["weight"] for _, v in top] + [rest]
    else:
        labels  = [a for a, _ in items]
        weights = [v["weight"] for _, v in items]
    cmap    = cm.get_cmap("tab20", len(labels))
    colors  = [cmap(i) for i in range(len(labels))]
    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    wedges, _, autotexts = ax.pie(weights, labels=None, autopct="%1.0f%%", colors=colors,
                                   startangle=90, wedgeprops={"linewidth":0.8,"edgecolor":"#0d0d0d"},
                                   pctdistance=0.76)
    for at in autotexts: at.set_fontsize(7); at.set_color("#ddd")
    ax.legend(wedges, labels, loc="lower center", bbox_to_anchor=(0.5,-0.2),
              ncol=3, fontsize=7, frameon=False, labelcolor="#999")
    ax.set_title("Mention Frequency", fontsize=9, pad=6)
    plt.tight_layout(); return fig


def fig_elo_ranking(df_rank):
    vendors = df_rank["vendor"].tolist()[::-1]
    stars   = df_rank["star_rating"].tolist()[::-1]
    ci_lo   = df_rank["ci_lo"].tolist()[::-1]
    ci_hi   = df_rank["ci_hi"].tolist()[::-1]
    palette = ["#f39c12","#aaaaaa","#cd7f32"] + ["#2a4a6a"]*max(0, len(vendors)-3)
    colors  = list(reversed(palette[:len(vendors)]))
    fig, ax = plt.subplots(figsize=(6, max(2.5, len(vendors)*0.5)))
    bars = ax.barh(vendors, stars, color=colors, height=0.55, alpha=0.88)
    ax.errorbar(stars, vendors,
                xerr=[np.array(stars)-np.array(ci_lo), np.array(ci_hi)-np.array(stars)],
                fmt="none", color="#333", capsize=4, lw=1.5)
    ax.set_xlim(0.5, 5.8); ax.axvline(3.0, color="#555", linestyle="--", lw=0.8)
    ax.set_xlabel("Star Rating (Elo-scaled)", fontsize=8)
    ax.set_title("Sentiment-Aware Elo Vendor Ranking", fontsize=9, fontweight="bold")
    ax.spines[["top","right"]].set_visible(False)
    for bar, val in zip(bars, stars):
        ax.text(val+0.08, bar.get_y()+bar.get_height()/2,
                f"{val:.2f}", va="center", fontsize=9, fontweight="bold")
    plt.tight_layout(); return fig


def fig_vendor_bar(df_ratings):
    """Compact horizontal bar chart sorted by rating."""
    vendors = df_ratings["vendor"].tolist()
    ratings = df_ratings["star_rating"].tolist()
    colors  = ["#2ecc71" if r >= 3.5 else ("#f39c12" if r >= 2.5 else "#e74c3c") for r in ratings]
    n       = len(vendors)
    fig, ax = plt.subplots(figsize=(5.5, max(2.5, n * 0.45)))
    bars    = ax.barh(vendors[::-1], ratings[::-1], color=colors[::-1], height=0.55)
    ax.set_xlim(0, 5.8)
    ax.axvline(3.5, color="#2a2a2a", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Star Rating", fontsize=8)
    ax.set_title("Vendor Ratings", fontsize=9, pad=6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=8)
    ax.tick_params(axis="x", labelsize=7.5)
    for bar, val in zip(bars, ratings[::-1]):
        ax.text(val + 0.08, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", fontsize=8, color="#aaa")
    plt.tight_layout()
    return fig


def fig_vendor_heatmap(df_ratings):
    asp_cols = [c for c in df_ratings.columns if c.startswith("asp_")]
    if not asp_cols:
        return None

    MAX_VENDORS = 20
    MAX_ASPECTS = 10

    heat = df_ratings.set_index("vendor")[asp_cols].copy().fillna(0)
    heat.columns = [c[4:] for c in asp_cols]

    if len(heat) > MAX_VENDORS:
        heat = heat.iloc[:MAX_VENDORS]
    if len(heat.columns) > MAX_ASPECTS:
        variance = heat.var(axis=0).sort_values(ascending=False)
        heat = heat[variance.index[:MAX_ASPECTS]]

    n_rows = len(heat)
    n_cols = len(heat.columns)
    fig_w  = min(14, max(4, n_cols * 1.2))
    fig_h  = min(10, max(2.5, n_rows * 0.45))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=90)
    sns.heatmap(
        heat, annot=True, fmt=".2f", cmap="RdYlGn", center=0,
        vmin=-1, vmax=1, ax=ax,
        linewidths=0.3, linecolor="#0d0d0d",
        annot_kws={"size": 8},
        cbar_kws={"shrink": 0.6},
    )
    title = "Per-Vendor Aspect Scores"
    if len(df_ratings) > MAX_VENDORS:
        title += f" (top {MAX_VENDORS} vendors shown)"
    ax.set_title(title, fontsize=9, pad=6)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=30, labelsize=8)
    ax.tick_params(axis="y", rotation=0, labelsize=8)
    plt.tight_layout()
    return fig


def fig_recommendation_podium(df_r):
    """
    Horizontal bar chart sized to fit, coloured by relative rank tier.
    Best vendor = gold, second = silver, third = bronze, rest = grey.
    """
    vendors = df_r["vendor"].tolist()
    ratings = df_r["star_rating"].tolist()
    n       = len(vendors)

    tier_colors = []
    for i in range(n):
        rk = int(df_r.iloc[i]["rank"])
        if rk == 1:
            tier_colors.append("#f39c12")
        elif rk == 2:
            tier_colors.append("#aaaaaa")
        elif rk == 3:
            tier_colors.append("#cd7f32")
        else:
            tier_colors.append("#2a4a6a")

    fig, ax = plt.subplots(figsize=(6, max(2.5, n * 0.48)))
    bars = ax.barh(vendors[::-1], ratings[::-1], color=tier_colors[::-1], height=0.6)
    ax.set_xlim(0.5, 5.5)
    ax.axvline(3.0, color="#2a2a2a", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Relative Star Rating", fontsize=8)
    ax.set_title("Vendor Recommendation Ranking", fontsize=9, pad=6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=8)
    ax.tick_params(axis="x", labelsize=7.5)
    for bar, val in zip(bars, ratings[::-1]):
        ax.text(val + 0.06, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", fontsize=8.5, color="#ffffff", fontweight="bold")
    plt.tight_layout()
    return fig


def fig_radar(profiles_dict):
    """
    Radar chart capped at the top 8 aspects by total mention count.
    Only drawn when there are 12 or fewer vendors. Returns None for larger sets.
    """
    MAX_RADAR_VENDORS = 12

    aspect_totals = defaultdict(int)
    aspect_vendor_count = defaultdict(int)
    for prof in profiles_dict.values():
        for a, v in prof["aspect_profile"].items():
            aspect_totals[a] += v["count"]
            aspect_vendor_count[a] += 1

    shared = [
        a for a, cnt in aspect_vendor_count.items()
        if cnt >= max(2, len(profiles_dict) // 2)
    ]
    shared = sorted(shared, key=lambda a: -aspect_totals[a])[:8]

    if len(shared) < 3:
        return None

    if len(profiles_dict) > MAX_RADAR_VENDORS:
        return None

    N      = len(shared)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    n_vend  = len(profiles_dict)
    fig_h   = 6 + max(0, (n_vend - 6) * 0.18)
    fig, ax = plt.subplots(figsize=(6, fig_h), subplot_kw={"polar": True})
    ax.set_facecolor("#111111")
    fig.patch.set_facecolor("#111111")

    cmap = cm.get_cmap("tab10", n_vend)
    for idx, (vname, prof) in enumerate(profiles_dict.items()):
        values = [prof["aspect_profile"].get(a, {}).get("score", 0.0) for a in shared]
        values = [(v + 1) / 2 for v in values]
        values += values[:1]
        color = cmap(idx)
        ax.plot(angles, values, linewidth=1.8, color=color, label=vname)
        ax.fill(angles, values, alpha=0.08, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(shared, fontsize=9, color="#cccccc")
    ax.set_yticks([0.25, 0.5, 0.75])
    ax.set_yticklabels(["neg", "neutral", "pos"], fontsize=7, color="#555")
    ax.set_ylim(0, 1)
    ax.grid(color="#2a2a2a", linewidth=0.6)
    ax.spines["polar"].set_color("#2a2a2a")

    ncols = min(3, n_vend)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.05 - (n_vend // ncols) * 0.04),
        ncol=ncols,
        fontsize=7.5,
        frameon=False,
        labelcolor="#aaa",
    )
    ax.set_title(
        f"Aspect Comparison (top {N} shared aspects)",
        fontsize=9, pad=16, color="#cccccc",
    )
    plt.tight_layout()
    return fig


def fig_grouped_bar(profiles_dict):
    """
    When there are many vendors, showing one bar group per aspect is cleaner than
    grouping per vendor. For large vendor sets (>12) we switch to showing mean + range per aspect.
    """
    aspect_totals = defaultdict(int)
    for prof in profiles_dict.values():
        for a, v in prof["aspect_profile"].items():
            aspect_totals[a] += v["count"]

    aspects = [a for a, _ in sorted(aspect_totals.items(), key=lambda x: -x[1])[:8]]
    if not aspects:
        return None

    vendors  = list(profiles_dict.keys())
    n_vend   = len(vendors)
    n_asp    = len(aspects)
    MAX_BARS = 10

    if n_vend <= MAX_BARS:
        bar_h   = 0.7 / n_vend
        y_pos   = np.arange(n_asp)
        cmap    = cm.get_cmap("tab10", n_vend)
        fig, ax = plt.subplots(figsize=(7, max(3.5, n_asp * 0.6)))

        for vi, (vname, prof) in enumerate(profiles_dict.items()):
            scores  = [prof["aspect_profile"].get(a, {}).get("score", 0.0) for a in aspects]
            offsets = y_pos - (n_vend - 1) * bar_h / 2 + vi * bar_h
            ax.barh(offsets, scores, height=bar_h * 0.85,
                    color=cmap(vi), label=vname, alpha=0.88)

        ax.axvline(0, color="#3a3a3a", linewidth=0.8)
        ax.set_xlim(-1.15, 1.4)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(aspects, fontsize=9)
        ax.set_xlabel("Sentiment Score", fontsize=8)
        ax.set_title("Aspect Score Comparison", fontsize=9, pad=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", labelsize=7.5)
        ax.legend(loc="lower right", fontsize=7, frameon=False, labelcolor="#aaa", ncol=1)
        plt.tight_layout()
        return fig

    else:
        fig, ax = plt.subplots(figsize=(7, max(3.5, n_asp * 0.6)))
        plotted_asps = []
        plotted_pos  = []

        for ai, asp in enumerate(aspects):
            scores = [
                prof["aspect_profile"].get(asp, {}).get("score", None)
                for prof in profiles_dict.values()
            ]
            scores = [s for s in scores if s is not None]
            if len(scores) < 3:
                continue
            mn   = min(scores)
            mx   = max(scores)
            mean = float(np.mean(scores))
            pos  = len(plotted_asps)
            ax.barh(pos, mx - mn, left=mn, height=0.38, color="#2a4a6a", alpha=0.55)
            ax.plot(mean, pos, "o", color="#f39c12", markersize=7, zorder=3)
            ax.text(1.17, pos, f"n={len(scores)}", va="center", fontsize=7, color="#555")
            plotted_asps.append(asp)
            plotted_pos.append(pos)

        if not plotted_asps:
            plt.close(fig)
            return None

        ax.axvline(0, color="#3a3a3a", linewidth=0.8)
        ax.set_xlim(-1.15, 1.35)
        ax.set_yticks(plotted_pos)
        ax.set_yticklabels(plotted_asps, fontsize=9)
        ax.set_xlabel("Sentiment Score", fontsize=8)
        ax.set_title(
            f"Aspect Score Range across {n_vend} vendors\n"
            "bar = min to max   dot = mean   n = vendors with data",
            fontsize=8.5, pad=8,
        )
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", labelsize=7.5)
        plt.tight_layout()
        return fig


def fig_confidence_hist(confs, title="Confidence Distribution"):
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.hist(confs, bins=30, color="#3498db", edgecolor="#0d0d0d", alpha=0.85)
    mean_c = float(np.mean(confs))
    ax.axvline(mean_c, color="#e74c3c", linestyle="--", linewidth=1.2,
               label=f"Mean: {mean_c:.3f}")
    ax.set_xlabel("Confidence Score", fontsize=8)
    ax.set_ylabel("Frequency", fontsize=8)
    ax.set_title(title, fontsize=9, pad=6)
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return fig


def fig_sentiment_dist(label_counts, title="Sentiment Distribution"):
    labels = SENTIMENT_LABEL_NAMES
    counts = [label_counts.get(l, 0) for l in labels]
    fig, ax = plt.subplots(figsize=(5.5, 3))
    bars = ax.bar(labels, counts, color=SENT_PALETTE, width=0.55)
    ax.set_ylabel("Count", fontsize=8)
    ax.set_title(title, fontsize=9, pad=6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)
    for bar, val in zip(bars, counts):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    val + max(counts) * 0.02,
                    str(val), ha="center", fontsize=8, color="#aaa")
    plt.tight_layout()
    return fig


def fig_aspects_per_review(aspects_per_review):
    cnt    = Counter(aspects_per_review)
    keys   = sorted(cnt.keys())
    values = [cnt[k] for k in keys]
    fig, ax = plt.subplots(figsize=(5.5, 3))
    ax.bar([str(k) for k in keys], values, color="#3498db", width=0.6)
    ax.set_xlabel("Aspects extracted per review", fontsize=8)
    ax.set_ylabel("Number of reviews", fontsize=8)
    ax.set_title("ATE: Aspects per Review", fontsize=9, pad=6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)
    plt.tight_layout()
    return fig


def fig_low_confidence_by_vendor(vendor_low_conf, threshold=0.6):
    if not vendor_low_conf:
        return None
    vendors = list(vendor_low_conf.keys())
    pcts    = [vendor_low_conf[v] for v in vendors]
    colors  = ["#e74c3c" if p > 30 else "#f39c12" if p > 15 else "#2ecc71"
               for p in pcts]
    fig, ax = plt.subplots(figsize=(6, max(3, len(vendors) * 0.38)))
    ax.barh(vendors[::-1], pcts[::-1], color=colors[::-1], height=0.6)
    ax.axvline(15, color="#f39c12", linestyle="--", linewidth=0.8, label="15% warning")
    ax.axvline(30, color="#e74c3c", linestyle="--", linewidth=0.8, label="30% critical")
    ax.set_xlabel(f"% predictions with confidence < {threshold}", fontsize=8)
    ax.set_title("Low-Confidence Predictions per Vendor", fontsize=9, pad=6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=7.5)
    ax.legend(loc="upper right", fontsize=7.5, frameon=False, labelcolor="#aaa")
    fig.subplots_adjust(bottom=0.18)
    return fig


def fig_score_distribution(scores):
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.hist(scores, bins=40, color="#2ecc71", edgecolor="#0d0d0d", alpha=0.85)
    mean_s = float(np.mean(scores))
    ax.axvline(mean_s, color="#f39c12", linestyle="--", linewidth=1.2,
               label=f"Mean: {mean_s:+.3f}")
    ax.axvline(0, color="#555", linewidth=0.8)
    ax.set_xlabel("Weighted Sentiment Score", fontsize=8)
    ax.set_ylabel("Frequency", fontsize=8)
    ax.set_title("Distribution of Weighted Sentiment Scores", fontsize=9, pad=6)
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return fig


def fig_vendor_confidence_box(vendor_confs):
    """Box plot of confidence per vendor, bottom 20 by median confidence."""
    if not vendor_confs:
        return None
    sorted_v = sorted(vendor_confs.items(),
                      key=lambda x: float(np.median(x[1])))[:20]
    labels = [v for v, _ in sorted_v]
    data   = [c for _, c in sorted_v]
    fig, ax = plt.subplots(figsize=(7, max(3.5, len(labels) * 0.38)))
    ax.boxplot(data, vert=False, patch_artist=True,
               medianprops={"color": "#f39c12", "linewidth": 1.5},
               boxprops={"facecolor": "#1a3a5a", "alpha": 0.8},
               whiskerprops={"color": "#555"},
               capprops={"color": "#555"},
               flierprops={"marker": "o", "color": "#00e5ff",
                           "markersize": 4, "alpha": 0.7,
                           "markeredgecolor": "none"})
    ax.set_yticks(range(1, len(labels) + 1))
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlabel("Prediction Confidence", fontsize=8)
    ax.set_title("Confidence Distribution per Vendor (bottom 20 by median)",
                 fontsize=9, pad=6)
    ax.axvline(0.6, color="#e74c3c", linestyle="--", linewidth=0.8, label="0.6 threshold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=7.5, frameon=False, labelcolor="#aaa")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Composite UI helpers
# ---------------------------------------------------------------------------

def render_aspect_breakdown(profile):
    """Renders aspect scores bar chart and mention-frequency pie chart side by side."""
    c1, c2 = st.columns([3, 2])
    with c1:
        st.pyplot(fig_aspect_scores(profile))
    with c2:
        st.pyplot(fig_aspect_pie(profile))


def render_aspect_table(profile, show_weight=False):
    """Renders the per-aspect detail table."""
    rows = []
    for a, v in sorted(profile.items(), key=lambda x: -x[1]["count"]):
        row = {
            "aspect":    a,
            "sentiment": v["dominant"],
            "score":     round(v["score"], 4),
            "mentions":  v["count"],
        }
        if show_weight:
            row["weight"] = f"{v['weight']:.1%}"
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), use_container_width=True)


def render_vendor_drill(profiles_dict, selectbox_label, selectbox_key, show_table=True):
    """Renders the vendor drill-down block with selectbox, rating badge, aspect charts."""
    selected = st.selectbox(selectbox_label, list(profiles_dict.keys()), key=selectbox_key)
    if selected and selected in profiles_dict:
        prf = profiles_dict[selected]["aspect_profile"]
        rating_badge(
            selected,
            profiles_dict[selected]["star_rating"],
            profiles_dict[selected]["review_count"],
        )
        render_aspect_breakdown(prf)
        if show_table:
            render_aspect_table(prf)


def render_heatmap(df_ratings):
    """Renders the aspect score heatmap section including the section header."""
    hmap = fig_vendor_heatmap(df_ratings)
    if hmap:
        sec("Aspect Score Heatmap")
        st.pyplot(hmap)


def recommendation_card_html(df_r, profiles_dict):
    """Top recommendation card with explanation of why this vendor is ranked first."""
    if df_r.empty:
        return ""
    top_row  = df_r.iloc[0]
    top_name = top_row["vendor"]
    top_star = top_row["star_rating"]
    top_prof = profiles_dict[top_name]["aspect_profile"]

    by_score = sorted(top_prof.items(), key=lambda x: -x[1]["score"])
    strong   = [a for a, v in by_score if v["score"] >= 0.3][:3]
    weak     = [a for a, v in by_score if v["score"] < 0][-2:]

    strong_txt = ", ".join(strong) if strong else "no standout aspects"
    weak_txt   = ", ".join(weak)   if weak   else "none identified"

    bar_pct = int((top_star - 1) / 4 * 100)

    return f"""
    <div style="background:linear-gradient(135deg,#0d1a0d 0%,#111 100%);
                border:1px solid #2ecc71;border-radius:6px;padding:1.6rem 2rem;margin-bottom:1.4rem;">
        <div style="font-family:IBM Plex Mono,monospace;font-size:0.62rem;text-transform:uppercase;
                    letter-spacing:0.14em;color:#2ecc71;margin-bottom:0.5rem;">Top Recommendation</div>
        <div style="font-family:IBM Plex Mono,monospace;font-size:1.8rem;font-weight:600;
                    color:#ffffff;line-height:1.1;margin-bottom:0.3rem;">{top_name}</div>
        <div style="font-family:IBM Plex Mono,monospace;font-size:2.4rem;font-weight:600;
                    color:#f39c12;line-height:1;">{top_star:.2f} <span style="font-size:1rem;color:#555;">/ 5.00</span></div>
        <div style="background:#1a1a1a;height:4px;border-radius:2px;overflow:hidden;margin:0.6rem 0;">
            <div style="width:{bar_pct}%;background:#f39c12;height:100%;border-radius:2px;"></div>
        </div>
        <div style="font-family:IBM Plex Sans,sans-serif;font-size:0.82rem;color:#666;margin-top:0.7rem;line-height:1.7;">
            <span style="color:#aaa;">Strong aspects:</span> {strong_txt}<br>
            <span style="color:#aaa;">Weak aspects:</span> {weak_txt}
        </div>
    </div>
    """


def leaderboard_html(df, show_recommendation=False):
    asp_cols     = [c for c in df.columns if c.startswith("asp_")]
    base_headers = ["Rank", "Vendor", "Stars", "Relative Rating"]
    if show_recommendation:
        base_headers.append("Recommended")
    headers      = base_headers + [c[4:] for c in asp_cols]
    th           = "padding:0.45rem 0.8rem;text-align:left;color:#3a3a3a;font-size:0.6rem;text-transform:uppercase;letter-spacing:0.1em;border-bottom:1px solid #161616;font-family:IBM Plex Mono,monospace;"
    td           = "padding:0.4rem 0.8rem;border-bottom:1px solid #111;font-family:IBM Plex Mono,monospace;font-size:0.8rem;"
    header_html  = "".join(f'<th style="{th}">{h}</th>' for h in headers)
    rows_html    = ""
    rank_colors  = ["#f39c12", "#aaaaaa", "#cd7f32"]
    for _, row in df.iterrows():
        r    = row["star_rating"]
        bp   = int((r - 1) / 4 * 100)
        bc   = "#2ecc71" if r >= 4.0 else ("#f39c12" if r >= 2.5 else "#e74c3c")
        rk   = int(row["rank"])
        rc   = rank_colors[rk - 1] if rk <= 3 else "#3a3a3a"
        atd  = ""
        if show_recommendation:
            rec_txt = "Best choice" if rk == 1 else ("Good choice" if rk == 2 else "")
            rec_col = "#2ecc71" if rk == 1 else ("#3498db" if rk == 2 else "#2a2a2a")
            atd += f'<td style="{td}color:{rec_col};">{rec_txt}</td>'
        for c in asp_cols:
            v = row.get(c, None)
            if pd.isna(v):
                atd += f'<td style="{td}color:#1e1e1e;">-</td>'
            else:
                vc = "#2ecc71" if v >= 0 else "#e74c3c"
                atd += f'<td style="{td}color:{vc};">{v:+.2f}</td>'
        rows_html += f"""
        <tr>
          <td style="{td}color:{rc};font-weight:600;">#{rk}</td>
          <td style="{td}color:#e8e8e8;font-weight:600;">{row['vendor']}</td>
          <td style="{td}">{stars_display(r, "0.95rem")}</td>
          <td style="{td}">
            <span style="font-weight:600;color:#fff;margin-right:0.45rem;">{r:.2f}</span>
            <span style="display:inline-block;width:60px;height:3px;background:#161616;border-radius:2px;vertical-align:middle;overflow:hidden;">
              <span style="display:block;width:{bp}%;height:100%;background:{bc};"></span>
            </span>
          </td>
          {atd}
        </tr>"""
    return (
        f'<div class="leaderboard-wrap">'
        f'<table class="lb-table">'
        f'<thead><tr style="background:#090909;">{header_html}</tr></thead>'
        f'<tbody>{rows_html}</tbody>'
        f'</table></div>'
    )


# ---------------------------------------------------------------------------
# File loading
# ---------------------------------------------------------------------------

def load_reviews_from_file(uploaded):
    raw = uploaded.read()
    if uploaded.name.endswith(".txt"):
        lines = raw.decode("utf-8", errors="ignore").splitlines()
        return [l.strip() for l in lines if l.strip()]
    df = pd.read_csv(io.BytesIO(raw))
    if "text" not in df.columns:
        err(f"CSV must have a 'text' column. Found: {df.columns.tolist()}")
        return []
    df["text"] = df["text"].astype(str).str.strip()
    return df[df["text"].str.len() > 0]["text"].tolist()


# ---------------------------------------------------------------------------
# Sidebar: model selection
# ---------------------------------------------------------------------------

sb_head = ("font-family:IBM Plex Mono,monospace;font-size:0.62rem;text-transform:uppercase;"
           "letter-spacing:0.13em;color:#3a3a3a;padding-bottom:0.4rem;"
           "border-bottom:1px solid #161616;margin-bottom:0.9rem;")

with st.sidebar:
    st.markdown(f'<div style="{sb_head}">Model Selection</div>', unsafe_allow_html=True)

    avail = available_models()

    if not avail:
        warn("No trained models found. Run the training notebooks first.")
        model_options = []
    else:
        model_options = list(avail.keys())

    pipeline_mode = st.radio(
        "Pipeline mode",
        ["Single model", "Ensemble (all available)"],
        key="pipeline_mode"
    )

    selected_model = None
    if model_options and pipeline_mode == "Single model":
        selected_model = st.selectbox("Select model", model_options, key="model_select")

    if model_options and st.button("Load Model(s)", type="primary"):
        if pipeline_mode == "Single model" and selected_model:
            try:
                dirs = avail[selected_model]
                tok_ate, mdl_ate, tok_sent, mdl_sent = load_single_model(
                    dirs["ate"], dirs["sent"]
                )
                st.session_state["active_ate_model"]   = mdl_ate
                st.session_state["active_ate_tok"]     = tok_ate
                st.session_state["active_sent_model"]  = mdl_sent
                st.session_state["active_sent_tok"]    = tok_sent
                st.session_state["active_model_name"]  = selected_model
                st.session_state["ensemble_models"]    = None
                st.session_state["_pipeline_mode_val"] = "single"
                st.sidebar.success(f"Loaded: {selected_model}")
            except Exception as e:
                st.sidebar.error(f"Error: {e}")

        elif pipeline_mode == "Ensemble (all available)":
            ensemble_dict = {}
            errors = []
            for name, dirs in avail.items():
                try:
                    tok_ate, mdl_ate, tok_sent, mdl_sent = load_single_model(
                        dirs["ate"], dirs["sent"]
                    )
                    ensemble_dict[name] = {
                        "tok_ate": tok_ate, "mdl_ate": mdl_ate,
                        "tok_sent": tok_sent, "mdl_sent": mdl_sent,
                    }
                except Exception as e:
                    errors.append(f"{name}: {e}")
            if ensemble_dict:
                # Write to a separate key to avoid mutating the widget-bound key
                st.session_state["active_ate_model"]   = list(ensemble_dict.values())[0]["mdl_ate"]
                st.session_state["active_ate_tok"]     = list(ensemble_dict.values())[0]["tok_ate"]
                st.session_state["active_sent_model"]  = list(ensemble_dict.values())[0]["mdl_sent"]
                st.session_state["active_sent_tok"]    = list(ensemble_dict.values())[0]["tok_sent"]
                st.session_state["active_model_name"]  = "Ensemble"
                st.session_state["ensemble_models"]    = ensemble_dict
                st.session_state["_pipeline_mode_val"] = "ensemble"
                st.sidebar.success(f"Ensemble: {list(ensemble_dict.keys())}")
            for e in errors:
                st.sidebar.warning(e)

    st.markdown(f'<div style="{sb_head};margin-top:1.3rem;">Status</div>', unsafe_allow_html=True)
    active = st.session_state.get("active_model_name", None)
    pm     = st.session_state.get("_pipeline_mode_val", None)
    if active:
        dot = "#2ecc71"; col = "#2ecc71"
        txt = f"{active} ({pm})"
    else:
        dot = "#222"; col = "#2a2a2a"; txt = "not loaded"
    st.markdown(
        f'<div style="font-family:IBM Plex Mono,monospace;font-size:0.75rem;">'
        f'<span style="display:inline-block;width:6px;height:6px;border-radius:50%;'
        f'background:{dot};margin-right:6px;vertical-align:middle;"></span>'
        f'<span style="color:{col};">{txt}</span></div>', unsafe_allow_html=True
    )
    st.markdown(f'<div style="{sb_head};margin-top:1.3rem;">Settings</div>', unsafe_allow_html=True)
    use_norm   = st.checkbox("Normalise aspect terms", value=True)
    use_elo    = st.checkbox("Elo ranking (recommendation)", value=True,
                             help="Uses Sentiment-Aware Elo. Disable for simple relative scaling.")
    st.sidebar.write("Device:", str(DEVICE))


# ---------------------------------------------------------------------------
# Helper: run pipeline in correct mode
# ---------------------------------------------------------------------------

def _run(text, ate_model=None, ate_tok=None, sent_model=None, sent_tok=None):
    """Run pipeline in single or ensemble mode based on session state."""
    pm = st.session_state.get("_pipeline_mode_val", "single")
    if pm == "ensemble":
        em = st.session_state.get("ensemble_models", {})
        return run_ensemble_pipeline(text, em)
    return run_pipeline(text, ate_model, ate_tok, sent_model, sent_tok)


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Single Review", "Single Vendor", "Recommendation",
    "Compare Vendors", "Model Evaluation", "Budget Planner", "How It Works"
])

# =============================================================================
# Tab 1: Single Review
# =============================================================================

with tab1:
    sec("Analyse a Single Review")
    info("Paste any review. The ATE model extracts aspect terms automatically.")
    if not models_ready():
        warn("Load models using the sidebar first.")

    review_text = st.text_area("Review",
        placeholder="The food was amazing but the service was really slow.",
        height=110, key="t1_input")

    if st.button("Analyse", type="primary", key="t1_run") and review_text.strip():
        if not models_ready():
            err("Models not loaded.")
        else:
            ate_mdl, ate_tok, sent_mdl, sent_tok = get_models()
            with st.spinner("Running pipeline..."):
                res = _run(review_text.strip(), ate_mdl, ate_tok, sent_mdl, sent_tok)
            st.session_state["t1_results"] = res

    if "t1_results" in st.session_state:
        res = st.session_state["t1_results"]
        if not res:
            st.markdown('<div class="no-asp">No aspect terms detected.</div>', unsafe_allow_html=True)
        else:
            overall = float(np.mean([r["weighted_score"] for r in res]))
            star    = round(float(np.clip((overall+1)/2*5, 0, 5)), 2)
            stat_row(("Aspects found", len(res)), ("Overall score", f"{overall:+.3f}"),
                     ("Star equivalent", f"{star:.2f} / 5"),
                     ("Positive", sum(1 for r in res if r["sentiment"]=="positive"),  "#2ecc71"),
                     ("Negative", sum(1 for r in res if r["sentiment"]=="negative"),  "#e74c3c"),
                     ("Neutral",  sum(1 for r in res if r["sentiment"]=="neutral"),   "#3498db"),
                     ("Conflict", sum(1 for r in res if r["sentiment"]=="conflict"),  "#f39c12"))
            sec("Extracted Aspects"); aspect_cards(res)
            sec("Probability Breakdown")
            cols = st.columns(min(4, len(res)))
            for i, r in enumerate(res):
                with cols[i % len(cols)]:
                    st.pyplot(fig_prob_bars(r["probs"], r["aspect"]))
            sec("Results Table")
            st.dataframe(pd.DataFrame([{"aspect": r["aspect"], "sentiment": r["sentiment"],
                "score": round(r["weighted_score"],4), "confidence": round(r["confidence"],4)}
                for r in res]), use_container_width=True)


# =============================================================================
# Tab 2: Single Vendor
# =============================================================================

with tab2:
    sec("Rate a Single Vendor")
    info("Upload a CSV with a <code>text</code> column, or a TXT file (one review per line).")
    if not models_ready(): warn("Load models using the sidebar first.")

    vf = st.file_uploader("Reviews file (CSV or TXT)", type=["csv","txt"], key="t2_upload")
    vname_override = st.text_input("Vendor name (optional)", key="t2_vname")

    if vf:
        reviews_t2 = load_reviews_from_file(vf)
        if reviews_t2:
            vname_t2 = vname_override.strip() or os.path.splitext(vf.name)[0]
            ok(f"Loaded {len(reviews_t2)} reviews for: {vname_t2}")
            if st.button("Run Analysis", type="primary", key="t2_run"):
                if not models_ready(): err("Models not loaded.")
                else:
                    ate_mdl, ate_tok, sent_mdl, sent_tok = get_models()
                    flat_t2 = []
                    prog_t2 = st.progress(0)
                    for i, rev in enumerate(reviews_t2):
                        flat_t2.extend(_run(rev, ate_mdl, ate_tok, sent_mdl, sent_tok))
                        prog_t2.progress((i+1)/len(reviews_t2))
                    prof = compute_vendor_profile(flat_t2, len(reviews_t2), use_norm=use_norm)
                    if prof:
                        star = round(float(np.clip((prof["raw_score"]+1)/2*5, 0, 5)), 2)
                        st.session_state.update({"t2_prof": prof, "t2_star": star, "t2_name": vname_t2})

    if "t2_prof" in st.session_state:
        prof = st.session_state["t2_prof"]
        star = st.session_state["t2_star"]
        vn   = st.session_state["t2_name"]
        sec(f"Rating: {vn}")
        rating_badge(vn, star, prof["review_count"])
        stat_row(("Raw score", f"{prof['raw_score']:+.3f}"),
                 ("Total aspects", sum(v["count"] for v in prof["aspect_profile"].values())),
                 ("Unique aspects", len(prof["aspect_profile"])))
        sec("Aspect Breakdown")
        c1, c2 = st.columns([3,2])
        with c1: st.pyplot(fig_aspect_scores(prof["aspect_profile"]))
        with c2: st.pyplot(fig_aspect_pie(prof["aspect_profile"]))


# =============================================================================
# Tab 3: Recommendation (Elo ranking)
# =============================================================================

with tab3:
    sec("Vendor Recommendation System")
    info("Ratings use Sentiment-Aware Elo ranking. Best vendor always scores 5.0, "
         "worst always scores 1.0.")
    if not models_ready(): warn("Load models using the sidebar first.")

    upload_mode_t3 = st.radio("Upload mode",
        ["Multiple files (one per vendor)", "Single CSV with vendor and text columns"],
        key="t3_mode", horizontal=True)

    vendor_map_rec = {}
    if upload_mode_t3 == "Multiple files (one per vendor)":
        rec_files = st.file_uploader("Upload vendor review files", type=["csv","txt"],
                                     accept_multiple_files=True, key="t3_upload_multi")
        if rec_files:
            for f in rec_files:
                revs = load_reviews_from_file(f)
                if revs: vendor_map_rec[os.path.splitext(f.name)[0]] = revs
    else:
        single_csv = st.file_uploader("Combined CSV (vendor + text columns)",
                                      type=["csv"], key="t3_upload_single")
        if single_csv:
            raw_sc = single_csv.read()
            df_sc  = pd.read_csv(io.BytesIO(raw_sc))
            if {"vendor","text"} <= set(df_sc.columns):
                df_sc["vendor"] = df_sc["vendor"].astype(str).str.strip()
                df_sc["text"]   = df_sc["text"].astype(str).str.strip()
                df_sc = df_sc[df_sc["text"].str.len() > 0]
                for vn, grp in df_sc.groupby("vendor", sort=False):
                    vendor_map_rec[vn] = grp["text"].tolist()
            else:
                err("CSV must have 'vendor' and 'text' columns.")

    if vendor_map_rec:
        ok(f"Loaded {sum(len(v) for v in vendor_map_rec.values())} reviews "
           f"across {len(vendor_map_rec)} vendor(s).")

    if vendor_map_rec and st.button("Run Recommendation", type="primary", key="t3_run"):
        if not models_ready(): err("Models not loaded.")
        else:
            ate_mdl, ate_tok, sent_mdl, sent_tok = get_models()
            total   = sum(len(v) for v in vendor_map_rec.values())
            done    = 0
            prog    = st.progress(0)
            status  = st.empty()
            profs   = {}
            ranker  = SentimentEloRanker(k=32, initial=1500, bayes_m=10)
            for vn, reviews in vendor_map_rec.items():
                flat = []
                for rev in reviews:
                    status.markdown(
                        f'<div style="font-family:IBM Plex Mono,monospace;font-size:0.72rem;'
                        f'color:#3a3a3a;">Processing {vn} ({done+1}/{total})</div>',
                        unsafe_allow_html=True)
                    flat.extend(_run(rev, ate_mdl, ate_tok, sent_mdl, sent_tok))
                    done += 1; prog.progress(done/total)
                prof = compute_vendor_profile(flat, len(reviews), use_norm=use_norm)
                if prof:
                    profs[vn] = prof
                    ranker.add_vendor(vn, prof)
            status.empty()
            df_rank = ranker.get_rankings()
            if not df_rank.empty:
                # Merge aspect cols
                rows = []
                for vn, prof in profs.items():
                    row = {"vendor": vn}
                    for a, av in prof["aspect_profile"].items():
                        row[f"asp_{a}"] = round(av["score"], 3)
                    rows.append(row)
                df_asp = pd.DataFrame(rows)
                df_rank = df_rank.merge(df_asp, on="vendor", how="left")
            st.session_state["t3_rank"]  = df_rank
            st.session_state["t3_profs"] = profs

    if "t3_rank" in st.session_state:
        df_r = st.session_state["t3_rank"]
        prfs = st.session_state["t3_profs"]

        if not df_r.empty:
            top = df_r.iloc[0]
            sec("Top Recommendation")
            rating_badge(top["vendor"], top["star_rating"], top["reviews"])

            sec("Elo Leaderboard")
            st.dataframe(df_r[["rank","vendor","star_rating","elo_rating","raw_score","reviews"]],
                         use_container_width=True, hide_index=True)

            sec("Ranking Chart")
            st.pyplot(fig_elo_ranking(df_r))

            sec("Vendor Drill Down")
            selected_vn = st.selectbox("Select vendor", list(prfs.keys()), key="t3_drill")
            if selected_vn in prfs:
                p = prfs[selected_vn]["aspect_profile"]
                c1, c2 = st.columns([3,2])
                with c1: st.pyplot(fig_aspect_scores(p))
                with c2: st.pyplot(fig_aspect_pie(p))


# =============================================================================
# Tab 4: Compare Vendors (absolute)
# =============================================================================

with tab4:
    sec("Compare Vendors (Absolute Scores)")
    info("Raw scores mapped from [-1,+1] to [0,5]. Use this for absolute comparison "
         "rather than relative ranking.")
    if not models_ready(): warn("Load models using the sidebar first.")

    upload_mode_t4 = st.radio("Upload mode",
        ["Multiple files (one per vendor)", "Single CSV with vendor and text columns"],
        key="t4_mode", horizontal=True)

    vendor_map_cmp = {}
    if upload_mode_t4 == "Multiple files (one per vendor)":
        cmp_files = st.file_uploader("Upload vendor review files", type=["csv","txt"],
                                     accept_multiple_files=True, key="t4_upload")
        if cmp_files:
            for f in cmp_files:
                revs = load_reviews_from_file(f)
                if revs: vendor_map_cmp[os.path.splitext(f.name)[0]] = revs
    else:
        single_csv_t4 = st.file_uploader("Combined CSV", type=["csv"], key="t4_upload_single")
        if single_csv_t4:
            raw_t4 = single_csv_t4.read()
            df_t4s = pd.read_csv(io.BytesIO(raw_t4))
            if {"vendor","text"} <= set(df_t4s.columns):
                df_t4s["vendor"] = df_t4s["vendor"].astype(str).str.strip()
                df_t4s["text"]   = df_t4s["text"].astype(str).str.strip()
                df_t4s = df_t4s[df_t4s["text"].str.len() > 0]
                for vn, grp in df_t4s.groupby("vendor", sort=False):
                    vendor_map_cmp[vn] = grp["text"].tolist()

    if vendor_map_cmp:
        ok(f"Loaded {sum(len(v) for v in vendor_map_cmp.values())} reviews.")

    if vendor_map_cmp and st.button("Run Comparison", type="primary", key="t4_run"):
        if not models_ready(): err("Models not loaded.")
        else:
            ate_mdl, ate_tok, sent_mdl, sent_tok = get_models()
            total = sum(len(v) for v in vendor_map_cmp.values())
            done  = 0; prog = st.progress(0); status = st.empty()
            rows_c = {}
            for vn, reviews in vendor_map_cmp.items():
                flat = []
                for rev in reviews:
                    status.markdown(
                        f'<div style="font-family:IBM Plex Mono,monospace;font-size:0.72rem;'
                        f'color:#3a3a3a;">Processing {vn} ({done+1}/{total})</div>',
                        unsafe_allow_html=True)
                    flat.extend(_run(rev, ate_mdl, ate_tok, sent_mdl, sent_tok))
                    done += 1; prog.progress(done/total)
                prof = compute_vendor_profile(flat, len(reviews), use_norm=use_norm)
                if prof:
                    star = round(float(np.clip((prof["raw_score"]+1)/2*5, 0, 5)), 2)
                    rows_c[vn] = {**prof, "star_rating": star}
            status.empty()
            df_c4 = pd.DataFrame([
                {"vendor": vn, "star_rating": v["star_rating"], "raw_score": v["raw_score"],
                 "reviews": v["review_count"]} for vn, v in rows_c.items()
            ]).sort_values("star_rating", ascending=False).reset_index(drop=True)
            df_c4.insert(0, "rank", range(1, len(df_c4)+1))
            st.session_state["t4_df"]   = df_c4
            st.session_state["t4_profs"] = rows_c

    if "t4_df" in st.session_state:
        df_c4  = st.session_state["t4_df"]
        prfs_c = st.session_state["t4_profs"]
        sec("Leaderboard (Absolute)")
        st.dataframe(df_c4, use_container_width=True, hide_index=True)
        sec("Vendor Drill Down")
        sel = st.selectbox("Select vendor", list(prfs_c.keys()), key="t4_drill")
        if sel in prfs_c:
            p = prfs_c[sel]["aspect_profile"]
            c1, c2 = st.columns([3,2])
            with c1: st.pyplot(fig_aspect_scores(p))
            with c2: st.pyplot(fig_aspect_pie(p))


# =============================================================================
# Tab 5: Model Evaluation
# =============================================================================

with tab5:
    sec("Model Evaluation")
    info("Upload reviews and evaluate confidence, sentiment distribution, and aspect extraction rate.")
    if not models_ready(): warn("Load models using the sidebar first.")

    eval_upload = st.file_uploader("Upload CSV (text + optional vendor columns)",
                                   type=["csv","txt"], accept_multiple_files=True, key="t5_upload")
    conf_threshold = st.slider("Low-confidence threshold", 0.4, 0.9, 0.6, 0.05, key="t5_thresh")

    eval_vendor_map = {}
    if eval_upload:
        for f in eval_upload:
            revs = load_reviews_from_file(f)
            if revs: eval_vendor_map[os.path.splitext(f.name)[0]] = revs

    if eval_vendor_map:
        ok(f"Loaded {sum(len(v) for v in eval_vendor_map.values())} reviews.")

    if eval_vendor_map and st.button("Run Evaluation", type="primary", key="t5_run"):
        if not models_ready(): err("Models not loaded.")
        else:
            ate_mdl, ate_tok, sent_mdl, sent_tok = get_models()
            all_confs, all_scores, all_sents, all_asp_counts = [], [], [], []
            vendor_confs, vendor_low_pct = {}, {}
            total = sum(len(v) for v in eval_vendor_map.values())
            done  = 0; prog = st.progress(0); status = st.empty()

            for vn, reviews in eval_vendor_map.items():
                v_confs = []
                for rev in reviews:
                    status.markdown(
                        f'<div style="font-family:IBM Plex Mono,monospace;font-size:0.72rem;'
                        f'color:#3a3a3a;">Evaluating {vn} ({done+1}/{total})</div>',
                        unsafe_allow_html=True)
                    results = _run(rev, ate_mdl, ate_tok, sent_mdl, sent_tok)
                    all_asp_counts.append(len(results))
                    for r in results:
                        all_confs.append(r["confidence"]); all_scores.append(r["weighted_score"])
                        all_sents.append(r["sentiment"]); v_confs.append(r["confidence"])
                    done += 1; prog.progress(done/total)
                vendor_confs[vn] = v_confs
                if v_confs:
                    vendor_low_pct[vn] = round(
                        sum(1 for c in v_confs if c < conf_threshold) / len(v_confs) * 100, 1)
            status.empty()

            st.session_state["t5_eval"] = {
                "all_confs": all_confs, "all_scores": all_scores,
                "all_sents": all_sents, "all_asp_counts": all_asp_counts,
                "vendor_confs": vendor_confs, "vendor_low_pct": vendor_low_pct,
                "n_reviews": total, "thresh": conf_threshold,
            }

    if "t5_eval" in st.session_state:
        ev = st.session_state["t5_eval"]
        total_preds = len(ev["all_confs"])
        low_conf    = sum(1 for c in ev["all_confs"] if c < ev["thresh"])
        zero_asp    = sum(1 for c in ev["all_asp_counts"] if c == 0)
        mean_asp    = float(np.mean(ev["all_asp_counts"])) if ev["all_asp_counts"] else 0

        sec("Overall Statistics")
        stat_row(
            ("Reviews",          ev["n_reviews"]),
            ("Total predictions",total_preds),
            ("Mean confidence",  f"{np.mean(ev['all_confs']):.3f}" if ev["all_confs"] else "N/A"),
            ("Low confidence",   f"{low_conf} ({low_conf/max(total_preds,1)*100:.1f}%)", "#f39c12"),
            ("Zero-aspect revs", zero_asp, "#e74c3c"),
            ("Mean aspects/rev", f"{mean_asp:.2f}"),
        )

        # ---- Sentiment distribution ----
        sec("Sentiment Distribution (All Predictions)")
        sent_counts = Counter(ev["all_sents"])
        c1, c2 = st.columns(2)
        with c1:
            st.pyplot(fig_sentiment_dist(sent_counts))
        with c2:
            df_sent = pd.DataFrame([
                {"sentiment": s,
                 "count": sent_counts.get(s, 0),
                 "percentage": f"{sent_counts.get(s,0)/max(total_preds,1)*100:.1f}%"}
                for s in SENTIMENT_LABEL_NAMES
            ])
            st.dataframe(df_sent, use_container_width=True, hide_index=True)

        # ---- ATE stats ----
        sec("ATE: Aspect Extraction Statistics")
        c3, c4 = st.columns(2)
        with c3:
            st.pyplot(fig_aspects_per_review(ev["all_asp_counts"]))
        with c4:
            asp_dist = Counter(ev["all_asp_counts"])
            df_asp = pd.DataFrame([
                {"aspects per review": k,
                 "reviews": asp_dist[k],
                 "percentage": f"{asp_dist[k]/max(ev['n_reviews'],1)*100:.1f}%"}
                for k in sorted(asp_dist)
            ])
            st.dataframe(df_asp, use_container_width=True, hide_index=True)

        # ---- Confidence distribution ----
        sec("Confidence Score Distribution")
        c5, c6 = st.columns(2)
        with c5:
            if ev["all_confs"]:
                st.pyplot(fig_confidence_hist(ev["all_confs"]))
        with c6:
            if ev["all_scores"]:
                st.pyplot(fig_score_distribution(ev["all_scores"]))

        # ---- Per-vendor breakdown ----
        sec("Per-Vendor Reliability Breakdown")
        info(
            f"Vendors with more than 30% of predictions below confidence {ev['thresh']} "
            f"are highlighted in red."
        )
        fig_lc = fig_low_confidence_by_vendor(ev["vendor_low_pct"], threshold=ev["thresh"])
        if fig_lc:
            st.pyplot(fig_lc)

        fig_box = fig_vendor_confidence_box(ev["vendor_confs"])
        if fig_box:
            st.pyplot(fig_box)

        # ---- Per-vendor summary table ----
        sec("Per-Vendor Summary Table")
        if ev["vendor_low_pct"]:
            df_vlcp = pd.DataFrame([
                {"vendor": vn, "low_conf_pct": pct, "n_predictions": len(ev["vendor_confs"][vn])}
                for vn, pct in sorted(ev["vendor_low_pct"].items(), key=lambda x: -x[1])
            ])
            st.dataframe(df_vlcp, use_container_width=True, hide_index=True)

        # ---- Download ----
        sec("Download")
        raw_rows = []
        for vn, v_confs in ev["vendor_confs"].items():
            for conf in v_confs:
                raw_rows.append({
                    "vendor": vn,
                    "confidence": round(conf, 4),
                    "low_confidence": conf < ev["thresh"],
                })
        if raw_rows:
            st.download_button(
                "Download Evaluation CSV",
                pd.DataFrame(raw_rows).to_csv(index=False).encode(),
                "evaluation_predictions.csv",
                "text/csv",
            )


# =============================================================================
# Tab 6: Budget Planner
# =============================================================================

# ---------------------------------------------------------------------------
# Budget Planner: imports and helpers (self-contained inside this file)
# ---------------------------------------------------------------------------

import itertools
from dataclasses import dataclass, field
from scipy.special import softmax as _softmax

try:
    import pulp as _pulp
    _ILP_AVAILABLE = True
except ImportError:
    _ILP_AVAILABLE = False


@dataclass
class _Vendor:
    name:           str
    category:       str
    price:          float
    absa_score:     float      # raw score in [-1, +1]
    review_count:   int = 0
    aspect_profile: dict = field(default_factory=dict)

    @property
    def norm_score(self) -> float:
        return float(np.clip((self.absa_score + 1) / 2, 0.0, 1.0))

    @property
    def star_rating(self) -> float:
        return float(np.clip((self.absa_score + 1) / 2 * 5, 0.0, 5.0))


def _priority_weights(priority_order, lam=2.0):
    k      = len(priority_order)
    ranks  = np.array([k - i for i in range(k)], dtype=float)
    w      = _softmax(lam * ranks)
    return {cat: float(w[i]) for i, cat in enumerate(priority_order)}


def _auto_split(total_budget, priority_order, lam=2.0):
    w = _priority_weights(priority_order, lam)
    return {cat: total_budget * wv for cat, wv in w.items()}


def _best_affordable(vlist, budget):
    aff = [v for v in vlist if v.price <= budget]
    return sorted(aff, key=lambda v: v.norm_score, reverse=True)


def _greedy(cat_budgets, vby_cat):
    sel = {}
    for cat, alloc in cat_budgets.items():
        cands      = vby_cat.get(cat, [])
        aff        = _best_affordable(cands, alloc)
        sel[cat]   = aff[0] if aff else None
    return sel


def _ilp(total_budget, priority_order, vby_cat, lam=2.0, cat_caps=None):
    if not _ILP_AVAILABLE:
        return None, None
    w    = _priority_weights(priority_order, lam)
    cats = [c for c in priority_order if c in vby_cat]
    prob = _pulp.LpProblem("VendorBudget", _pulp.LpMaximize)
    x    = {(cat, v.name): _pulp.LpVariable(
                f"x_{cat}_{v.name}".replace(" ", "_"), cat=_pulp.LpBinary)
            for cat in cats for v in vby_cat[cat]}
    prob += _pulp.lpSum(w[cat] * v.norm_score * x[(cat, v.name)]
                        for cat in cats for v in vby_cat[cat])
    for cat in cats:
        prob += _pulp.lpSum(x[(cat, v.name)] for v in vby_cat[cat]) == 1
    prob += _pulp.lpSum(v.price * x[(cat, v.name)]
                        for cat in cats for v in vby_cat[cat]) <= total_budget
    if cat_caps:
        for cat in cats:
            if cat in cat_caps:
                prob += _pulp.lpSum(v.price * x[(cat, v.name)]
                                    for v in vby_cat[cat]) <= cat_caps[cat]
    _pulp.PULP_CBC_CMD(msg=False).solve(prob)
    sel   = {}
    cost  = 0.0
    for cat in cats:
        chosen = None
        for v in vby_cat[cat]:
            val = _pulp.value(x[(cat, v.name)])
            if val is not None and round(val) == 1:
                chosen = v
                cost  += v.price
                break
        sel[cat] = chosen
    return sel, cost


def _exhaustive(total_budget, priority_order, vby_cat, lam=2.0, cat_caps=None):
    w        = _priority_weights(priority_order, lam)
    cats     = [c for c in priority_order if c in vby_cat]
    best_obj = -1.0
    best     = {cat: None for cat in cats}
    best_c   = 0.0
    for combo in itertools.product(*[vby_cat.get(c, []) for c in cats]):
        total = sum(v.price for v in combo)
        if total > total_budget:
            continue
        if cat_caps and any(combo[i].price > cat_caps.get(cats[i], 1e18)
                            for i in range(len(cats))):
            continue
        obj = sum(w[cats[i]] * combo[i].norm_score for i in range(len(cats)))
        if obj > best_obj:
            best_obj = obj
            best     = {cats[i]: combo[i] for i in range(len(cats))}
            best_c   = total
    return best, best_c


def _recommend_auto(total_budget, priority_order, vby_cat, lam=2.0):
    w       = _priority_weights(priority_order, lam)
    splits  = _auto_split(total_budget, priority_order, lam)
    greedy  = _greedy(splits, vby_cat)
    missing = [c for c, v in greedy.items() if v is None]
    if not missing:
        sel   = greedy
        cost  = sum(v.price for v in sel.values() if v is not None)
        solver = "greedy"
    else:
        if _ILP_AVAILABLE:
            sel, cost = _ilp(total_budget, priority_order, vby_cat, lam)
            solver = "ILP"
        else:
            sel, cost = _exhaustive(total_budget, priority_order, vby_cat, lam)
            solver = "exhaustive"
        if sel is None:
            sel   = greedy
            cost  = sum(v.price for v in greedy.values() if v is not None)
            solver = "greedy (fallback)"
    obj = sum(w.get(cat, 0) * v.norm_score
              for cat, v in sel.items() if v is not None)
    return {"selection": sel, "total_cost": cost,
            "remaining": total_budget - cost,
            "splits": splits, "weights": w,
            "obj": round(obj, 4), "solver": solver}


def _recommend_manual(total_budget, cat_pct, vby_cat, reallocate=True):
    total    = sum(cat_pct.values()) or 1
    norm_pct = {cat: pct / total * 100 for cat, pct in cat_pct.items()}
    budgets  = {cat: total_budget * (pct / 100) for cat, pct in norm_pct.items()}
    sel      = _greedy(budgets, vby_cat)
    cost     = sum(v.price for v in sel.values() if v is not None)
    upgrades = []
    if reallocate:
        savings = total_budget - cost
        changed = True
        while changed and savings > 0:
            changed = False
            for cat, cur in list(sel.items()):
                cands = vby_cat.get(cat, [])
                cur_p = cur.price if cur else 0.0
                ups   = [v for v in cands
                         if v.norm_score > (cur.norm_score if cur else -1)
                         and v.price <= cur_p + savings]
                if ups:
                    best   = max(ups, key=lambda v: v.norm_score)
                    delta  = best.price - cur_p
                    savings -= delta
                    cost    += delta
                    upgrades.append(
                        f"{cat}: {cur.name if cur else 'None'} "
                        f"-> {best.name} "
                        f"(+{delta:,.0f}, "
                        f"score {(cur.absa_score if cur else 0):+.3f} "
                        f"-> {best.absa_score:+.3f})"
                    )
                    sel[cat] = best
                    changed  = True
                    break
    scored = [v for v in sel.values() if v is not None]
    obj    = float(np.mean([v.norm_score for v in scored])) if scored else 0.0
    return {"selection": sel, "total_cost": cost,
            "remaining": total_budget - cost,
            "budgets": budgets, "pct": norm_pct,
            "obj": round(obj, 4), "upgrades": upgrades}


# ---------------------------------------------------------------------------
# Built-in sample data (only used when user activates the sample toggle)
# ---------------------------------------------------------------------------

_SAMPLE_VENDORS = [
    ("Grand Ballroom",        "venue",         120_000,  0.78, 42),
    ("The Garden Estate",     "venue",          85_000,  0.61, 35),
    ("City View Hall",        "venue",          60_000,  0.44, 28),
    ("Heritage House",        "venue",          45_000,  0.30, 19),
    ("Budget Banquet Hall",   "venue",          22_000,  0.05, 11),
    ("Elite Cuisine Co.",     "catering",       95_000,  0.82, 58),
    ("Spice Garden Caterers", "catering",       60_000,  0.55, 44),
    ("Homestyle Feast",       "catering",       40_000,  0.41, 31),
    ("QuickBite Catering",    "catering",       25_000,  0.08, 17),
    ("Budget Bites",          "catering",       15_000, -0.20,  9),
    ("Pixel Perfect Studios", "photography",    55_000,  0.88, 63),
    ("Capture Moments",       "photography",    38_000,  0.67, 47),
    ("Frame & Flash",         "photography",    25_000,  0.45, 29),
    ("Snapshot Pro",          "photography",    15_000,  0.22, 18),
    ("Economy Clicks",        "photography",     8_000, -0.05,  7),
    ("Floral Fantasy",        "decoration",     45_000,  0.75, 38),
    ("Bloom & Decor",         "decoration",     30_000,  0.58, 26),
    ("Simple Touches",        "decoration",     18_000,  0.33, 14),
    ("DIY Decor Supplies",    "decoration",      8_000, -0.10,  6),
    ("Harmony Live Band",     "entertainment",  50_000,  0.80, 41),
    ("DJ Maestro",            "entertainment",  25_000,  0.60, 33),
    ("Acoustic Duo",          "entertainment",  15_000,  0.38, 22),
    ("Karaoke Setup Rental",  "entertainment",   8_000,  0.10,  8),
]


def _build_sample_vendors():
    vby = defaultdict(list)
    for (name, cat, price, score, rc) in _SAMPLE_VENDORS:
        vby[cat].append(_Vendor(name=name, category=cat, price=price,
                                absa_score=score, review_count=rc))
    return dict(vby)


# ---------------------------------------------------------------------------
# Budget Planner chart helpers
# ---------------------------------------------------------------------------

def _fig_budget_alloc(result, mode_label):
    """Pie + allocated-vs-spend bar chart."""
    sel     = result["selection"]
    cats    = [c for c, v in sel.items() if v is not None]
    if not cats:
        return None
    budgets = ([result["splits"][c] for c in cats] if "splits" in result
               else [result["budgets"][c] for c in cats])
    prices  = [sel[c].price for c in cats]
    names   = [sel[c].name  for c in cats]
    scores  = [sel[c].absa_score for c in cats]
    cmap    = cm.get_cmap("tab10", len(cats))
    colors  = [cmap(i) for i in range(len(cats))]

    fig, axes = plt.subplots(1, 2, figsize=(13, max(3.5, len(cats) * 0.6)))

    # Pie
    wedges, _, atxts = axes[0].pie(
        budgets, labels=None, autopct="%1.1f%%", colors=colors,
        startangle=90, wedgeprops={"linewidth": 0.8, "edgecolor": "#0d0d0d"},
        pctdistance=0.78)
    for at in atxts:
        at.set_fontsize(8); at.set_color("#ffffff")
    axes[0].legend(wedges, cats, loc="lower center",
                   bbox_to_anchor=(0.5, -0.18), ncol=2,
                   fontsize=8, frameon=False, labelcolor="#ccc")
    axes[0].set_title("Budget Allocation", fontsize=10, pad=8)

    # Bar
    y = np.arange(len(cats))
    axes[1].barh(y, budgets[::-1], height=0.55,
                 color=[(*c[:3], 0.22) for c in colors[::-1]], label="Allocated")
    axes[1].barh(y, prices[::-1], height=0.35,
                 color=colors[::-1], label="Actual spend")
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(cats[::-1], fontsize=9)
    axes[1].set_xlabel("Amount", fontsize=9)
    axes[1].set_title("Allocated vs Actual Spend", fontsize=10, pad=8)
    axes[1].spines[["top", "right"]].set_visible(False)
    axes[1].legend(fontsize=8, frameon=False, labelcolor="#aaa")
    mx = max(budgets) if budgets else 1
    for i, (cat, name, sc) in enumerate(
            zip(cats[::-1], names[::-1], scores[::-1])):
        axes[1].text(mx * 0.01, i, f"{name}  ({sc:+.2f})",
                     va="center", fontsize=7.5, color="#ddd")
    fig.suptitle(mode_label, fontsize=11, y=1.01, color="#ffffff")
    plt.tight_layout()
    return fig


def _fig_quality_price(vby_cat, selected=None):
    cats   = list(vby_cat.keys())
    cmap   = cm.get_cmap("tab10", len(cats))
    colors = {cat: cmap(i) for i, cat in enumerate(cats)}
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for cat, vlist in vby_cat.items():
        ax.scatter([v.price for v in vlist], [v.absa_score for v in vlist],
                   color=colors[cat], s=65, alpha=0.75, label=cat, zorder=2)
        for v in vlist:
            ax.annotate(v.name, (v.price, v.absa_score),
                        textcoords="offset points", xytext=(6, 2),
                        fontsize=6.5, color="#888", clip_on=True)
    if selected:
        for cat, v in selected.items():
            if v:
                ax.scatter([v.price], [v.absa_score], s=260,
                           facecolors="none", edgecolors="white",
                           linewidths=2.2, zorder=5)
    ax.axhline(0, color="#2a2a2a", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Price", fontsize=9)
    ax.set_ylabel("ABSA Score", fontsize=9)
    ax.set_title("Quality vs Price  (white ring = selected vendor)", fontsize=10, pad=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", fontsize=8, frameon=True,
              framealpha=0.12, edgecolor="#2a2a2a", labelcolor="#aaa")
    plt.tight_layout()
    return fig


def _result_table_html(result, mode):
    sel     = result["selection"]
    cats    = list(sel.keys())
    th = ("padding:0.4rem 0.75rem;text-align:left;color:#3a3a3a;font-size:0.6rem;"
          "text-transform:uppercase;letter-spacing:0.1em;border-bottom:1px solid #161616;"
          "font-family:IBM Plex Mono,monospace;")
    td = ("padding:0.38rem 0.75rem;border-bottom:1px solid #111;"
          "font-family:IBM Plex Mono,monospace;font-size:0.8rem;")
    heads = ["Category", "Allocated Budget", "Vendor", "Price", "ABSA Score", "Stars", "Budget Used"]
    hrow  = "".join(f'<th style="{th}">{h}</th>' for h in heads)
    rows  = ""
    for cat in cats:
        v     = sel[cat]
        if mode == "auto":
            alloc = result["splits"].get(cat, 0)
        else:
            alloc = result["budgets"].get(cat, 0)
        if v:
            used_pct = int(v.price / alloc * 100) if alloc else 0
            sc       = v.absa_score
            sc_col   = "#2ecc71" if sc >= 0 else "#e74c3c"
            bar_col  = "#2ecc71" if used_pct <= 100 else "#e74c3c"
            rows += f"""<tr>
              <td style="{td}color:#aaa;">{cat}</td>
              <td style="{td}color:#777;">{alloc:,.0f}</td>
              <td style="{td}color:#e8e8e8;font-weight:600;">{v.name}</td>
              <td style="{td}color:#ccc;">{v.price:,.0f}</td>
              <td style="{td}color:{sc_col};">{sc:+.3f}</td>
              <td style="{td}">{stars_display(v.star_rating, "0.9rem")}</td>
              <td style="{td}">
                <span style="color:{bar_col};">{used_pct}%</span>
                <span style="display:inline-block;width:48px;height:2px;background:#161616;
                             border-radius:2px;vertical-align:middle;margin-left:4px;overflow:hidden;">
                  <span style="display:block;width:{min(used_pct,100)}%;height:100%;
                               background:{bar_col};"></span>
                </span>
              </td>
            </tr>"""
        else:
            rows += f"""<tr>
              <td style="{td}color:#aaa;">{cat}</td>
              <td style="{td}color:#777;">{alloc:,.0f}</td>
              <td colspan="5" style="{td}color:#e74c3c;">No affordable vendor found</td>
            </tr>"""
    return (f'<div class="leaderboard-wrap"><table class="lb-table">'
            f'<thead><tr style="background:#090909;">{hrow}</tr></thead>'
            f'<tbody>{rows}</tbody></table></div>')


# ---------------------------------------------------------------------------
# Tab 6: Budget Planner
# ---------------------------------------------------------------------------

with tab6:
    sec("Event Budget Planner")
    info(
        "Upload your vendor review files to score them with ABSA, then set your total budget. "
        "The planner finds the best combination of vendors across your chosen service categories. "
        "Choose Auto mode to let the system allocate budget by priority, or Manual mode to "
        "set your own percentage split per category."
    )

    if not models_ready():
        warn("Load models from the sidebar before running ABSA scoring on uploaded files.")

    # ------------------------------------------------------------------
    # Step 1: Vendor data source
    # ------------------------------------------------------------------
    sec("Step 1 - Vendor Data")

    use_sample = st.toggle(
        "Use pre-loaded example vendors",
        value=False,
        key="bp_use_sample",
        help="Loads a built-in set of 23 vendors across 5 categories so you can "
             "explore the planner without uploading files.",
    )

    bp_vendors_by_cat = {}   # {category: [_Vendor, ...]}

    if use_sample:
        bp_vendors_by_cat = _build_sample_vendors()
        ok(f"Pre-loaded {sum(len(v) for v in bp_vendors_by_cat.values())} vendors "
           f"across {len(bp_vendors_by_cat)} categories.")
    else:
        info(
            "Upload one file per vendor. Each file must be a CSV with a <code>text</code> column "
            "or a plain TXT file with one review per line. "
            "The filename is used as the vendor name. "
            "After uploading, enter the price for each vendor and assign it to a category."
        )

        bp_files = st.file_uploader(
            "Upload vendor review files",
            type=["csv", "txt"],
            accept_multiple_files=True,
            key="bp_upload",
        )

        if bp_files and models_ready():
            # Score each uploaded vendor once and cache the result
            cached = st.session_state.get("bp_scored_cache", {})
            ate_mdl, ate_tok, sent_mdl, sent_tok = get_models()

            for f in bp_files:
                vname = os.path.splitext(f.name)[0]
                if vname in cached:
                    continue
                revs = load_reviews_from_file(f)
                if not revs:
                    continue
                flat = []
                prog_tmp = st.progress(0, text=f"Scoring {vname}...")
                for idx, rev in enumerate(revs):
                    flat.extend(_run(rev, ate_mdl, ate_tok, sent_mdl, sent_tok))
                    prog_tmp.progress((idx + 1) / len(revs), text=f"Scoring {vname}...")
                prog_tmp.empty()
                prof = compute_vendor_profile(flat, len(revs), use_norm=use_norm)
                if prof:
                    cached[vname] = {
                        "raw_score":      prof["raw_score"],
                        "review_count":   prof["review_count"],
                        "aspect_profile": prof["aspect_profile"],
                    }
            st.session_state["bp_scored_cache"] = cached

            if cached:
                ok(f"Scored {len(cached)} vendor(s). Assign a category and price to each.")

                # Build the config table: category + price inputs per vendor
                st.markdown(
                    '<div style="font-family:IBM Plex Mono,monospace;font-size:0.68rem;'
                    'text-transform:uppercase;letter-spacing:0.1em;color:#3a3a3a;'
                    'margin-bottom:0.6rem;">Vendor Configuration</div>',
                    unsafe_allow_html=True,
                )

                # Collect all categories already assigned so we can suggest them
                existing_cats = sorted({
                    st.session_state.get(f"bp_cat_{vn}", "")
                    for vn in cached
                    if st.session_state.get(f"bp_cat_{vn}", "")
                })

                for vn, prof_data in cached.items():
                    c1, c2, c3, c4 = st.columns([2.5, 2, 1.8, 1.2])
                    with c1:
                        st.markdown(
                            f'<div style="font-family:IBM Plex Mono,monospace;font-size:0.82rem;'
                            f'color:#e8e8e8;padding-top:0.55rem;">{vn}</div>',
                            unsafe_allow_html=True,
                        )
                    with c2:
                        st.text_input(
                            "Category",
                            key=f"bp_cat_{vn}",
                            placeholder="e.g. venue",
                            label_visibility="collapsed",
                        )
                    with c3:
                        st.number_input(
                            "Price",
                            min_value=0,
                            value=0,
                            step=1000,
                            key=f"bp_price_{vn}",
                            label_visibility="collapsed",
                        )
                    with c4:
                        score = prof_data["raw_score"]
                        sc    = "#2ecc71" if score >= 0 else "#e74c3c"
                        st.markdown(
                            f'<div style="font-family:IBM Plex Mono,monospace;font-size:0.82rem;'
                            f'color:{sc};padding-top:0.55rem;">ABSA {score:+.3f}</div>',
                            unsafe_allow_html=True,
                        )

                # Build bp_vendors_by_cat from the config inputs
                for vn, prof_data in cached.items():
                    cat   = st.session_state.get(f"bp_cat_{vn}", "").strip().lower()
                    price = float(st.session_state.get(f"bp_price_{vn}", 0) or 0)
                    if cat and price > 0:
                        bp_vendors_by_cat.setdefault(cat, []).append(
                            _Vendor(
                                name           = vn,
                                category       = cat,
                                price          = price,
                                absa_score     = prof_data["raw_score"],
                                review_count   = prof_data["review_count"],
                                aspect_profile = prof_data["aspect_profile"],
                            )
                        )
        elif bp_files and not models_ready():
            warn("Models must be loaded before vendor files can be scored.")

    # ------------------------------------------------------------------
    # Step 2: Budget and mode configuration
    # ------------------------------------------------------------------
    if bp_vendors_by_cat:
        all_cats = sorted(bp_vendors_by_cat.keys())

        sec("Step 2 - Budget and Mode")

        col_budget, col_mode = st.columns([1, 1])
        with col_budget:
            total_budget_bp = st.number_input(
                "Total budget",
                min_value=1_000,
                max_value=100_000_000,
                value=250_000,
                step=5_000,
                key="bp_total_budget",
            )
        with col_mode:
            mode_bp = st.radio(
                "Planning mode",
                ["Auto - priority-based split", "Manual - set my own percentages"],
                key="bp_mode",
                help=(
                    "Auto mode allocates budget by priority order. "
                    "Manual mode lets you set exact percentage splits per category."
                ),
            )

        # ------------------------------------------------------------------
        # Auto mode config
        # ------------------------------------------------------------------
        if mode_bp == "Auto - priority-based split":
            sec("Step 3 - Priority Order")
            info(
                "Drag the categories into your preferred order using the multiselect below. "
                "Categories listed first receive a larger share of the budget. "
                "You must include every category that has vendors."
            )

            priority_bp = st.multiselect(
                "Category priority (most important first)",
                options=all_cats,
                default=all_cats,
                key="bp_priority",
            )

            lam_bp = st.slider(
                "Priority skew strength",
                min_value=0.5, max_value=4.0, value=2.0, step=0.25,
                key="bp_lambda",
                help=(
                    "Controls how strongly the priority order skews the budget. "
                    "0.5 = nearly equal split. 4.0 = top category gets much more."
                ),
            )

            # Preview the allocation
            if priority_bp:
                splits_preview = _auto_split(total_budget_bp, priority_bp, lam_bp)
                weights_preview = _priority_weights(priority_bp, lam_bp)
                preview_rows = [
                    {
                        "category":  cat,
                        "priority_weight": f"{weights_preview[cat]:.3f}",
                        "allocated": f"{splits_preview[cat]:,.0f}",
                        "vendors_available": len(bp_vendors_by_cat.get(cat, [])),
                    }
                    for cat in priority_bp
                ]
                st.dataframe(
                    pd.DataFrame(preview_rows),
                    use_container_width=True,
                    hide_index=True,
                )

            run_auto = st.button("Find Best Combination", type="primary", key="bp_run_auto")
            if run_auto:
                if not priority_bp:
                    err("Select at least one category.")
                else:
                    missing_cats = [c for c in priority_bp if c not in bp_vendors_by_cat]
                    if missing_cats:
                        err(f"No vendors found for: {', '.join(missing_cats)}")
                    else:
                        with st.spinner("Optimising..."):
                            res_auto = _recommend_auto(
                                total_budget_bp, priority_bp, bp_vendors_by_cat, lam_bp
                            )
                        st.session_state["bp_result"] = ("auto", res_auto, bp_vendors_by_cat)

        # ------------------------------------------------------------------
        # Manual mode config
        # ------------------------------------------------------------------
        else:
            sec("Step 3 - Percentage Split")
            info(
                "Set the percentage of your budget to allocate to each category. "
                "Values do not need to sum to exactly 100; they are normalised automatically. "
                "Any budget saved within a category is pooled and used to upgrade other categories."
            )

            pct_inputs = {}
            n_cats     = len(all_cats)
            default_pct = round(100 / n_cats, 1) if n_cats else 20.0

            cols_pct = st.columns(min(n_cats, 5))
            for i, cat in enumerate(all_cats):
                with cols_pct[i % len(cols_pct)]:
                    pct_inputs[cat] = st.number_input(
                        cat,
                        min_value=0.0,
                        max_value=100.0,
                        value=default_pct,
                        step=1.0,
                        key=f"bp_pct_{cat}",
                    )

            total_entered = sum(pct_inputs.values())
            if total_entered > 0:
                st.markdown(
                    f'<div style="font-family:IBM Plex Mono,monospace;font-size:0.75rem;'
                    f'color:#555;margin-bottom:0.6rem;">'
                    f'Entered total: {total_entered:.1f}%  '
                    f'(will be normalised to 100%)</div>',
                    unsafe_allow_html=True,
                )

            reallocate_bp = st.checkbox(
                "Reallocate savings to upgrade other categories",
                value=True,
                key="bp_reallocate",
                help="After initial selection, any money saved within a category's "
                     "budget is pooled and used to upgrade another category to the next tier.",
            )

            run_manual = st.button("Find Best Combination", type="primary", key="bp_run_manual")
            if run_manual:
                active_pct = {cat: pct for cat, pct in pct_inputs.items() if pct > 0}
                if not active_pct:
                    err("Set a non-zero percentage for at least one category.")
                else:
                    with st.spinner("Optimising..."):
                        res_manual = _recommend_manual(
                            total_budget_bp, active_pct, bp_vendors_by_cat, reallocate_bp
                        )
                    st.session_state["bp_result"] = ("manual", res_manual, bp_vendors_by_cat)

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    if "bp_result" in st.session_state:
        mode_used, result_bp, vby_cat_bp = st.session_state["bp_result"]
        sel_bp = result_bp["selection"]

        sec("Recommended Combination")

        # Summary stat cards
        selected_vendors = [v for v in sel_bp.values() if v is not None]
        missing_cats     = [cat for cat, v in sel_bp.items() if v is None]
        avg_score        = float(np.mean([v.absa_score for v in selected_vendors])) \
                           if selected_vendors else 0.0
        avg_star         = float(np.clip((avg_score + 1) / 2 * 5, 0, 5))

        stat_row(
            ("Total cost",        f"{result_bp['total_cost']:,.0f}"),
            ("Remaining budget",  f"{result_bp['remaining']:,.0f}"),
            ("Vendors selected",  len(selected_vendors)),
            ("Avg ABSA score",    f"{avg_score:+.3f}"),
            ("Avg stars",         f"{avg_star:.2f} / 5.00"),
            ("Quality index",     f"{result_bp['obj']:.4f}",
             "#2ecc71" if result_bp["obj"] >= 0.5 else "#f39c12"),
        )

        if missing_cats:
            warn(
                f"No affordable vendor was found for: {', '.join(missing_cats)}. "
                "Consider increasing your budget or adding lower-cost vendors."
            )

        # Result table
        st.markdown(
            _result_table_html(result_bp, mode_used),
            unsafe_allow_html=True,
        )

        # Upgrades applied (manual mode)
        if mode_used == "manual" and result_bp.get("upgrades"):
            sec("Upgrades Applied via Reallocation")
            for u in result_bp["upgrades"]:
                ok(u)

        if mode_used == "auto" and "solver" in result_bp:
            st.markdown(
                f'<div style="font-family:IBM Plex Mono,monospace;font-size:0.65rem;'
                f'color:#2a2a2a;margin-top:0.4rem;">solver: {result_bp["solver"]}</div>',
                unsafe_allow_html=True,
            )

        sec("Budget Allocation Chart")
        mode_label = ("Auto - Priority Split" if mode_used == "auto"
                      else "Manual - Custom Split")
        fig_alloc = _fig_budget_alloc(result_bp, mode_label)
        if fig_alloc:
            st.pyplot(fig_alloc)

        sec("Quality vs Price")
        st.pyplot(_fig_quality_price(vby_cat_bp, selected=sel_bp))

        sec("Download Results")
        rows_dl = []
        for cat, v in sel_bp.items():
            if mode_used == "auto":
                alloc = result_bp["splits"].get(cat, 0)
            else:
                alloc = result_bp["budgets"].get(cat, 0)
            rows_dl.append({
                "category":        cat,
                "allocated_budget": round(alloc, 2),
                "vendor":          v.name if v else None,
                "price":           v.price if v else None,
                "absa_score":      round(v.absa_score, 4) if v else None,
                "star_rating":     round(v.star_rating, 2) if v else None,
                "review_count":    v.review_count if v else None,
            })
        st.download_button(
            "Download Recommendation CSV",
            pd.DataFrame(rows_dl).to_csv(index=False).encode(),
            "budget_recommendation.csv",
            "text/csv",
        )


# =============================================================================
# Tab 7: How It Works
# =============================================================================

with tab7:
    absa_how_it_works.render(sec)