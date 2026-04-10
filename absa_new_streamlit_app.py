"""
ABSA Streamlit Application - Multi-Model Edition (v2)
Supports: BERT, DeBERTa-v3, RoBERTa, ELECTRA, and Ensemble pipeline.
Elo-based vendor ranking replaces the simple relative star scaling.

Run:
    streamlit run absa_new_streamlit.py
"""

import io
import os
import warnings
from collections import defaultdict, Counter

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
)

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

MODEL_REGISTRY = {
    "BERT-base":       {"ate": "./bert_ate_model_final",     "sent": "./bert_sentiment_model_final",
                        "checkpoint": "bert-base-uncased"},
    "DeBERTa-v3-base": {"ate": "./deberta_ate_model_final",  "sent": "./deberta_sentiment_model_final",
                        "checkpoint": "microsoft/deberta-v3-base"},
    "RoBERTa-base":    {"ate": "./roberta_ate_model_final",  "sent": "./roberta_sentiment_model_final",
                        "checkpoint": "roberta-base"},
    "ELECTRA-base":    {"ate": "./electra_ate_model_final",  "sent": "./electra_sentiment_model_final",
                        "checkpoint": "google/electra-base-discriminator"},
    "BERT (legacy)":   {"ate": "./ate_model_final",          "sent": "./sentiment_model_final",
                        "checkpoint": "bert-base-uncased"},
}

ASPECT_NORM_MAP = {
    # Food and product quality
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

    # Photography-specific keywords (expanded)
    "candid": "quality", "portrait": "quality", "portraits": "quality",
    "shoot": "quality", "photoshoot": "quality", "session": "quality",
    "album": "quality", "albums": "quality", "photobook": "quality",
    "lens": "quality", "lenses": "quality", "framing": "quality",
    "angles": "quality", "angle": "quality", "composition": "quality",
    "exposure": "quality", "shutter": "quality", "focus": "quality",
    "bokeh": "quality", "depth": "quality", "colour": "quality",
    "contrast": "quality", "saturation": "quality", "tones": "quality",
    "retouching": "quality", "retouch": "quality", "post-processing": "quality",
    "post_processing": "quality", "filters": "quality", "preset": "quality",
    "presets": "quality", "raw": "quality", "resolution": "quality",
    "prints": "quality", "print": "quality", "digital": "quality",
    "slideshow": "quality", "gallery": "quality", "galleries": "quality",
    "drone": "quality", "aerial": "quality", "timelapse": "quality",
    "coverage": "quality", "shot": "quality", "shots": "quality",
    "snapshot": "quality", "snapshots": "quality", "captures": "quality",
    "capture": "quality", "cinematography": "quality", "cinematic": "quality",
    "reel": "quality", "highlight": "quality", "highlights": "quality",

    # Service and personnel
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
    "direction": "service", "guidance": "service", "posing": "service",
    "directions": "service", "instruction": "service", "instructions": "service",

    # Value and pricing
    "price": "value", "prices": "value", "cost": "value", "value": "value",
    "pricing": "value", "fee": "value", "fees": "value", "charge": "value",
    "charges": "value", "rate": "value", "rates": "value", "quote": "value",
    "package": "value", "deal": "value", "expensive": "value",
    "cheap": "value", "affordable": "value", "budget": "value",
    "packages": "value", "pricing_packages": "value", "contract": "value",

    # Reliability and logistics
    "reliability": "reliability", "punctuality": "reliability", "timing": "reliability",
    "delivery": "reliability", "wait": "reliability", "waiting": "reliability",
    "delay": "reliability", "delays": "reliability", "time": "reliability",
    "speed": "reliability", "turnaround": "reliability", "schedule": "reliability",
    "deadline": "reliability", "promptness": "reliability",
    "availability": "reliability", "booking": "reliability",

    # Ambiance and venue
    "ambiance": "ambiance", "ambience": "ambiance", "atmosphere": "ambiance",
    "venue": "ambiance", "location": "ambiance", "place": "ambiance",
    "space": "ambiance", "cleanliness": "ambiance", "noise": "ambiance",
    "environment": "ambiance", "setting": "ambiance", "hall": "ambiance",
    "room": "ambiance", "area": "ambiance", "facility": "ambiance",
    "parking": "ambiance", "access": "ambiance", "accessibility": "ambiance",

    # Overall experience
    "experience": "experience", "overall": "experience", "visit": "experience",
    "recommendation": "experience", "event": "experience", "occasion": "experience",
    "celebration": "experience", "party": "experience", "wedding": "experience",
    "function": "experience", "memories": "experience", "memory": "experience",
    "moment": "experience", "moments": "experience",
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
.absa-header {
    border-bottom: 1px solid #1e1e1e;
    padding: 2.2rem 0 1.6rem 0;
    margin-bottom: 2rem;
    display: flex;
    align-items: flex-end;
    gap: 2rem;
}
.absa-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2.4rem;
    font-weight: 600;
    letter-spacing: -0.03em;
    color: #ffffff;
    margin: 0;
    line-height: 1;
}
.absa-subtitle {
    font-size: 0.78rem;
    color: #3a3a3a;
    font-weight: 300;
    margin-top: 0.45rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}
.absa-pill {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.62rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #2ecc71;
    border: 1px solid #1a3a28;
    background: #051209;
    border-radius: 3px;
    padding: 0.2rem 0.55rem;
    margin-bottom: 0.25rem;
    display: inline-block;
}
.stat-row { display: flex; gap: 0.65rem; margin-bottom: 1.4rem; flex-wrap: wrap; }
.stat-card {
    flex: 1;
    min-width: 105px;
    background: #111;
    border: 1px solid #1e1e1e;
    border-radius: 5px;
    padding: 0.95rem 1.15rem;
    transition: border-color 0.2s;
}
.stat-card:hover { border-color: #2a2a2a; }
.stat-label {
    font-size: 0.62rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #333;
    margin-bottom: 0.3rem;
    font-family: 'IBM Plex Mono', monospace;
}
.stat-value {
    font-size: 1.55rem;
    font-weight: 600;
    color: #ffffff;
    font-family: 'IBM Plex Mono', monospace;
    line-height: 1;
}
.sec {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.67rem;
    text-transform: uppercase;
    letter-spacing: 0.13em;
    color: #333;
    border-bottom: 1px solid #161616;
    padding-bottom: 0.35rem;
    margin: 1.8rem 0 0.9rem 0;
}
.info-box {
    background: #0d1620;
    border-left: 3px solid #3498db;
    padding: 0.75rem 1rem;
    font-size: 0.84rem;
    color: #666;
    margin-bottom: 0.9rem;
    border-radius: 0 4px 4px 0;
}
.warn-box {
    background: #160f00;
    border-left: 3px solid #f39c12;
    padding: 0.75rem 1rem;
    font-size: 0.84rem;
    color: #666;
    margin-bottom: 0.9rem;
    border-radius: 0 4px 4px 0;
}
.ok-box {
    background: #051209;
    border-left: 3px solid #2ecc71;
    padding: 0.75rem 1rem;
    font-size: 0.84rem;
    color: #666;
    margin-bottom: 0.9rem;
    border-radius: 0 4px 4px 0;
}
.err-box {
    background: #160505;
    border-left: 3px solid #e74c3c;
    padding: 0.75rem 1rem;
    font-size: 0.84rem;
    color: #666;
    margin-bottom: 0.9rem;
    border-radius: 0 4px 4px 0;
}
.asp-grid { display: flex; flex-wrap: wrap; gap: 0.65rem; margin-bottom: 1.2rem; }
.asp-card {
    background: #111;
    border: 1px solid #1e1e1e;
    border-radius: 5px;
    padding: 0.9rem 1.15rem;
    min-width: 155px;
    flex: 1;
    max-width: 215px;
    transition: border-color 0.15s;
}
.asp-card:hover { border-color: #2a2a2a; }
.asp-name {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #444;
    margin-bottom: 0.28rem;
}
.asp-sent {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.2rem;
    font-weight: 600;
    line-height: 1;
}
.asp-meta {
    font-size: 0.67rem;
    color: #2a2a2a;
    margin-top: 0.22rem;
    font-family: 'IBM Plex Mono', monospace;
}
.asp-bar { background: #161616; height: 3px; border-radius: 2px; margin-top: 0.5rem; overflow: hidden; }
.asp-fill { height: 100%; border-radius: 2px; }
.no-asp {
    background: #0f0f0f;
    border: 1px dashed #222;
    border-radius: 5px;
    padding: 1.4rem;
    text-align: center;
    color: #333;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    margin: 0.8rem 0;
}
.leaderboard-wrap { overflow-x: auto; margin-bottom: 1.5rem; }
.lb-table { width: 100%; border-collapse: collapse; background: #0f0f0f; }
.lb-th {
    padding: 0.5rem 0.9rem;
    text-align: left;
    color: #333;
    font-size: 0.6rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    border-bottom: 1px solid #161616;
    font-family: 'IBM Plex Mono', monospace;
}
.lb-td {
    padding: 0.45rem 0.9rem;
    border-bottom: 1px solid #131313;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
}
[data-testid="stTabs"] button {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.07em;
    text-transform: uppercase;
}
.stButton > button {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    border-radius: 4px;
}
.stTextArea textarea, .stTextInput input {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.86rem;
    background-color: #0f0f0f !important;
    color: #e0e0e0 !important;
    border: 1px solid #1e1e1e !important;
}
.chart-wrap {
    background: #0f0f0f;
    border: 1px solid #1a1a1a;
    border-radius: 5px;
    padding: 0.5rem;
    margin-bottom: 0.8rem;
}
#MainMenu, footer { visibility: hidden; }
</style>""", unsafe_allow_html=True)

plt.rcParams.update({
    "figure.facecolor":  "#111111",
    "axes.facecolor":    "#111111",
    "axes.edgecolor":    "#1e1e1e",
    "axes.labelcolor":   "#666",
    "xtick.color":       "#444",
    "ytick.color":       "#444",
    "text.color":        "#bbbbbb",
    "grid.color":        "#161616",
    "grid.linestyle":    "--",
    "font.family":       "monospace",
    "figure.dpi":        110,
})

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown("""
<div class="absa-header">
    <div>
        <div class="absa-title">ABSA</div>
        <div class="absa-subtitle">Aspect-Based Sentiment Analysis &nbsp;/&nbsp; Multi-Model Pipeline</div>
    </div>
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
        inner += (
            f'<div class="stat-card">'
            f'<div class="stat-label">{label}</div>'
            f'<div class="stat-value" style="color:{color};">{value}</div>'
            f'</div>'
        )
    st.markdown(f'<div class="stat-row">{inner}</div>', unsafe_allow_html=True)


def aspect_cards(results):
    if not results:
        st.markdown(
            '<div class="no-asp">No aspect terms detected. See How It Works tab.</div>',
            unsafe_allow_html=True,
        )
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
        cards += (
            f'<div class="asp-card">'
            f'<div class="asp-name">{r["aspect"]}</div>'
            f'<div class="asp-sent" style="color:{sc};">{sent}</div>'
            f'<div class="asp-meta">{score:+.3f} | conf {conf:.2f}</div>'
            f'<div class="asp-bar"><div class="asp-fill" '
            f'style="width:{bp}%;background:{bc};"></div></div>'
            f'</div>'
        )
    st.markdown(f'<div class="asp-grid">{cards}</div>', unsafe_allow_html=True)


def stars_display(rating, size="1.2rem"):
    filled = max(0, min(5, int(round(rating))))
    empty  = 5 - filled
    s  = f'<span style="color:#f39c12;font-size:{size};">' + "&#9733;" * filled + "</span>"
    s += f'<span style="color:#1e1e1e;font-size:{size};">' + "&#9733;" * empty + "</span>"
    return s


def rating_badge(vendor_name, star_rating, review_count):
    bar_pct = int(star_rating / 5 * 100)
    bar_col = "#2ecc71" if star_rating >= 3.5 else ("#f39c12" if star_rating >= 2.5 else "#e74c3c")
    st.markdown(f"""
    <div style="background:#111;border:1px solid #1e1e1e;border-radius:6px;
                padding:1.4rem 1.9rem;display:inline-block;min-width:210px;margin-bottom:1rem;">
        <div style="font-family:IBM Plex Mono,monospace;font-size:0.62rem;text-transform:uppercase;
                    letter-spacing:0.1em;color:#333;margin-bottom:0.35rem;">Star Rating</div>
        <div style="font-family:IBM Plex Mono,monospace;font-size:2.9rem;font-weight:600;
                    color:#ffffff;line-height:1;">{star_rating:.2f}</div>
        <div style="margin:0.4rem 0;">{stars_display(star_rating, "1.3rem")}</div>
        <div style="background:#161616;height:3px;border-radius:2px;overflow:hidden;margin:0.45rem 0;">
            <div style="width:{bar_pct}%;background:{bar_col};height:100%;border-radius:2px;"></div>
        </div>
        <div style="font-family:IBM Plex Mono,monospace;font-size:0.62rem;color:#333;margin-top:0.28rem;">
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
    mdl_ate.eval()
    mdl_sent.eval()
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
    return (
        st.session_state["active_ate_model"],
        st.session_state["active_ate_tok"],
        st.session_state["active_sent_model"],
        st.session_state["active_sent_tok"],
    )


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def extract_aspects(text, ate_model, ate_tok):
    words = text.split()
    if not words:
        return []
    enc = ate_tok(
        words, is_split_into_words=True, max_length=128,
        padding="max_length", truncation=True, return_tensors="pt",
    )
    with torch.no_grad():
        out   = ate_model(
            input_ids=enc["input_ids"].to(DEVICE),
            attention_mask=enc["attention_mask"].to(DEVICE),
        )
        preds = torch.argmax(out.logits.squeeze(0), dim=-1).cpu().numpy()
    word_preds = {}
    for ti, wi in enumerate(enc.word_ids()):
        if wi is not None and wi not in word_preds:
            word_preds[wi] = preds[ti]
    aspects, current = [], []
    for wi, word in enumerate(words):
        label = ATE_LABEL_NAMES[word_preds.get(wi, 0)]
        if label == "B-ASP":
            if current:
                aspects.append(" ".join(current))
            current = [word]
        elif label == "I-ASP" and current:
            current.append(word)
        else:
            if current:
                aspects.append(" ".join(current))
            current = []
    if current:
        aspects.append(" ".join(current))
    return aspects


def classify_sentiment(text, aspect, sent_model, sent_tok):
    enc = sent_tok(
        f"{aspect} [SEP] {text}", max_length=128,
        padding="max_length", truncation=True, return_tensors="pt",
    )
    with torch.no_grad():
        out   = sent_model(
            input_ids=enc["input_ids"].to(DEVICE),
            attention_mask=enc["attention_mask"].to(DEVICE),
        )
        probs = torch.softmax(out.logits, dim=1).squeeze().cpu().numpy()
    pred_idx = int(np.argmax(probs))
    return SENTIMENT_LABEL_NAMES[pred_idx], float(probs[pred_idx]), probs


def run_pipeline(text, ate_model, ate_tok, sent_model, sent_tok):
    aspects = extract_aspects(text, ate_model, ate_tok)
    results = []
    for asp in aspects:
        label, conf, probs = classify_sentiment(text, asp, sent_model, sent_tok)
        results.append({
            "aspect":         asp,
            "sentiment":      label,
            "confidence":     conf,
            "weighted_score": float(np.dot(probs, SCORE_VEC)),
            "probs":          probs,
        })
    return results


def run_ensemble_pipeline(text, ensemble_models, min_votes=2):
    if not ensemble_models:
        return []
    votes = Counter()
    for key, m in ensemble_models.items():
        for span in extract_aspects(text, m["mdl_ate"], m["tok_ate"]):
            votes[span.lower().strip()] += 1
    retained = [sp for sp, cnt in votes.items()
                if cnt >= min(min_votes, len(ensemble_models))]
    if not retained:
        return []
    results = []
    for aspect in retained:
        prob_stack = []
        for key, m in ensemble_models.items():
            _, _, probs = classify_sentiment(text, aspect, m["mdl_sent"], m["tok_sent"])
            prob_stack.append(probs)
        mp       = np.mean(prob_stack, axis=0)
        pred_idx = int(np.argmax(mp))
        results.append({
            "aspect":         aspect,
            "sentiment":      SENTIMENT_LABEL_NAMES[pred_idx],
            "confidence":     float(mp[pred_idx]),
            "weighted_score": float(np.dot(mp, SCORE_VEC)),
            "probs":          mp,
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
    if not asp_scores:
        return None
    total     = sum(len(v) for v in asp_scores.values())
    asp_means = {a: float(np.mean(v)) for a, v in asp_scores.items()}
    raw_score = sum(asp_means[a] * (len(asp_scores[a]) / total) for a in asp_means)
    profile   = {}
    for a in asp_means:
        sents = [
            r["sentiment"] for r in results_list
            if (normalise_aspect(r["aspect"]) if use_norm else r["aspect"].lower()) == a
        ]
        dominant = max(set(sents), key=sents.count) if sents else "neutral"
        profile[a] = {
            "score":    asp_means[a],
            "count":    len(asp_scores[a]),
            "weight":   len(asp_scores[a]) / total,
            "dominant": dominant,
        }
    return {
        "raw_score":      round(raw_score, 4),
        "review_count":   review_count,
        "aspect_profile": profile,
    }


class SentimentEloRanker:
    """Elo-based vendor ranking. Reference: Elo (1978)."""
    def __init__(self, k=32, initial=1500, bayes_m=10):
        self.k        = k
        self.initial  = initial
        self.bayes_m  = bayes_m
        self.ratings  = {}
        self.raw_scores = {}
        self.rev_counts = {}

    def add_vendor(self, name, profile_dict):
        if not profile_dict:
            return
        if name not in self.ratings:
            self.ratings[name] = self.initial
        self.rev_counts[name] = profile_dict["review_count"]
        raw = profile_dict["raw_score"]
        v   = profile_dict["review_count"]
        m   = self.bayes_m
        gm  = np.mean(list(self.raw_scores.values())) if self.raw_scores else 0.0
        self.raw_scores[name] = (v * raw + m * gm) / (v + m)

    def _update(self, rounds=5):
        vendors = list(self.ratings.keys())
        for _ in range(rounds):
            for i in range(len(vendors)):
                for j in range(len(vendors)):
                    if i == j:
                        continue
                    vi, vj = vendors[i], vendors[j]
                    ri, rj = self.ratings[vi], self.ratings[vj]
                    si = (self.raw_scores[vi] + 1) / 2
                    sj = (self.raw_scores[vj] + 1) / 2
                    actual   = si / (si + sj + 1e-9)
                    expected = 1.0 / (1.0 + 10 ** ((rj - ri) / 400))
                    self.ratings[vi] = ri + self.k * (actual - expected)

    def get_rankings(self):
        if not self.ratings:
            return pd.DataFrame()
        self._update()
        names   = list(self.ratings.keys())
        elo_arr = np.array([self.ratings[n] for n in names])
        raw_arr = np.array([self.raw_scores[n] for n in names])
        mn, mx  = elo_arr.min(), elo_arr.max()
        stars   = (
            np.full(len(names), 3.0) if mx == mn
            else 1.0 + (elo_arr - mn) / (mx - mn) * 4.0
        )
        se = np.abs(raw_arr) * 0.08 + 0.05
        df = pd.DataFrame({
            "vendor":      names,
            "elo_rating":  np.round(elo_arr, 1),
            "raw_score":   np.round(raw_arr, 4),
            "star_rating": np.round(np.clip(stars, 1, 5), 2),
            "ci_lo":       np.round(np.clip(stars - 1.96 * se, 1, 5), 2),
            "ci_hi":       np.round(np.clip(stars + 1.96 * se, 1, 5), 2),
            "reviews":     [self.rev_counts[n] for n in names],
        }).sort_values("elo_rating", ascending=False).reset_index(drop=True)
        df.insert(0, "rank", range(1, len(df) + 1))
        return df


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------

def fig_prob_bars(probs, title=""):
    fig, ax = plt.subplots(figsize=(4.5, 2.6))
    bars = ax.barh(SENTIMENT_LABEL_NAMES, probs, color=SENT_PALETTE, height=0.5)
    ax.set_xlim(0, 1.18)
    ax.set_xlabel("Probability", fontsize=8)
    if title:
        ax.set_title(title, fontsize=8.5, pad=5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)
    for bar, val in zip(bars, probs):
        ax.text(val + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", fontsize=7.5)
    plt.tight_layout()
    return fig


def fig_aspect_scores(profile, max_show=18):
    items  = sorted(profile.items(), key=lambda x: -x[1]["count"])[:max_show]
    labels = [a for a, _ in items]
    scores = [profile[a]["score"] for a in labels]
    colors = ["#2ecc71" if s >= 0 else "#e74c3c" for s in scores]
    h = max(3.0, len(labels) * 0.38)
    fig, ax = plt.subplots(figsize=(5.5, h))
    ax.barh(labels, scores, color=colors, height=0.55)
    ax.axvline(0, color="#2a2a2a", linewidth=0.8)
    ax.set_xlim(-1.18, 1.18)
    ax.set_xlabel("Sentiment Score", fontsize=8)
    ax.set_title("Aspect Scores", fontsize=9, pad=6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=7.5)
    ax.tick_params(axis="x", labelsize=7.5)
    for i, (label, val) in enumerate(zip(labels, scores)):
        xp = val + 0.03 if val >= 0 else val - 0.03
        ax.text(xp, i, f"{val:+.2f}", va="center",
                ha="left" if val >= 0 else "right", fontsize=7)
    plt.tight_layout()
    return fig


def fig_aspect_pie(profile, max_slices=9):
    items = sorted(profile.items(), key=lambda x: -x[1]["weight"])
    if len(items) > max_slices:
        top     = items[:max_slices]
        rest    = sum(v["weight"] for _, v in items[max_slices:])
        labels  = [a for a, _ in top] + ["other"]
        weights = [v["weight"] for _, v in top] + [rest]
    else:
        labels  = [a for a, _ in items]
        weights = [v["weight"] for _, v in items]
    cmap   = cm.get_cmap("tab20", len(labels))
    colors = [cmap(i) for i in range(len(labels))]
    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    wedges, _, autotexts = ax.pie(
        weights, labels=None, autopct="%1.0f%%", colors=colors,
        startangle=90,
        wedgeprops={"linewidth": 0.8, "edgecolor": "#0d0d0d"},
        pctdistance=0.76,
    )
    for at in autotexts:
        at.set_fontsize(7)
        at.set_color("#ddd")
    ax.legend(wedges, labels, loc="lower center", bbox_to_anchor=(0.5, -0.22),
              ncol=3, fontsize=7, frameon=False, labelcolor="#999")
    ax.set_title("Mention Frequency", fontsize=9, pad=6)
    plt.tight_layout()
    return fig


def fig_sentiment_distribution(results_list):
    """Stacked sentiment count by normalised aspect category."""
    if not results_list:
        return None
    counts = defaultdict(lambda: defaultdict(int))
    for r in results_list:
        cat = normalise_aspect(r["aspect"])
        counts[cat][r["sentiment"]] += 1
    cats  = sorted(counts.keys(), key=lambda c: -sum(counts[c].values()))[:14]
    if not cats:
        return None
    data  = {s: [counts[c].get(s, 0) for c in cats] for s in SENTIMENT_LABEL_NAMES}
    h     = max(3.5, len(cats) * 0.44)

    fig, ax = plt.subplots(figsize=(7, h))
    lefts = np.zeros(len(cats))

    for s, color in zip(SENTIMENT_LABEL_NAMES, SENT_PALETTE):
        vals = np.array(data[s])
        ax.barh(cats, vals, left=lefts, color=color, height=0.58, label=s)
        lefts += vals

    ax.set_xlabel("Count", fontsize=8)
    ax.set_title("Sentiment Distribution by Aspect Category", fontsize=9, pad=6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=7.5)
    ax.tick_params(axis="x", labelsize=7.5)

    # Move legend outside to avoid overlap
    ax.legend(loc="upper left",
              bbox_to_anchor=(1.02, 0),
              fontsize=7.5,
              frameon=False,
              labelcolor="#aaa",
              ncol=1,
              handlelength=1.2)

    # Add right padding so legend fits
    plt.tight_layout(rect=[0, 0, 0.82, 1])

    return fig

def fig_confidence_histogram(confs, threshold=0.6):
    """Histogram of prediction confidences with low-conf threshold marked."""
    if not confs:
        return None
    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    ax.hist(confs, bins=20, range=(0, 1), color="#3498db", alpha=0.80, edgecolor="#0d0d0d")
    ax.axvline(threshold, color="#f39c12", linewidth=1.2, linestyle="--",
               label=f"threshold {threshold:.2f}")
    ax.set_xlabel("Confidence", fontsize=8)
    ax.set_ylabel("Count", fontsize=8)
    ax.set_title("Prediction Confidence Distribution", fontsize=9, pad=6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=7.5, frameon=False, labelcolor="#aaa")
    plt.tight_layout()
    return fig


def fig_score_heatmap(vendor_profiles, max_cats=5):
    """Heatmap of top aspect scores across vendors."""
    
    if len(vendor_profiles) < 2:
        return None

    # Aggregate weights
    cat_weight = defaultdict(float)
    for prof in vendor_profiles.values():
        for cat, av in prof["aspect_profile"].items():
            cat_weight[cat] += av["weight"]

    # Top categories (max 5)
    top_cats = sorted(cat_weight, key=cat_weight.get, reverse=True)[:max_cats]
    vendors = list(vendor_profiles.keys())

    # Build matrix (fill missing with 0 instead of NaN)
    matrix = np.zeros((len(vendors), len(top_cats)))

    for vi, vn in enumerate(vendors):
        for ci, cat in enumerate(top_cats):
            if cat in vendor_profiles[vn]["aspect_profile"]:
                matrix[vi, ci] = vendor_profiles[vn]["aspect_profile"][cat]["score"]
            else:
                matrix[vi, ci] = 0.0  # Neutral fallback

    # Better dynamic sizing
    max_name_len = max(len(v) for v in vendors)
    fig_width = max(7, len(top_cats) * 1.4)
    fig_height = max(4, len(vendors) * 0.8)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Diverging colormap centered at 0
    im = ax.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)

    # Labels
    ax.set_xticks(range(len(top_cats)))
    ax.set_xticklabels(top_cats, rotation=30, ha="right", fontsize=9)

    ax.set_yticks(range(len(vendors)))
    ax.set_yticklabels(vendors, fontsize=8)

    # Add padding so labels don’t collide
    ax.tick_params(axis='y', pad=10)

    ax.set_title("Top Aspect Score Heatmap", fontsize=10, pad=10)

    # Annotate values with better scaling
    for vi in range(len(vendors)):
        for ci in range(len(top_cats)):
            val = matrix[vi, ci]
            ax.text(ci, vi, f"{val:+.2f}", ha="center", va="center", fontsize=6)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.04)
    cbar.ax.tick_params(labelsize=8)

    # Extra margin for long names
    plt.subplots_adjust(left=0.3)

    plt.tight_layout()
    return fig


def fig_vendor_radar(vendor_profiles, max_cats=6):
    """Radar chart comparing vendors across top aspect categories."""
    if len(vendor_profiles) < 2:
        return None
    cat_weight = defaultdict(float)
    for prof in vendor_profiles.values():
        for cat, av in prof["aspect_profile"].items():
            cat_weight[cat] += av["weight"]
    top_cats = [c for c, _ in sorted(cat_weight.items(), key=lambda x: -x[1])][:max_cats]
    if len(top_cats) < 3:
        return None
    N      = len(top_cats)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(5.5, 5.5), subplot_kw=dict(polar=True))
    ax.set_facecolor("#111")
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(top_cats, fontsize=7.5, color="#888")
    ax.set_ylim(-1, 1)
    ax.set_yticks([-0.5, 0, 0.5, 1])
    ax.set_yticklabels(["-0.5", "0", "0.5", "1.0"], fontsize=6, color="#555")
    ax.yaxis.set_tick_params(labelsize=6)
    cmap    = cm.get_cmap("tab10", len(vendor_profiles))
    for idx, (vn, prof) in enumerate(vendor_profiles.items()):
        vals = []
        for cat in top_cats:
            sc = prof["aspect_profile"].get(cat, {}).get("score", 0.0)
            vals.append(sc)
        vals += vals[:1]
        color = cmap(idx)
        ax.plot(angles, vals, linewidth=1.5, linestyle="solid", color=color, label=vn)
        ax.fill(angles, vals, alpha=0.10, color=color)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.18),
              ncol=2, fontsize=7.5, frameon=False, labelcolor="#bbb")
    ax.set_title("Vendor Aspect Radar", fontsize=9, pad=16)
    plt.tight_layout()
    return fig


def fig_elo_ranking(df_rank):
    vendors = df_rank["vendor"].tolist()[::-1]
    stars   = df_rank["star_rating"].tolist()[::-1]
    ci_lo   = df_rank["ci_lo"].tolist()[::-1]
    ci_hi   = df_rank["ci_hi"].tolist()[::-1]
    palette = ["#f39c12", "#aaaaaa", "#cd7f32"] + ["#2a4a6a"] * max(0, len(vendors) - 3)
    colors  = list(reversed(palette[:len(vendors)]))
    fig, ax = plt.subplots(figsize=(6, max(2.5, len(vendors) * 0.55)))
    bars = ax.barh(vendors, stars, color=colors, height=0.55, alpha=0.88)
    ax.errorbar(
        stars, vendors,
        xerr=[np.array(stars) - np.array(ci_lo), np.array(ci_hi) - np.array(stars)],
        fmt="none", color="#333", capsize=4, lw=1.5,
    )
    ax.set_xlim(0.5, 5.9)
    ax.axvline(3.0, color="#555", linestyle="--", lw=0.8)
    ax.set_xlabel("Star Rating (Elo-scaled)", fontsize=8)
    ax.set_title("Sentiment-Aware Elo Vendor Ranking", fontsize=9, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    for bar, val in zip(bars, stars):
        ax.text(val + 0.09, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", fontsize=9, fontweight="bold")
    plt.tight_layout()
    return fig


def fig_sentiment_donut(results_list, title="Sentiment Overview"):
    """Donut chart for sentiment breakdown of a review or vendor batch."""
    counts = Counter(r["sentiment"] for r in results_list)
    labels = [s for s in SENTIMENT_LABEL_NAMES if counts.get(s, 0) > 0]
    sizes  = [counts[s] for s in labels]
    colors = [c for s, c in zip(SENTIMENT_LABEL_NAMES, SENT_PALETTE) if counts.get(s, 0) > 0]
    if not sizes:
        return None

    fig, ax = plt.subplots(figsize=(3.8, 3.8))

    wedges, _ = ax.pie(
        sizes, labels=None, colors=colors,
        startangle=90,
        wedgeprops={"linewidth": 0.8, "edgecolor": "#0d0d0d", "width": 0.52},
    )

    total = sum(sizes)

    legend_labels = [
        f"{label} ({(100 * counts[label] / total):.2f}%)"
        for label in labels
    ]

    ax.legend(wedges, legend_labels, loc="lower center", bbox_to_anchor=(0.5, -0.14),
              ncol=2, fontsize=7.5, frameon=False, labelcolor="#aaa")

    ax.set_title(title, fontsize=9, pad=6)

    plt.tight_layout()
    return fig


def fig_aspect_count_bar(results_list, max_show=15):
    """Horizontal bar of most-mentioned aspects."""
    counts = Counter(normalise_aspect(r["aspect"]) for r in results_list)
    top    = counts.most_common(max_show)
    if not top:
        return None
    labels = [t[0] for t in top][::-1]
    vals   = [t[1] for t in top][::-1]
    h      = max(3.0, len(labels) * 0.38)
    fig, ax = plt.subplots(figsize=(5.5, h))
    ax.barh(labels, vals, color="#3498db", height=0.55, alpha=0.85)
    ax.set_xlabel("Mentions", fontsize=8)
    ax.set_title("Top Aspect Mentions", fontsize=9, pad=6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=7.5)
    ax.tick_params(axis="x", labelsize=7.5)
    for i, val in enumerate(vals):
        ax.text(val + 0.1, i, str(val), va="center", fontsize=7.5)
    plt.tight_layout()
    return fig


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
# Batch processing helper (chunked for large datasets)
# ---------------------------------------------------------------------------

def batch_process_reviews(reviews, ate_mdl, ate_tok, sent_mdl, sent_tok,
                           progress_bar=None, status_el=None, label=""):
    flat = []
    n    = len(reviews)
    for i, rev in enumerate(reviews):
        if status_el and label:
            status_el.markdown(
                f'<div style="font-family:IBM Plex Mono,monospace;font-size:0.72rem;'
                f'color:#3a3a3a;">Processing {label} ({i+1}/{n})</div>',
                unsafe_allow_html=True,
            )
        flat.extend(_run(rev, ate_mdl, ate_tok, sent_mdl, sent_tok))
        if progress_bar is not None:
            progress_bar.progress((i + 1) / n)
    return flat


# ---------------------------------------------------------------------------
# Sidebar: model selection
# ---------------------------------------------------------------------------

sb_head = (
    "font-family:IBM Plex Mono,monospace;font-size:0.62rem;text-transform:uppercase;"
    "letter-spacing:0.13em;color:#3a3a3a;padding-bottom:0.4rem;"
    "border-bottom:1px solid #161616;margin-bottom:0.9rem;"
)

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
        key="pipeline_mode",
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
                first = list(ensemble_dict.values())[0]
                st.session_state["active_ate_model"]   = first["mdl_ate"]
                st.session_state["active_ate_tok"]     = first["tok_ate"]
                st.session_state["active_sent_model"]  = first["mdl_sent"]
                st.session_state["active_sent_tok"]    = first["tok_sent"]
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
        f'<span style="display:inline-block;width:7px;height:7px;border-radius:50%;'
        f'background:{dot};margin-right:7px;vertical-align:middle;"></span>'
        f'<span style="color:{col};">{txt}</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(f'<div style="{sb_head};margin-top:1.3rem;">Settings</div>', unsafe_allow_html=True)
    use_norm = st.checkbox("Normalise aspect terms", value=True)
    use_elo  = st.checkbox(
        "Elo ranking (recommendation)", value=True,
        help="Uses Sentiment-Aware Elo. Disable for simple relative scaling.",
    )
    st.sidebar.write("Device:", str(DEVICE))


# ---------------------------------------------------------------------------
# Helper: run pipeline in correct mode
# ---------------------------------------------------------------------------

def _run(text, ate_model=None, ate_tok=None, sent_model=None, sent_tok=None):
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
    "Compare Vendors", "Model Evaluation", "Budget Planner", "How It Works",
])

# =============================================================================
# Tab 1: Single Review
# =============================================================================

with tab1:
    sec("Analyse a Single Review")
    info("Paste any review. The ATE model extracts aspect terms automatically.")
    if not models_ready():
        warn("Load models using the sidebar first.")

    review_text = st.text_area(
        "Review",
        placeholder="The food was amazing but the service was really slow.",
        height=115,
        key="t1_input",
    )

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
            star    = round(float(np.clip((overall + 1) / 2 * 5, 0, 5)), 2)
            n_pos   = sum(1 for r in res if r["sentiment"] == "positive")
            n_neg   = sum(1 for r in res if r["sentiment"] == "negative")
            n_neu   = sum(1 for r in res if r["sentiment"] == "neutral")
            n_con   = sum(1 for r in res if r["sentiment"] == "conflict")
            stat_row(
                ("Aspects found",   len(res)),
                ("Overall score",   f"{overall:+.3f}"),
                ("Star equivalent", f"{star:.2f} / 5"),
                ("Positive",        n_pos,  "#2ecc71"),
                ("Negative",        n_neg,  "#e74c3c"),
                ("Neutral",         n_neu,  "#3498db"),
                ("Conflict",        n_con,  "#f39c12"),
            )

            sec("Extracted Aspects")
            aspect_cards(res)

            sec("Sentiment and Aspect Breakdown")
            col_donut, col_probs = st.columns([1, 2])
            with col_donut:
                fig_d = fig_sentiment_donut(res)
                if fig_d:
                    st.pyplot(fig_d)
            with col_probs:
                sec("Probability Breakdown")
                prob_cols = st.columns(min(4, len(res)))
                for i, r in enumerate(res):
                    with prob_cols[i % len(prob_cols)]:
                        st.pyplot(fig_prob_bars(r["probs"], r["aspect"]))

            sec("Results Table")
            st.dataframe(
                pd.DataFrame([{
                    "aspect":     r["aspect"],
                    "sentiment":  r["sentiment"],
                    "score":      round(r["weighted_score"], 4),
                    "confidence": round(r["confidence"], 4),
                } for r in res]),
                use_container_width=True,
            )


# =============================================================================
# Tab 2: Single Vendor
# =============================================================================

with tab2:
    sec("Rate a Single Vendor")
    info("Upload a CSV with a <code>text</code> column, or a TXT file (one review per line).")
    if not models_ready():
        warn("Load models using the sidebar first.")

    vf = st.file_uploader("Reviews file (CSV or TXT)", type=["csv", "txt"], key="t2_upload")
    vname_override = st.text_input("Vendor name (optional)", key="t2_vname")

    if vf:
        reviews_t2 = load_reviews_from_file(vf)
        if reviews_t2:
            vname_t2 = vname_override.strip() or os.path.splitext(vf.name)[0]
            ok(f"Loaded {len(reviews_t2)} reviews for: {vname_t2}")
            if st.button("Run Analysis", type="primary", key="t2_run"):
                if not models_ready():
                    err("Models not loaded.")
                else:
                    ate_mdl, ate_tok, sent_mdl, sent_tok = get_models()
                    flat_t2 = []
                    prog_t2 = st.progress(0)
                    status_t2 = st.empty()
                    for i, rev in enumerate(reviews_t2):
                        status_t2.markdown(
                            f'<div style="font-family:IBM Plex Mono,monospace;font-size:0.72rem;'
                            f'color:#3a3a3a;">Analysing review {i+1}/{len(reviews_t2)}</div>',
                            unsafe_allow_html=True,
                        )
                        flat_t2.extend(_run(rev, ate_mdl, ate_tok, sent_mdl, sent_tok))
                        prog_t2.progress((i + 1) / len(reviews_t2))
                    status_t2.empty()
                    prof = compute_vendor_profile(flat_t2, len(reviews_t2), use_norm=use_norm)
                    if prof:
                        star = round(float(np.clip((prof["raw_score"] + 1) / 2 * 5, 0, 5)), 2)
                        st.session_state.update({
                            "t2_prof":  prof,
                            "t2_star":  star,
                            "t2_name":  vname_t2,
                            "t2_flat":  flat_t2,
                        })

    if "t2_prof" in st.session_state:
        prof    = st.session_state["t2_prof"]
        star    = st.session_state["t2_star"]
        vn      = st.session_state["t2_name"]
        flat_t2 = st.session_state.get("t2_flat", [])

        sec(f"Rating: {vn}")
        col_badge, col_donut = st.columns([1, 1])
        with col_badge:
            rating_badge(vn, star, prof["review_count"])
            stat_row(
                ("Raw score",     f"{prof['raw_score']:+.3f}"),
                ("Total aspects", sum(v["count"] for v in prof["aspect_profile"].values())),
                ("Unique aspects", len(prof["aspect_profile"])),
            )
        with col_donut:
            if flat_t2:
                fig_d2 = fig_sentiment_donut(flat_t2, title="Sentiment Mix")
                if fig_d2:
                    st.pyplot(fig_d2)

        sec("Aspect Breakdown")
        c1, c2 = st.columns([3, 2])
        with c1:
            st.pyplot(fig_aspect_scores(prof["aspect_profile"]))
        with c2:
            st.pyplot(fig_aspect_pie(prof["aspect_profile"]))

        sec("Sentiment Distribution by Aspect")
        if flat_t2:
            fig_sd = fig_sentiment_distribution(flat_t2)
            if fig_sd:
                st.pyplot(fig_sd)

        sec("Top Mentioned Aspects")
        if flat_t2:
            fig_ac = fig_aspect_count_bar(flat_t2)
            if fig_ac:
                st.pyplot(fig_ac)


# =============================================================================
# Tab 3: Recommendation (Elo ranking)
# =============================================================================

with tab3:
    sec("Vendor Recommendation System")
    info("Ratings use Sentiment-Aware Elo ranking. Best vendor always scores 5.0, "
         "worst always scores 1.0.")
    if not models_ready():
        warn("Load models using the sidebar first.")

    upload_mode_t3 = st.radio(
        "Upload mode",
        ["Multiple files (one per vendor)", "Single CSV with vendor and text columns"],
        key="t3_mode",
        horizontal=True,
    )

    vendor_map_rec = {}
    if upload_mode_t3 == "Multiple files (one per vendor)":
        rec_files = st.file_uploader(
            "Upload vendor review files", type=["csv", "txt"],
            accept_multiple_files=True, key="t3_upload_multi",
        )
        if rec_files:
            for f in rec_files:
                revs = load_reviews_from_file(f)
                if revs:
                    vendor_map_rec[os.path.splitext(f.name)[0]] = revs
    else:
        single_csv = st.file_uploader(
            "Combined CSV (vendor + text columns)", type=["csv"], key="t3_upload_single",
        )
        if single_csv:
            raw_sc = single_csv.read()
            df_sc  = pd.read_csv(io.BytesIO(raw_sc))
            if {"vendor", "text"} <= set(df_sc.columns):
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
        if not models_ready():
            err("Models not loaded.")
        else:
            ate_mdl, ate_tok, sent_mdl, sent_tok = get_models()
            total   = sum(len(v) for v in vendor_map_rec.values())
            done    = 0
            prog    = st.progress(0)
            status  = st.empty()
            profs   = {}
            flat_all = {}
            ranker  = SentimentEloRanker(k=32, initial=1500, bayes_m=10)
            for vn, reviews in vendor_map_rec.items():
                flat = []
                for rev in reviews:
                    status.markdown(
                        f'<div style="font-family:IBM Plex Mono,monospace;font-size:0.72rem;'
                        f'color:#3a3a3a;">Processing {vn} ({done+1}/{total})</div>',
                        unsafe_allow_html=True,
                    )
                    flat.extend(_run(rev, ate_mdl, ate_tok, sent_mdl, sent_tok))
                    done += 1
                    prog.progress(done / total)
                prof = compute_vendor_profile(flat, len(reviews), use_norm=use_norm)
                if prof:
                    profs[vn]    = prof
                    flat_all[vn] = flat
                    ranker.add_vendor(vn, prof)
            status.empty()
            df_rank = ranker.get_rankings()
            if not df_rank.empty:
                rows = []
                for vn, prof in profs.items():
                    row = {"vendor": vn}
                    for a, av in prof["aspect_profile"].items():
                        row[f"asp_{a}"] = round(av["score"], 3)
                    rows.append(row)
                df_asp  = pd.DataFrame(rows)
                df_rank = df_rank.merge(df_asp, on="vendor", how="left")
            st.session_state["t3_rank"]     = df_rank
            st.session_state["t3_profs"]    = profs
            st.session_state["t3_flat_all"] = flat_all

    if "t3_rank" in st.session_state:
        df_r     = st.session_state["t3_rank"]
        prfs     = st.session_state["t3_profs"]
        flat_all = st.session_state.get("t3_flat_all", {})

        if not df_r.empty:
            top = df_r.iloc[0]
            sec("Top Recommendation")
            rating_badge(top["vendor"], top["star_rating"], top["reviews"])

            sec("Elo Leaderboard")
            st.dataframe(
                df_r[["rank", "vendor", "star_rating", "elo_rating", "raw_score", "reviews"]],
                use_container_width=True, hide_index=True,
            )

            sec("Ranking Chart")
            st.pyplot(fig_elo_ranking(df_r))

            if len(prfs) >= 2:
                sec("Vendor Comparison Charts")
                col_heat, col_radar = st.columns([3, 2])
                with col_heat:
                    fig_hm = fig_score_heatmap(prfs)
                    if fig_hm:
                        st.pyplot(fig_hm)
                with col_radar:
                    fig_rad = fig_vendor_radar(prfs)
                    if fig_rad:
                        st.pyplot(fig_rad)

            sec("Vendor Drill Down")
            selected_vn = st.selectbox("Select vendor", list(prfs.keys()), key="t3_drill")
            if selected_vn in prfs:
                p = prfs[selected_vn]["aspect_profile"]
                c1, c2 = st.columns([3, 2])
                with c1:
                    st.pyplot(fig_aspect_scores(p))
                with c2:
                    st.pyplot(fig_aspect_pie(p))
                if selected_vn in flat_all:
                    fig_sd3 = fig_sentiment_distribution(flat_all[selected_vn])
                    if fig_sd3:
                        st.pyplot(fig_sd3)


# =============================================================================
# Tab 4: Compare Vendors (absolute)
# =============================================================================

with tab4:
    sec("Compare Vendors (Absolute Scores)")
    info("Raw scores mapped from [-1,+1] to [0,5]. Use this for absolute comparison "
         "rather than relative ranking.")
    if not models_ready():
        warn("Load models using the sidebar first.")

    upload_mode_t4 = st.radio(
        "Upload mode",
        ["Multiple files (one per vendor)", "Single CSV with vendor and text columns"],
        key="t4_mode",
        horizontal=True,
    )

    vendor_map_cmp = {}
    if upload_mode_t4 == "Multiple files (one per vendor)":
        cmp_files = st.file_uploader(
            "Upload vendor review files", type=["csv", "txt"],
            accept_multiple_files=True, key="t4_upload",
        )
        if cmp_files:
            for f in cmp_files:
                revs = load_reviews_from_file(f)
                if revs:
                    vendor_map_cmp[os.path.splitext(f.name)[0]] = revs
    else:
        single_csv_t4 = st.file_uploader("Combined CSV", type=["csv"], key="t4_upload_single")
        if single_csv_t4:
            raw_t4 = single_csv_t4.read()
            df_t4s = pd.read_csv(io.BytesIO(raw_t4))
            if {"vendor", "text"} <= set(df_t4s.columns):
                df_t4s["vendor"] = df_t4s["vendor"].astype(str).str.strip()
                df_t4s["text"]   = df_t4s["text"].astype(str).str.strip()
                df_t4s = df_t4s[df_t4s["text"].str.len() > 0]
                for vn, grp in df_t4s.groupby("vendor", sort=False):
                    vendor_map_cmp[vn] = grp["text"].tolist()

    if vendor_map_cmp:
        ok(f"Loaded {sum(len(v) for v in vendor_map_cmp.values())} reviews.")

    if vendor_map_cmp and st.button("Run Comparison", type="primary", key="t4_run"):
        if not models_ready():
            err("Models not loaded.")
        else:
            ate_mdl, ate_tok, sent_mdl, sent_tok = get_models()
            total    = sum(len(v) for v in vendor_map_cmp.values())
            done     = 0
            prog     = st.progress(0)
            status   = st.empty()
            rows_c   = {}
            flat_all4 = {}
            for vn, reviews in vendor_map_cmp.items():
                flat = []
                for rev in reviews:
                    status.markdown(
                        f'<div style="font-family:IBM Plex Mono,monospace;font-size:0.72rem;'
                        f'color:#3a3a3a;">Processing {vn} ({done+1}/{total})</div>',
                        unsafe_allow_html=True,
                    )
                    flat.extend(_run(rev, ate_mdl, ate_tok, sent_mdl, sent_tok))
                    done += 1
                    prog.progress(done / total)
                prof = compute_vendor_profile(flat, len(reviews), use_norm=use_norm)
                if prof:
                    star = round(float(np.clip((prof["raw_score"] + 1) / 2 * 5, 0, 5)), 2)
                    rows_c[vn]    = {**prof, "star_rating": star}
                    flat_all4[vn] = flat
            status.empty()
            df_c4 = pd.DataFrame([
                {"vendor":      vn,
                 "star_rating": v["star_rating"],
                 "raw_score":   v["raw_score"],
                 "reviews":     v["review_count"]}
                for vn, v in rows_c.items()
            ]).sort_values("star_rating", ascending=False).reset_index(drop=True)
            df_c4.insert(0, "rank", range(1, len(df_c4) + 1))
            st.session_state["t4_df"]      = df_c4
            st.session_state["t4_profs"]   = rows_c
            st.session_state["t4_flat_all"] = flat_all4

    if "t4_df" in st.session_state:
        df_c4     = st.session_state["t4_df"]
        prfs_c    = st.session_state["t4_profs"]
        flat_all4 = st.session_state.get("t4_flat_all", {})

        sec("Leaderboard (Absolute)")
        st.dataframe(df_c4, use_container_width=True, hide_index=True)

        if len(prfs_c) >= 2:
            sec("Vendor Comparison Charts")
            col_hm4, col_rad4 = st.columns([3, 2])
            with col_hm4:
                fig_hm4 = fig_score_heatmap(prfs_c)
                if fig_hm4:
                    st.pyplot(fig_hm4)
            with col_rad4:
                fig_rad4 = fig_vendor_radar(prfs_c)
                if fig_rad4:
                    st.pyplot(fig_rad4)

        sec("Vendor Drill Down")
        sel = st.selectbox("Select vendor", list(prfs_c.keys()), key="t4_drill")
        if sel in prfs_c:
            p = prfs_c[sel]["aspect_profile"]
            c1, c2 = st.columns([3, 2])
            with c1:
                st.pyplot(fig_aspect_scores(p))
            with c2:
                st.pyplot(fig_aspect_pie(p))
            if sel in flat_all4:
                fig_sd4 = fig_sentiment_distribution(flat_all4[sel])
                if fig_sd4:
                    st.pyplot(fig_sd4)


# =============================================================================
# Tab 5: Model Evaluation
# =============================================================================

with tab5:
    sec("Model Evaluation")
    info("Upload reviews and evaluate confidence, sentiment distribution, and aspect extraction rate.")
    if not models_ready():
        warn("Load models using the sidebar first.")

    eval_upload = st.file_uploader(
        "Upload CSV (text + optional vendor columns)",
        type=["csv", "txt"],
        accept_multiple_files=True,
        key="t5_upload",
    )
    conf_threshold = st.slider(
        "Low-confidence threshold", 0.4, 0.9, 0.6, 0.05, key="t5_thresh",
    )

    eval_vendor_map = {}
    if eval_upload:
        for f in eval_upload:
            revs = load_reviews_from_file(f)
            if revs:
                eval_vendor_map[os.path.splitext(f.name)[0]] = revs

    if eval_vendor_map:
        ok(f"Loaded {sum(len(v) for v in eval_vendor_map.values())} reviews.")

    if eval_vendor_map and st.button("Run Evaluation", type="primary", key="t5_run"):
        if not models_ready():
            err("Models not loaded.")
        else:
            ate_mdl, ate_tok, sent_mdl, sent_tok = get_models()
            all_confs, all_scores, all_sents, all_asp_counts = [], [], [], []
            all_results_flat = []
            vendor_confs, vendor_low_pct = {}, {}
            total = sum(len(v) for v in eval_vendor_map.values())
            done  = 0
            prog  = st.progress(0)
            status = st.empty()

            for vn, reviews in eval_vendor_map.items():
                v_confs = []
                for rev in reviews:
                    status.markdown(
                        f'<div style="font-family:IBM Plex Mono,monospace;font-size:0.72rem;'
                        f'color:#3a3a3a;">Evaluating {vn} ({done+1}/{total})</div>',
                        unsafe_allow_html=True,
                    )
                    results = _run(rev, ate_mdl, ate_tok, sent_mdl, sent_tok)
                    all_asp_counts.append(len(results))
                    for r in results:
                        all_confs.append(r["confidence"])
                        all_scores.append(r["weighted_score"])
                        all_sents.append(r["sentiment"])
                        v_confs.append(r["confidence"])
                        all_results_flat.append(r)
                    done += 1
                    prog.progress(done / total)
                vendor_confs[vn] = v_confs
                if v_confs:
                    vendor_low_pct[vn] = round(
                        sum(1 for c in v_confs if c < conf_threshold) / len(v_confs) * 100, 1,
                    )
            status.empty()

            st.session_state["t5_eval"] = {
                "all_confs":       all_confs,
                "all_scores":      all_scores,
                "all_sents":       all_sents,
                "all_asp_counts":  all_asp_counts,
                "vendor_confs":    vendor_confs,
                "vendor_low_pct":  vendor_low_pct,
                "n_reviews":       total,
                "thresh":          conf_threshold,
                "all_results":     all_results_flat,
            }

    if "t5_eval" in st.session_state:
        ev          = st.session_state["t5_eval"]
        total_preds = len(ev["all_confs"])
        low_conf    = sum(1 for c in ev["all_confs"] if c < ev["thresh"])
        zero_asp    = sum(1 for c in ev["all_asp_counts"] if c == 0)
        mean_asp    = float(np.mean(ev["all_asp_counts"])) if ev["all_asp_counts"] else 0

        sec("Overall Statistics")
        stat_row(
            ("Reviews",           ev["n_reviews"]),
            ("Total predictions", total_preds),
            ("Mean confidence",   f"{np.mean(ev['all_confs']):.3f}" if ev["all_confs"] else "N/A"),
            ("Low confidence",    f"{low_conf} ({low_conf/max(total_preds,1)*100:.1f}%)", "#f39c12"),
            ("Zero-aspect revs",  zero_asp, "#e74c3c"),
            ("Mean aspects/rev",  f"{mean_asp:.2f}"),
        )

        sec("Confidence and Sentiment Distribution")
        col_conf_hist, col_sent_donut = st.columns([3, 2])
        with col_conf_hist:
            fig_ch = fig_confidence_histogram(ev["all_confs"], ev["thresh"])
            if fig_ch:
                st.pyplot(fig_ch)
        with col_sent_donut:
            if ev["all_results"]:
                fig_ds = fig_sentiment_donut(ev["all_results"], title="Overall Sentiment")
                if fig_ds:
                    st.pyplot(fig_ds)

        if ev.get("all_results"):
            sec("Aspect Mention Distribution")
            fig_acd = fig_aspect_count_bar(ev["all_results"])
            if fig_acd:
                st.pyplot(fig_acd)

            sec("Sentiment Distribution by Aspect")
            fig_sda = fig_sentiment_distribution(ev["all_results"])
            if fig_sda:
                st.pyplot(fig_sda)

        sec("Per-Vendor Low-Confidence Rate")
        if ev["vendor_low_pct"]:
            df_vlcp = pd.DataFrame([
                {
                    "vendor":         vn,
                    "low_conf_pct":   pct,
                    "n_predictions":  len(ev["vendor_confs"][vn]),
                }
                for vn, pct in sorted(ev["vendor_low_pct"].items(), key=lambda x: -x[1])
            ])
            st.dataframe(df_vlcp, use_container_width=True, hide_index=True)


# =============================================================================
# Tab 6: Budget Planner
# =============================================================================

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
    absa_score:     float
    review_count:   int = 0
    aspect_profile: dict = field(default_factory=dict)

    @property
    def norm_score(self) -> float:
        return float(np.clip((self.absa_score + 1) / 2, 0.0, 1.0))

    @property
    def star_rating(self) -> float:
        return float(np.clip((self.absa_score + 1) / 2 * 5, 0.0, 5.0))


def _priority_weights(priority_order, lam=2.0):
    k     = len(priority_order)
    ranks = np.array([k - i for i in range(k)], dtype=float)
    w     = _softmax(lam * ranks)
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
        cands    = vby_cat.get(cat, [])
        aff      = _best_affordable(cands, alloc)
        sel[cat] = aff[0] if aff else None
    return sel


def _ilp(total_budget, priority_order, vby_cat, lam=2.0, cat_caps=None):
    if not _ILP_AVAILABLE:
        return None, None
    w    = _priority_weights(priority_order, lam)
    cats = [c for c in priority_order if c in vby_cat]
    prob = _pulp.LpProblem("VendorBudget", _pulp.LpMaximize)
    x    = {
        (cat, v.name): _pulp.LpVariable(
            f"x_{cat}_{v.name}".replace(" ", "_"), cat=_pulp.LpBinary,
        )
        for cat in cats for v in vby_cat[cat]
    }
    prob += _pulp.lpSum(
        w[cat] * v.norm_score * x[(cat, v.name)]
        for cat in cats for v in vby_cat[cat]
    )
    for cat in cats:
        prob += _pulp.lpSum(x[(cat, v.name)] for v in vby_cat[cat]) == 1
    prob += _pulp.lpSum(
        v.price * x[(cat, v.name)] for cat in cats for v in vby_cat[cat]
    ) <= total_budget
    if cat_caps:
        for cat in cats:
            if cat in cat_caps:
                prob += _pulp.lpSum(
                    v.price * x[(cat, v.name)] for v in vby_cat[cat]
                ) <= cat_caps[cat]
    _pulp.PULP_CBC_CMD(msg=False).solve(prob)
    sel  = {}
    cost = 0.0
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
        if cat_caps and any(
            combo[i].price > cat_caps.get(cats[i], 1e18)
            for i in range(len(cats))
        ):
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
        sel    = greedy
        cost   = sum(v.price for v in sel.values() if v is not None)
        solver = "greedy"
    else:
        if _ILP_AVAILABLE:
            sel, cost = _ilp(total_budget, priority_order, vby_cat, lam, cat_caps=splits)
            solver = "ILP"
        else:
            sel, cost = _exhaustive(total_budget, priority_order, vby_cat, lam, cat_caps=splits)
            solver = "exhaustive"
        if sel is None:
            sel    = greedy
            cost   = sum(v.price for v in greedy.values() if v is not None)
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
            for cat, cur in sorted(sel.items(), key=lambda x: -norm_pct.get(x[0], 0)):
                cands = vby_cat.get(cat, [])
                cur_p = cur.price if cur else 0.0
                ups   = [v for v in cands
                         if v.norm_score > (cur.norm_score if cur else -1)
                         and v.price <= cur_p + savings]
                if ups:
                    best    = max(ups, key=lambda v: v.norm_score)
                    delta   = best.price - cur_p
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
#  sample data
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
    sel  = result["selection"]
    cats = [c for c, v in sel.items() if v is not None]
    if not cats:
        return None
    budgets = (
        [result["splits"][c] for c in cats] if "splits" in result
        else [result["budgets"][c] for c in cats]
    )
    prices = [sel[c].price for c in cats]
    names  = [sel[c].name  for c in cats]
    scores = [sel[c].absa_score for c in cats]
    cmap   = cm.get_cmap("tab10", len(cats))
    colors = [cmap(i) for i in range(len(cats))]

    fig, axes = plt.subplots(1, 2, figsize=(13, max(3.5, len(cats) * 0.6)))

    wedges, _, atxts = axes[0].pie(
        budgets, labels=None, autopct="%1.1f%%", colors=colors,
        startangle=90,
        wedgeprops={"linewidth": 0.8, "edgecolor": "#0d0d0d"},
        pctdistance=0.78,
    )
    for at in atxts:
        at.set_fontsize(8)
        at.set_color("#ffffff")
    axes[0].legend(wedges, cats, loc="lower center", bbox_to_anchor=(0.5, -0.18),
                   ncol=2, fontsize=8, frameon=False, labelcolor="#ccc")
    axes[0].set_title("Budget Allocation", fontsize=10, pad=8)

    y = np.arange(len(cats))
    axes[1].barh(y, budgets[::-1], height=0.55,
                 color=[(*c[:3], 0.22) for c in colors[::-1]], label="Allocated")
    axes[1].barh(y, prices[::-1], height=0.35, color=colors[::-1], label="Actual spend")
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(cats[::-1], fontsize=9)
    axes[1].set_xlabel("Amount", fontsize=9)
    axes[1].set_title("Allocated vs Actual Spend", fontsize=10, pad=8)
    axes[1].spines[["top", "right"]].set_visible(False)
    axes[1].legend(fontsize=8, frameon=False, labelcolor="#aaa")
    mx = max(budgets) if budgets else 1
    for i, (cat, name, sc) in enumerate(zip(cats[::-1], names[::-1], scores[::-1])):
        axes[1].text(mx * 0.01, i, f"{name}  ({sc:+.2f})", va="center",
                     fontsize=7.5, color="#ddd")
    fig.suptitle(mode_label, fontsize=11, y=1.01, color="#ffffff")
    plt.tight_layout()
    return fig


def _fig_quality_price(vby_cat, selected=None):
    cats   = list(vby_cat.keys())
    cmap   = cm.get_cmap("tab10", len(cats))
    colors = {cat: cmap(i) for i, cat in enumerate(cats)}
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for cat, vlist in vby_cat.items():
        ax.scatter(
            [v.price for v in vlist], [v.absa_score for v in vlist],
            color=colors[cat], s=65, alpha=0.75, label=cat, zorder=2,
        )
        for v in vlist:
            ax.annotate(
                v.name, (v.price, v.absa_score),
                textcoords="offset points", xytext=(6, 2),
                fontsize=6.5, color="#888", clip_on=True,
            )
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
    sel  = result["selection"]
    cats = list(sel.keys())
    th   = (
        "padding:0.4rem 0.75rem;text-align:left;color:#3a3a3a;font-size:0.6rem;"
        "text-transform:uppercase;letter-spacing:0.1em;border-bottom:1px solid #161616;"
        "font-family:IBM Plex Mono,monospace;"
    )
    td   = (
        "padding:0.38rem 0.75rem;border-bottom:1px solid #111;"
        "font-family:IBM Plex Mono,monospace;font-size:0.8rem;"
    )
    heads = ["Category", "Allocated Budget", "Vendor", "Price", "ABSA Score", "Stars", "Budget Used"]
    hrow  = "".join(f'<th style="{th}">{h}</th>' for h in heads)
    rows  = ""
    for cat in cats:
        v     = sel[cat]
        alloc = result["splits"].get(cat, 0) if mode == "auto" else result["budgets"].get(cat, 0)
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
    return (
        f'<div class="leaderboard-wrap"><table class="lb-table">'
        f'<thead><tr style="background:#090909;">{hrow}</tr></thead>'
        f'<tbody>{rows}</tbody></table></div>'
    )


# ---------------------------------------------------------------------------
# Tab 6: Budget Planner UI
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

    sec("Step 1 - Vendor Data")

    use_sample = st.toggle(
        "Use pre-loaded example vendors",
        value=False,
        key="bp_use_sample",
        help="Loads a built-in set of 23 vendors across 5 categories.",
    )

    bp_vendors_by_cat = {}

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
            cached   = st.session_state.get("bp_scored_cache", {})
            ate_mdl, ate_tok, sent_mdl, sent_tok = get_models()

            for f in bp_files:
                vname = os.path.splitext(f.name)[0]
                if vname in cached:
                    continue
                revs = load_reviews_from_file(f)
                if not revs:
                    continue
                flat     = []
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
                st.markdown(
                    '<div style="font-family:IBM Plex Mono,monospace;font-size:0.68rem;'
                    'text-transform:uppercase;letter-spacing:0.1em;color:#3a3a3a;'
                    'margin-bottom:0.6rem;">Vendor Configuration</div>',
                    unsafe_allow_html=True,
                )
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
                            "Category", key=f"bp_cat_{vn}",
                            placeholder="e.g. venue", label_visibility="collapsed",
                        )
                    with c3:
                        st.number_input(
                            "Price", min_value=0, value=0, step=1000,
                            key=f"bp_price_{vn}", label_visibility="collapsed",
                        )
                    with c4:
                        score = prof_data["raw_score"]
                        sc    = "#2ecc71" if score >= 0 else "#e74c3c"
                        st.markdown(
                            f'<div style="font-family:IBM Plex Mono,monospace;font-size:0.82rem;'
                            f'color:{sc};padding-top:0.55rem;">ABSA {score:+.3f}</div>',
                            unsafe_allow_html=True,
                        )
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
            )

        if mode_bp == "Auto - priority-based split":
            sec("Step 3 - Priority Order")
            info(
                "Drag the categories into your preferred order using the multiselect below. "
                "Categories listed first receive a larger share of the budget."
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
                help="0.5 = nearly equal split. 4.0 = top category gets much more.",
            )
            if priority_bp:
                splits_preview  = _auto_split(total_budget_bp, priority_bp, lam_bp)
                weights_preview = _priority_weights(priority_bp, lam_bp)
                st.dataframe(
                    pd.DataFrame([{
                        "category":          cat,
                        "priority_weight":   f"{weights_preview[cat]:.3f}",
                        "allocated":         f"{splits_preview[cat]:,.0f}",
                        "vendors_available": len(bp_vendors_by_cat.get(cat, [])),
                    } for cat in priority_bp]),
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
                                total_budget_bp, priority_bp, bp_vendors_by_cat, lam_bp,
                            )
                        st.session_state["bp_result"] = ("auto", res_auto, bp_vendors_by_cat)

        else:
            sec("Step 3 - Percentage Split")
            info(
                "Set the percentage of your budget to allocate to each category. "
                "Values do not need to sum to exactly 100; they are normalised automatically."
            )
            pct_inputs  = {}
            n_cats      = len(all_cats)
            default_pct = round(100 / n_cats, 1) if n_cats else 20.0
            cols_pct    = st.columns(min(n_cats, 5))
            for i, cat in enumerate(all_cats):
                with cols_pct[i % len(cols_pct)]:
                    pct_inputs[cat] = st.number_input(
                        cat, min_value=0.0, max_value=100.0,
                        value=default_pct, step=1.0, key=f"bp_pct_{cat}",
                    )
            total_entered = sum(pct_inputs.values())
            if total_entered > 0:
                st.markdown(
                    f'<div style="font-family:IBM Plex Mono,monospace;font-size:0.75rem;'
                    f'color:#444;margin-bottom:0.6rem;">'
                    f'Entered total: {total_entered:.1f}%  (will be normalised to 100%)</div>',
                    unsafe_allow_html=True,
                )
            reallocate_bp = st.checkbox(
                "Reallocate savings to upgrade other categories",
                value=True,
                key="bp_reallocate",
            )
            run_manual = st.button("Find Best Combination", type="primary", key="bp_run_manual")
            if run_manual:
                active_pct = {cat: pct for cat, pct in pct_inputs.items() if pct > 0}
                if not active_pct:
                    err("Set a non-zero percentage for at least one category.")
                else:
                    with st.spinner("Optimising..."):
                        res_manual = _recommend_manual(
                            total_budget_bp, active_pct, bp_vendors_by_cat, reallocate_bp,
                        )
                    st.session_state["bp_result"] = ("manual", res_manual, bp_vendors_by_cat)

    if "bp_result" in st.session_state:
        mode_used, result_bp, vby_cat_bp = st.session_state["bp_result"]
        sel_bp = result_bp["selection"]

        sec("Recommended Combination")

        selected_vendors = [v for v in sel_bp.values() if v is not None]
        missing_cats     = [cat for cat, v in sel_bp.items() if v is None]
        avg_score        = float(np.mean([v.absa_score for v in selected_vendors])) \
                           if selected_vendors else 0.0
        avg_star         = float(np.clip((avg_score + 1) / 2 * 5, 0, 5))

        stat_row(
            ("Total cost",       f"{result_bp['total_cost']:,.0f}"),
            ("Remaining budget", f"{result_bp['remaining']:,.0f}"),
            ("Vendors selected", len(selected_vendors)),
            ("Avg ABSA score",   f"{avg_score:+.3f}"),
            ("Avg stars",        f"{avg_star:.2f} / 5.00"),
            ("Quality index",    f"{result_bp['obj']:.4f}",
             "#2ecc71" if result_bp["obj"] >= 0.5 else "#f39c12"),
        )

        if missing_cats:
            warn(
                f"No affordable vendor was found for: {', '.join(missing_cats)}. "
                "Consider increasing your budget or adding lower-cost vendors."
            )

        st.markdown(_result_table_html(result_bp, mode_used), unsafe_allow_html=True)

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
            alloc = (result_bp["splits"].get(cat, 0) if mode_used == "auto"
                     else result_bp["budgets"].get(cat, 0))
            rows_dl.append({
                "category":         cat,
                "allocated_budget": round(alloc, 2),
                "vendor":           v.name if v else None,
                "price":            v.price if v else None,
                "absa_score":       round(v.absa_score, 4) if v else None,
                "star_rating":      round(v.star_rating, 2) if v else None,
                "review_count":     v.review_count if v else None,
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