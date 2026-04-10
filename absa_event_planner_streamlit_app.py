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
import absa_event_matcher

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


# ---------------------------------------------------------------------------
# Model Evaluation chart helpers
# ---------------------------------------------------------------------------

def fig_confidence_hist(all_confs, threshold):
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    if not all_confs:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return fig
    bins = np.linspace(0, 1, 26)
    lo_mask = np.array(all_confs) < threshold
    ax.hist(np.array(all_confs)[~lo_mask], bins=bins, color="#2ecc71",
            alpha=0.82, label=f"conf >= {threshold:.2f}")
    if lo_mask.any():
        ax.hist(np.array(all_confs)[lo_mask], bins=bins, color="#e74c3c",
                alpha=0.82, label=f"conf < {threshold:.2f}")
    ax.axvline(threshold, color="#f39c12", linewidth=1.2, linestyle="--",
               label="threshold")
    ax.set_xlabel("Confidence", fontsize=8)
    ax.set_ylabel("Count", fontsize=8)
    ax.set_title("Confidence Distribution", fontsize=9, pad=6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=7.5)
    ax.legend(fontsize=7.5, frameon=False, labelcolor="#aaa")
    plt.tight_layout()
    return fig


def fig_sentiment_dist(all_sents):
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    if not all_sents:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return fig
    counts  = [all_sents.count(l) for l in SENTIMENT_LABEL_NAMES]
    total   = max(sum(counts), 1)
    pcts    = [c / total * 100 for c in counts]
    bars    = ax.barh(SENTIMENT_LABEL_NAMES, pcts, color=SENT_PALETTE, height=0.52)
    ax.set_xlim(0, max(pcts) * 1.22)
    ax.set_xlabel("Percentage of predictions (%)", fontsize=8)
    ax.set_title("Sentiment Distribution", fontsize=9, pad=6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)
    for bar, pct, cnt in zip(bars, pcts, counts):
        ax.text(pct + 0.4, bar.get_y() + bar.get_height() / 2,
                f"{pct:.1f}%  ({cnt})", va="center", fontsize=7.5)
    plt.tight_layout()
    return fig


def fig_aspects_per_review(all_asp_counts):
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    if not all_asp_counts:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return fig
    mx   = max(all_asp_counts)
    bins = np.arange(0, mx + 2) - 0.5
    ax.hist(all_asp_counts, bins=bins, color="#3498db", alpha=0.85)
    mean_v = float(np.mean(all_asp_counts))
    ax.axvline(mean_v, color="#f39c12", linewidth=1.2, linestyle="--",
               label=f"mean {mean_v:.2f}")
    ax.set_xlabel("Aspects extracted per review", fontsize=8)
    ax.set_ylabel("Reviews", fontsize=8)
    ax.set_title("Aspects per Review", fontsize=9, pad=6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=7.5)
    ax.legend(fontsize=7.5, frameon=False, labelcolor="#aaa")
    plt.tight_layout()
    return fig


def fig_vendor_conf_box(vendor_confs, threshold):
    vendors = list(vendor_confs.keys())
    data    = [vendor_confs[vn] for vn in vendors]
    data    = [d if d else [0.0] for d in data]
    h       = max(3.0, len(vendors) * 0.5)
    fig, ax = plt.subplots(figsize=(7.0, h))
    bp = ax.boxplot(data, vert=False, patch_artist=True,
                    medianprops={"color": "#f39c12", "linewidth": 1.8},
                    whiskerprops={"color": "#555"},
                    capprops={"color": "#555"},
                    flierprops={"marker": "o", "markersize": 3,
                                "markerfacecolor": "#e74c3c", "alpha": 0.5})
    for patch in bp["boxes"]:
        patch.set_facecolor("#1a2a3a")
        patch.set_edgecolor("#2a4a6a")
    ax.set_yticks(range(1, len(vendors) + 1))
    ax.set_yticklabels(vendors, fontsize=8)
    ax.set_xlim(0, 1.05)
    ax.axvline(threshold, color="#e74c3c", linewidth=1.0, linestyle="--",
               label=f"threshold {threshold:.2f}")
    ax.set_xlabel("Confidence", fontsize=8)
    ax.set_title("Per-Vendor Confidence Distribution", fontsize=9, pad=6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=7.5)
    ax.legend(fontsize=7.5, frameon=False, labelcolor="#aaa")
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
# DB persistence: unified vendor_profiles table
# Populated by Tab 3 (Elo), Tab 4 (Compare/Relative), Tab 6 (Budget Planner)
# ---------------------------------------------------------------------------

def get_mysql_conn():
    """Connects directly to the Eventify MySQL database."""
    import pymysql
    from urllib.parse import urlparse
    
    db_url = os.environ.get("DATABASE_URL", "mysql+pymysql://eventify_user:eventify_password@localhost:3306/eventify_db")
    parsed = urlparse(db_url)
    return pymysql.connect(
        host=parsed.hostname or "localhost",
        user=parsed.username or "eventify_user",
        password=parsed.password or "eventify_password",
        database=parsed.path[1:] if parsed.path else "eventify_db",
        port=parsed.port or 3306
    )


def _aspect_val(profile, key):
    entry = profile.get(key)
    return float(entry["score"]) if entry else None


def _db_insert_vendor_profiles(rows):
    """
    Creates User accounts, Vendor Profiles, AND imports the CSV Reviews 
    into the Eventify database.
    """
    from werkzeug.security import generate_password_hash
    import uuid
    import streamlit as st # needed to grab the cached reviews

    if not rows:
        return None, 0, "no rows to insert"

    try:
        conn = get_mysql_conn()
        cur  = conn.cursor()
        n = 0
        run_id = str(uuid.uuid4())

        # Grab the raw review files we cached earlier
        cached_reviews = st.session_state.get("bp_scored_cache", {})

        for r in rows:
            vendor_name = r["vendor"]
            clean_name = "".join(e for e in vendor_name if e.isalnum()).lower()
            email = f"{clean_name[:20]}@vendor.com"
            
            # 1. User Account
            cur.execute("SELECT id FROM users WHERE email = %s", (email,))
            user_row = cur.fetchone()
            if not user_row:
                pw_hash = generate_password_hash("demo1234")
                cur.execute(
                    "INSERT INTO users (first_name, last_name, email, password_hash, role) VALUES (%s, %s, %s, %s, %s)",
                    (vendor_name[:80], "Vendor", email, pw_hash, "vendor")
                )
                user_id = cur.lastrowid
            else:
                user_id = user_row[0]
                
            # 2. Vendor Profile
            cur.execute("SELECT id FROM vendor_profiles WHERE user_id = %s", (user_id,))
            vp_row = cur.fetchone()
            
            cat = r.get("category") or "other"
            price = r.get("price") or 0
            city = r.get("city") or ""
            phone = r.get("phone") or ""
            tags = r.get("tags") or ""
            desc = r.get("description") or "" # New description
            
            if not vp_row:
                cur.execute("""
                    INSERT INTO vendor_profiles 
                    (user_id, business_name, category, min_price, max_price, 
                     city, phone, tags, description, raw_score, star_rating, review_count,
                     aspect_service, aspect_value, aspect_reliability, 
                     aspect_quality, aspect_ambiance, aspect_experience)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    user_id, vendor_name, cat, price, price, city, phone, tags, desc,
                    r.get("raw_score", 0), r.get("star_rating", 3), r.get("review_count", 0),
                    r.get("aspect_service"), r.get("aspect_value"), r.get("aspect_reliability"),
                    r.get("aspect_quality"), r.get("aspect_ambiance"), r.get("aspect_experience")
                ))
                vp_id = cur.lastrowid
            else:
                cur.execute("""
                    UPDATE vendor_profiles 
                    SET raw_score=%s, star_rating=%s, review_count=%s, min_price=%s, max_price=%s, category=%s,
                        city=%s, phone=%s, tags=%s, description=%s,
                        aspect_service=%s, aspect_value=%s, aspect_reliability=%s, 
                        aspect_quality=%s, aspect_ambiance=%s, aspect_experience=%s
                    WHERE user_id=%s
                """, (
                    r.get("raw_score", 0), r.get("star_rating", 3), r.get("review_count", 0),
                    price, price, cat, city, phone, tags, desc,
                    r.get("aspect_service"), r.get("aspect_value"), r.get("aspect_reliability"),
                    r.get("aspect_quality"), r.get("aspect_ambiance"), r.get("aspect_experience"),
                    user_id
                ))
                vp_id = vp_row[0]

            # 3. Import the Raw Reviews from the CSV into the Database
            # We wipe old reviews so we don't duplicate them if you sync twice
            cur.execute("DELETE FROM reviews WHERE vendor_id = %s", (vp_id,))
            
            # If we have the raw texts saved in cache, insert them
            if vendor_name in cached_reviews and "raw_texts" in cached_reviews[vendor_name]:
                for review_text in cached_reviews[vendor_name]["raw_texts"]:
                    cur.execute("""
                        INSERT INTO reviews (vendor_id, review_text, overall_rating)
                        VALUES (%s, %s, %s)
                    """, (vp_id, review_text[:5000], 5)) # Default to 5 stars for legacy imported reviews

            n += 1
            
        conn.commit()
        cur.close()
        conn.close()
        return run_id, n, None
    except Exception as exc:
        return None, 0, str(exc)

def _build_profile_rows(source_tab, vendor_dict, aspect_profiles,
                        elo_map=None, category_map=None, price_map=None,
                        alloc_map=None, total_budget=None):
    """
    Build a list of row dicts ready for _db_insert_vendor_profiles.

    vendor_dict   : {vendor: {raw_score, star_rating, review_count}}
    aspect_profiles : {vendor: {aspect_name: {score: float, ...}}}
    elo_map       : {vendor: elo_rating float} or None
    category_map  : {vendor: str} or None
    price_map     : {vendor: float} or None
    alloc_map     : {vendor: allocated_budget float} or None (Budget Planner)
    total_budget  : float or None (Budget Planner)
    """
    rows = []
    for vn, vdata in vendor_dict.items():
        ap = aspect_profiles.get(vn, {})
        rows.append({
            "source_tab":        source_tab,
            "vendor":            vn,
            "category":          (category_map or {}).get(vn, ""),
            "price":             (price_map or {}).get(vn),
            "raw_score":         vdata["raw_score"],
            "star_rating":       vdata["star_rating"],
            "elo_rating":        (elo_map or {}).get(vn),
            "review_count":      vdata["review_count"],
            "allocated_budget":  (alloc_map or {}).get(vn),
            "total_budget":      total_budget,
            "aspect_service":    _aspect_val(ap, "service"),
            "aspect_value":      _aspect_val(ap, "value"),
            "aspect_reliability":_aspect_val(ap, "reliability"),
            "aspect_quality":    _aspect_val(ap, "quality"),
            "aspect_ambiance":   _aspect_val(ap, "ambiance"),
            "aspect_experience": _aspect_val(ap, "experience"),
        })
    return rows


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

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "Single Review", "Single Vendor", "Recommendation",
    "Compare Vendors", "Model Evaluation", "Budget Planner",
    "Event Matcher", "How It Works"
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

            # Persist to DB: Elo rating + raw score + all six aspects
            if not df_rank.empty:
                _elo_map  = dict(zip(df_rank["vendor"], df_rank["elo_rating"]))
                _star_map = dict(zip(df_rank["vendor"], df_rank["star_rating"]))
                _t3_vdict = {
                    vn: {
                        "raw_score":    p["raw_score"],
                        "star_rating":  _star_map.get(vn, 3.0),
                        "review_count": p["review_count"],
                    }
                    for vn, p in profs.items()
                }
                _t3_rows = _build_profile_rows(
                    source_tab      = "elo_recommendation",
                    vendor_dict     = _t3_vdict,
                    aspect_profiles = {vn: p["aspect_profile"] for vn, p in profs.items()},
                    elo_map         = _elo_map,
                )
                _run_id, _n, _db_err = _db_insert_vendor_profiles(_t3_rows)
                if _db_err:
                    warn(f"DB write skipped: {_db_err}")
                else:
                    ok(f"Saved {_n} vendor record(s) to database (run {_run_id[:8]}...).")

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

            # Persist to DB
            _t4_rows = _build_profile_rows(
                source_tab      = "compare_vendors",
                vendor_dict     = rows_c,
                aspect_profiles = {vn: v["aspect_profile"] for vn, v in rows_c.items()},
            )
            _run_id, _n, _db_err = _db_insert_vendor_profiles(_t4_rows)
            if _db_err:
                warn(f"DB write skipped: {_db_err}")
            else:
                ok(f"Saved {_n} vendor record(s) to database (run {_run_id[:8]}...).")

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

        sec("Per-Vendor Low-Confidence Rate")
        if ev["vendor_low_pct"]:
            df_vlcp = pd.DataFrame([
                {"vendor": vn, "low_conf_pct": pct, "n_predictions": len(ev["vendor_confs"][vn])}
                for vn, pct in sorted(ev["vendor_low_pct"].items(), key=lambda x: -x[1])
            ])
            st.dataframe(df_vlcp, use_container_width=True, hide_index=True)

        sec("Confidence Distribution")
        c1, c2 = st.columns(2)
        with c1:
            st.pyplot(fig_confidence_hist(ev["all_confs"], ev["thresh"]))
        with c2:
            st.pyplot(fig_sentiment_dist(ev["all_sents"]))

        sec("Aspect Extraction")
        st.pyplot(fig_aspects_per_review(ev["all_asp_counts"]))

        if len(ev["vendor_confs"]) > 1:
            sec("Per-Vendor Confidence Spread")
            st.pyplot(fig_vendor_conf_box(ev["vendor_confs"], ev["thresh"]))


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
    absa_score:     float      
    review_count:   int = 0
    aspect_profile: dict = field(default_factory=dict)
    city:           str = ""   
    phone:          str = ""   
    tags:           str = ""   
    description:    str = ""   
    
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
        help="Loads a built-in set of 23 vendors across 5 categories so you can explore the planner without uploading files."
    )

    bp_vendors_by_cat = {}

    if use_sample:
        bp_vendors_by_cat = _build_sample_vendors()
        ok(f"Pre-loaded {sum(len(v) for v in bp_vendors_by_cat.values())} vendors across {len(bp_vendors_by_cat)} categories.")
    else:
        info(
            "Upload multiple individual files OR a single combined CSV. "
            "A combined CSV must have 'vendor' and 'text' columns, and can optionally include 'contact' and 'location' to auto-fill details."
        )

        upload_mode_bp = st.radio("Upload mode", 
            ["Multiple files (one per vendor)", "Single combined CSV (Auto-fill metadata)"], 
            key="bp_mode_upload", horizontal=True)

        cached = st.session_state.get("bp_scored_cache", {})
        
        if upload_mode_bp == "Multiple files (one per vendor)":
            bp_files = st.file_uploader("Upload vendor review files", type=["csv", "txt"], accept_multiple_files=True, key="bp_upload_multi")
            if bp_files and models_ready():
                ate_mdl, ate_tok, sent_mdl, sent_tok = get_models()
                for f in bp_files:
                    vname = os.path.splitext(f.name)[0]
                    if vname in cached: continue
                    revs = load_reviews_from_file(f)
                    if not revs: continue
                    flat = []
                    prog_tmp = st.progress(0, text=f"Scoring {vname}...")
                    for idx, rev in enumerate(revs):
                        flat.extend(_run(rev, ate_mdl, ate_tok, sent_mdl, sent_tok))
                        prog_tmp.progress((idx + 1) / len(revs), text=f"Scoring {vname}...")
                    prog_tmp.empty()
                    prof = compute_vendor_profile(flat, len(revs), use_norm=use_norm)
                    if prof:
                        cached[vname] = {
                            "raw_score": prof["raw_score"],
                            "review_count": prof["review_count"],
                            "aspect_profile": prof["aspect_profile"],
                            "raw_texts": revs,
                        }
                st.session_state["bp_scored_cache"] = cached
                
        else:
            info("Upload your Reviews CSV(s). You can also upload an optional Vendor Data CSV to automatically fill in contact details, locations, categories, prices, and descriptions!")
            
            c1, c2 = st.columns(2)
            with c1:
                bp_review_csvs = st.file_uploader("1. Reviews CSV(s) (Required)", type=["csv"], accept_multiple_files=True, key="bp_upload_single")
            with c2:
                bp_data_csv = st.file_uploader("2. Vendor Data CSV (Optional Auto-fill)", type=["csv"], key="bp_upload_data")

            cached = st.session_state.get("bp_scored_cache", {})

            if bp_review_csvs and models_ready():
                import random
                import re
                
                # --- TAG DICTIONARY ---
                CATEGORY_TAGS = {
                    "catering": ["Veg", "Non-Veg", "Live Counters", "Multi-Cuisine", "Dessert Bar", "Buffet", "Jain Food", "Welcome Drinks"],
                    "venue": ["AC Hall", "Open Lawn", "Ample Parking", "Rooms Available", "In-house Catering", "Valet Parking", "Poolside"],
                    "decoration": ["Floral", "Theme Based", "Lighting", "Mandap", "Stage Setup", "Drapery", "Props", "Balloon Decor"],
                    "photography": ["Candid", "Drone", "Traditional", "Pre-wedding", "Cinematography", "Photo Booth", "Albums", "Same Day Edit"],
                    "entertainment": ["DJ", "Live Band", "Emcee", "Folk Dance", "Magic Show", "Celebrity Entry", "Sound System", "Interactive Games"],
                    "other": ["Customizable", "Premium", "Budget-Friendly", "Experienced", "Professional"]
                }
                
                # --- NEW: Robust Fuzzy Column Hunter ---
                def get_fuzzy_value(row, keywords):
                    # Looks for the keyword ANYWHERE inside the column name
                    for col in row.index:
                        if any(k in str(col).lower() for k in keywords):
                            val = row[col]
                            if pd.notna(val) and str(val).strip() != "":
                                return str(val).strip()
                    return ""

                # --- NEW: Robust Price Extractor ---
                def extract_price(val_str):
                    if not val_str: return 0.0
                    # Finds the first actual formatted number (e.g., "Rs 1,500 - 2000" -> "1,500")
                    matches = re.findall(r'\d+(?:,\d+)*(?:\.\d+)?', str(val_str))
                    if matches:
                        try: return float(matches[0].replace(',', ''))
                        except: return 0.0
                    return 0.0

                # Safely read and combine ALL uploaded review files on the fly
                dfs = []
                for f in bp_review_csvs:
                    temp_df = pd.read_csv(f)
                    temp_df.columns = temp_df.columns.astype(str).str.strip().str.lower()
                    temp_df.rename(columns={"vendor_name": "vendor", "review_text": "text", "contact_info": "contact"}, inplace=True)
                    if "vendor" in temp_df.columns and "text" in temp_df.columns:
                        dfs.append(temp_df)
                
                if dfs:
                    df = pd.concat(dfs, ignore_index=True)
                    
                    # --- NEW: OVERLOOK EXISTING VENDORS ---
                    try:
                        import requests
                        # 1. Ask the live database for all currently saved vendors
                        res = requests.get("http://127.0.0.1:5000/api/vendors")
                        if res.status_code == 200:
                            # 2. Extract their names and make them lowercase for safe matching
                            existing_names = [v["business_name"].lower().strip() for v in res.json()]
                            
                            # 3. Filter the CSV data: Keep ONLY rows where the vendor is NOT in the database
                            df["temp_match_name"] = df["vendor"].astype(str).str.strip().str.lower()
                            df = df[~df["temp_match_name"].isin(existing_names)]
                            df.drop(columns=["temp_match_name"], inplace=True)
                            
                            if df.empty:
                                st.success("All vendors in these files are already in the database. Nothing new to process!")
                    except Exception as e:
                        st.warning("Could not check existing vendors against the database.")
                else:
                    df = pd.DataFrame(columns=["vendor", "text"])

                # --- SUPERCHARGED: Process SCATTERED Master Vendor Data ---
                vendor_metadata = {}
                if bp_data_csv:
                    df_data = pd.read_csv(bp_data_csv)
                    df_data.columns = df_data.columns.astype(str).str.strip().str.lower()
                    
                    # Aggressively hunt for the vendor name column
                    vendor_col = None
                    for possible in ["vendor_name", "vendor", "name", "business_name", "title", "company"]:
                        if possible in df_data.columns:
                            vendor_col = possible
                            break
                    
                    if vendor_col:
                        df_data.rename(columns={vendor_col: "vendor"}, inplace=True)
                        df_data["vendor"] = df_data["vendor"].astype(str).str.strip().str.lower()
                        
                        for _, row in df_data.iterrows():
                            v_key = row["vendor"]
                            
                            # Fuzzy hunting for data across scattered columns
                            raw_cat = get_fuzzy_value(row, ["category", "type", "service"])
                            raw_price = get_fuzzy_value(row, ["price", "pricing", "cost", "budget", "plate"])
                            raw_desc = get_fuzzy_value(row, ["description", "about", "details", "info", "bio"])
                            
                            # Smart Category Mapping
                            clean_cat = raw_cat.lower()
                            if any(x in clean_cat for x in ["food", "cater", "menu", "dining"]): clean_cat = "catering"
                            elif any(x in clean_cat for x in ["decor", "flower", "theme"]): clean_cat = "decoration"
                            elif any(x in clean_cat for x in ["photo", "video", "shoot", "camera"]): clean_cat = "photography"
                            elif any(x in clean_cat for x in ["music", "dj", "entertain", "band"]): clean_cat = "entertainment"
                            elif any(x in clean_cat for x in ["venue", "hall", "lawn", "hotel", "resort"]): clean_cat = "venue"
                            
                            vendor_metadata[v_key] = {
                                "contact": get_fuzzy_value(row, ["contact", "phone", "mobile", "number"]),
                                "location": get_fuzzy_value(row, ["location", "city", "address", "area", "region"]),
                                "category": clean_cat,
                                "price": extract_price(raw_price),
                                "description": raw_desc
                            }
                # ----------------------------------------

                if {"vendor", "text"}.issubset(df.columns):
                    df["vendor"] = df["vendor"].astype(str).str.strip()
                    df["text"] = df["text"].astype(str).str.strip()
                    df = df[df["text"].str.len() > 0]
                    
                    ate_mdl, ate_tok, sent_mdl, sent_tok = get_models()
                    
                    # Group by vendor
                    vendors_grouped = list(df.groupby("vendor", sort=False))
                    for i, (vn, grp) in enumerate(vendors_grouped):
                        if vn in cached: continue
                        revs = grp["text"].tolist()
                        
                        # NORMALIZE KEY TO MATCH MASTER DATA EXACTLY
                        clean_vn = str(vn).strip().lower()
                        
                        # Extract metadata: Check the Data CSV first
                        meta = vendor_metadata.get(clean_vn, {})
                        phone = meta.get("contact", "")
                        city = meta.get("location", "")
                        auto_cat = meta.get("category", "")
                        auto_price = meta.get("price", 0.0)
                        auto_desc = meta.get("description", "")
                        
                        # --- Fallback Hunter: Check the Reviews CSV if missing ---
                        row = grp.iloc[0] 
                        if not phone: phone = get_fuzzy_value(row, ["contact", "phone", "mobile", "number"])
                        if not city: city = get_fuzzy_value(row, ["location", "city", "address", "area", "region"])
                        if not auto_desc: auto_desc = get_fuzzy_value(row, ["description", "about", "details", "info", "bio"])
                        
                        if not auto_cat:
                            raw_cat = get_fuzzy_value(row, ["category", "type", "service"])
                            if raw_cat:
                                clean_cat = raw_cat.lower()
                                if any(x in clean_cat for x in ["food", "cater", "menu", "dining"]): auto_cat = "catering"
                                elif any(x in clean_cat for x in ["decor", "flower", "theme"]): auto_cat = "decoration"
                                elif any(x in clean_cat for x in ["photo", "video", "shoot", "camera"]): auto_cat = "photography"
                                elif any(x in clean_cat for x in ["music", "dj", "entertain", "band"]): auto_cat = "entertainment"
                                elif any(x in clean_cat for x in ["venue", "hall", "lawn", "hotel", "resort"]): auto_cat = "venue"

                        if auto_price == 0.0:
                            raw_price = get_fuzzy_value(row, ["price", "pricing", "cost", "budget", "plate"])
                            auto_price = extract_price(raw_price)
                                    
                        if str(phone).endswith(".0"): phone = str(phone)[:-2] # Fix pandas float formatting
                        
                        # Generate Random Tags based on Category
                        tag_pool = CATEGORY_TAGS.get(auto_cat, CATEGORY_TAGS["other"])
                        auto_tags = ", ".join(random.sample(tag_pool, min(random.randint(3, 5), len(tag_pool))))
                        
                        flat = []
                        prog_tmp = st.progress(0, text=f"Scoring {vn[:25]} ({i+1}/{len(vendors_grouped)})...")
                        for idx, rev in enumerate(revs):
                            flat.extend(_run(rev, ate_mdl, ate_tok, sent_mdl, sent_tok))
                            prog_tmp.progress((idx + 1) / len(revs), text=f"Scoring {vn[:25]} ({i+1}/{len(vendors_grouped)})...")
                        prog_tmp.empty()
                        
                        prof = compute_vendor_profile(flat, len(revs), use_norm=use_norm)
                        if prof:
                            cached[vn] = {
                                "raw_score": prof["raw_score"],
                                "review_count": prof["review_count"],
                                "aspect_profile": prof["aspect_profile"],
                                "raw_texts": revs,
                                "auto_phone": phone,
                                "auto_city": city,
                                "auto_cat": auto_cat,
                                "auto_price": auto_price,
                                "auto_desc": auto_desc,
                                "auto_tags": auto_tags
                            }
                    st.session_state["bp_scored_cache"] = cached
                else:
                    err("Reviews CSV must contain at least a vendor name and review text column.")

            if cached:
                ok(f"Scored {len(cached)} vendor(s). Assign a category and price to each.")

                st.markdown(
                    '<div style="font-family:IBM Plex Mono,monospace;font-size:0.68rem;'
                    'text-transform:uppercase;letter-spacing:0.1em;color:#3a3a3a;'
                    'margin-bottom:0.6rem;">Vendor Configuration</div>',
                    unsafe_allow_html=True,
                )

                # Build the UI Expanders with auto-filled metadata
                for vn, prof_data in cached.items():
                    with st.expander(f"{vn} (ABSA Score: {prof_data['raw_score']:+.3f})", expanded=True):
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            st.text_input("Category", value=prof_data.get("auto_cat", ""), key=f"bp_cat_{vn}", placeholder="e.g. venue")
                            st.number_input("Price (₹)", value=float(prof_data.get("auto_price", 0.0)), min_value=0.0, step=1000.0, key=f"bp_price_{vn}")
                        with c2:
                            st.text_input("City", value=prof_data.get("auto_city", ""), key=f"bp_city_{vn}", placeholder="e.g. Pune")
                            st.text_input("Phone", value=prof_data.get("auto_phone", ""), key=f"bp_phone_{vn}", placeholder="+91...")
                        with c3:
                            st.text_input("Tags (comma separated)", value=prof_data.get("auto_tags", ""), key=f"bp_tags_{vn}", placeholder="Veg, Non-Veg, AC")
                        
                        st.text_area("Description", value=prof_data.get("auto_desc", ""), key=f"bp_desc_{vn}", placeholder="Describe this vendor...", height=68)

                # Build bp_vendors_by_cat from the config inputs
                for vn, prof_data in cached.items():
                    cat   = st.session_state.get(f"bp_cat_{vn}", "").strip().lower()
                    price = float(st.session_state.get(f"bp_price_{vn}", 0) or 0)
                    city  = st.session_state.get(f"bp_city_{vn}", "").strip()
                    phone = st.session_state.get(f"bp_phone_{vn}", "").strip()
                    tags  = st.session_state.get(f"bp_tags_{vn}", "").strip()
                    desc  = st.session_state.get(f"bp_desc_{vn}", "").strip()
                    
                    if cat and price > 0:
                        bp_vendors_by_cat.setdefault(cat, []).append(
                            _Vendor(
                                name           = vn,
                                category       = cat,
                                price          = price,
                                city           = city,
                                phone          = phone,
                                tags           = tags,
                                description    = desc,
                                absa_score     = prof_data["raw_score"],
                                review_count   = prof_data["review_count"],
                                aspect_profile = prof_data["aspect_profile"],
                            )
                        )

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

                        # Persist to DB: selected vendors with their category, price, alloc
                        # Persist ALL uploaded vendors to Eventify Live Database
                        _bp_rows = []
                        
                        for cat, vlist in bp_vendors_by_cat.items():
                            for v in vlist:
                                ap = v.aspect_profile
                                _bp_rows.append({
                                    "vendor": v.name,
                                    "category": v.category,
                                    "price": v.price,
                                    "city": v.city,
                                    "phone": v.phone,
                                    "tags": v.tags,
                                    "raw_score": v.absa_score,
                                    "star_rating": v.star_rating,
                                    "review_count": v.review_count,
                                    "aspect_service": float(ap.get("service", {}).get("score", 0)) if "service" in ap else None,
                                    "aspect_value": float(ap.get("value", {}).get("score", 0)) if "value" in ap else None,
                                    "aspect_reliability": float(ap.get("reliability", {}).get("score", 0)) if "reliability" in ap else None,
                                    "aspect_quality": float(ap.get("quality", {}).get("score", 0)) if "quality" in ap else None,
                                    "aspect_ambiance": float(ap.get("ambiance", {}).get("score", 0)) if "ambiance" in ap else None,
                                    "aspect_experience": float(ap.get("experience", {}).get("score", 0)) if "experience" in ap else None,
                                })
                        
                        _run_id, _n, _db_err = _db_insert_vendor_profiles(_bp_rows)
                        if _db_err:
                            warn(f"Eventify DB sync failed: {_db_err}")
                        else:
                            ok(f"Successfully synced ALL {_n} uploaded vendors to the live Eventify database!")

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

                    # Persist ALL uploaded vendors to Eventify Live Database
                    _bp_rows = []
                    
                    for cat, vlist in bp_vendors_by_cat.items():
                        for v in vlist:
                            ap = v.aspect_profile
                            _bp_rows.append({
                                "vendor": v.name,
                                "category": v.category,
                                "price": v.price,
                                "city": v.city,
                                "phone": v.phone,
                                "tags": v.tags,
                                "raw_score": v.absa_score,
                                "star_rating": v.star_rating,
                                "review_count": v.review_count,
                                "aspect_service": float(ap.get("service", {}).get("score", 0)) if "service" in ap else None,
                                "aspect_value": float(ap.get("value", {}).get("score", 0)) if "value" in ap else None,
                                "aspect_reliability": float(ap.get("reliability", {}).get("score", 0)) if "reliability" in ap else None,
                                "aspect_quality": float(ap.get("quality", {}).get("score", 0)) if "quality" in ap else None,
                                "aspect_ambiance": float(ap.get("ambiance", {}).get("score", 0)) if "ambiance" in ap else None,
                                "aspect_experience": float(ap.get("experience", {}).get("score", 0)) if "experience" in ap else None,
                            })
                    
                    _run_id, _n, _db_err = _db_insert_vendor_profiles(_bp_rows)
                    if _db_err:
                        warn(f"Eventify DB sync failed: {_db_err}")
                    else:
                        ok(f"Successfully synced ALL {_n} uploaded vendors to the live Eventify database!")
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
# Tab 7: Event Matcher
# =============================================================================

with tab7:
    absa_event_matcher.render(
        sec_fn    = sec,
        info_fn   = info,
        warn_fn   = warn,
        ok_fn     = ok,
        err_fn    = err,
        stat_row_fn = stat_row,
        stars_fn  = stars_display,
    )


# =============================================================================
# Tab 8: How It Works
# =============================================================================

with tab8:
    absa_how_it_works.render(sec)