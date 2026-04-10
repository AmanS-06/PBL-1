"""
absa_event_matcher.py
Smart Event Matcher tab for the ABSA Streamlit application.

Provides:
  - Event-type context scoring (aspect profile weighted by event template)
  - Elo-integrated budget-aware vendor selection
  - Keyword / must-have aspect filtering
  - Per-vendor customisation (exclude, pin, min-star gate)
  - Full visual suite: radar, heatmap, scatter, Elo bar, match gauge

Import and call render(sec, info, warn, ok, err, stat_row, stars_display, _Vendor,
                       SentimentEloRanker, _priority_weights, _softmax,
                       bp_vendors_by_cat_or_sample) from the main app.
"""

from __future__ import annotations

import itertools
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import streamlit as st
import os
from scipy.special import softmax as _softmax

# ---------------------------------------------------------------------------
# Event profile templates
# Each value: {aspect_key: importance_weight}  (will be L1-normalised)
# ---------------------------------------------------------------------------

EVENT_TEMPLATES: Dict[str, Dict[str, float]] = {
    "Wedding": {
        "quality":     0.25,
        "ambiance":    0.22,
        "reliability": 0.20,
        "service":     0.18,
        "experience":  0.10,
        "value":       0.05,
    },
    "Corporate Conference": {
        "reliability": 0.28,
        "service":     0.25,
        "value":       0.20,
        "ambiance":    0.15,
        "quality":     0.08,
        "experience":  0.04,
    },
    "Birthday Party": {
        "quality":     0.28,
        "ambiance":    0.20,
        "value":       0.20,
        "service":     0.17,
        "experience":  0.10,
        "reliability": 0.05,
    },
    "Concert / Live Event": {
        "quality":     0.30,
        "reliability": 0.25,
        "service":     0.20,
        "ambiance":    0.15,
        "experience":  0.07,
        "value":       0.03,
    },
    "Product Launch": {
        "reliability": 0.30,
        "quality":     0.25,
        "service":     0.20,
        "ambiance":    0.12,
        "value":       0.08,
        "experience":  0.05,
    },
    "Gala Dinner": {
        "ambiance":    0.28,
        "quality":     0.25,
        "service":     0.22,
        "experience":  0.15,
        "reliability": 0.06,
        "value":       0.04,
    },
    "Trade Exhibition": {
        "reliability": 0.28,
        "value":       0.22,
        "service":     0.20,
        "ambiance":    0.15,
        "quality":     0.10,
        "experience":  0.05,
    },
    "Custom": {},   # user-defined via sliders
}

# Normalise all templates
for _ek, _ev in EVENT_TEMPLATES.items():
    _s = sum(_ev.values()) or 1.0
    EVENT_TEMPLATES[_ek] = {k: v / _s for k, v in _ev.items()}

ALL_ASPECTS = ["quality", "service", "value", "reliability", "ambiance", "experience"]

# Local keyword -> canonical aspect map (subset of main app ASPECT_NORM_MAP)
# Avoids circular import while preserving synonym resolution.
_KEYWORD_NORM: Dict[str, str] = {
    # quality
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
    # service
    "service": "service", "staff": "service", "waiter": "service",
    "waitress": "service", "server": "service", "host": "service",
    "bartender": "service", "chef": "service", "cook": "service",
    "photographer": "service", "videographer": "service",
    "dj": "service", "emcee": "service", "mc": "service",
    "crew": "service", "team": "service", "worker": "service",
    "planner": "service", "coordinator": "service", "organiser": "service",
    "organizer": "service", "florist": "service", "baker": "service",
    "caterer": "service", "driver": "service", "chauffeur": "service",
    "support": "service", "communication": "service", "responsiveness": "service",
    "professionalism": "service", "attitude": "service", "helpfulness": "service",
    "behaviour": "service", "behavior": "service", "manner": "service",
    "courtesy": "service", "friendliness": "service", "politeness": "service",
    # value
    "price": "value", "prices": "value", "cost": "value", "value": "value",
    "pricing": "value", "fee": "value", "fees": "value", "charge": "value",
    "charges": "value", "rate": "value", "rates": "value", "quote": "value",
    "package": "value", "deal": "value", "expensive": "value",
    "cheap": "value", "affordable": "value", "budget": "value",
    # reliability
    "reliability": "reliability", "punctuality": "reliability", "timing": "reliability",
    "delivery": "reliability", "wait": "reliability", "waiting": "reliability",
    "delay": "reliability", "delays": "reliability", "time": "reliability",
    "speed": "reliability", "turnaround": "reliability", "schedule": "reliability",
    "deadline": "reliability", "promptness": "reliability",
    # ambiance
    "ambiance": "ambiance", "ambience": "ambiance", "atmosphere": "ambiance",
    "venue": "ambiance", "location": "ambiance", "place": "ambiance",
    "space": "ambiance", "cleanliness": "ambiance", "noise": "ambiance",
    "environment": "ambiance", "setting": "ambiance", "hall": "ambiance",
    "room": "ambiance", "area": "ambiance", "facility": "ambiance",
    "parking": "ambiance", "access": "ambiance", "accessibility": "ambiance",
    # experience
    "experience": "experience", "overall": "experience", "visit": "experience",
    "recommendation": "experience", "event": "experience", "occasion": "experience",
    "celebration": "experience", "party": "experience", "wedding": "experience",
    "function": "experience",
}

# ---------------------------------------------------------------------------
# Demo vendor data (mirrors Budget Planner sample set + richer profiles)
# ---------------------------------------------------------------------------

_DEMO_VENDORS = [
    # (name, category, price, absa_score, reviews, aspect_profile_overrides)
    # aspect_profile format: {aspect: score}  (only overrides; rest inferred)
    ("Grand Ballroom",        "venue",         120_000,  0.78, 42,
     {"ambiance": 0.91, "reliability": 0.72, "service": 0.80, "value": 0.40,
      "quality": 0.82, "experience": 0.85}),
    ("The Garden Estate",     "venue",          85_000,  0.61, 35,
     {"ambiance": 0.75, "reliability": 0.65, "service": 0.60, "value": 0.55,
      "quality": 0.65, "experience": 0.68}),
    ("City View Hall",        "venue",          60_000,  0.44, 28,
     {"ambiance": 0.55, "reliability": 0.50, "service": 0.45, "value": 0.70,
      "quality": 0.48, "experience": 0.50}),
    ("Heritage House",        "venue",          45_000,  0.30, 19,
     {"ambiance": 0.60, "reliability": 0.35, "service": 0.30, "value": 0.75,
      "quality": 0.38, "experience": 0.42}),
    ("Budget Banquet Hall",   "venue",          22_000,  0.05, 11,
     {"ambiance": 0.15, "reliability": 0.20, "service": 0.10, "value": 0.85,
      "quality": 0.12, "experience": 0.20}),

    ("Elite Cuisine Co.",     "catering",       95_000,  0.82, 58,
     {"quality": 0.92, "service": 0.85, "value": 0.50, "reliability": 0.80,
      "ambiance": 0.60, "experience": 0.88}),
    ("Spice Garden Caterers", "catering",       60_000,  0.55, 44,
     {"quality": 0.65, "service": 0.60, "value": 0.65, "reliability": 0.55,
      "ambiance": 0.40, "experience": 0.60}),
    ("Homestyle Feast",       "catering",       40_000,  0.41, 31,
     {"quality": 0.55, "service": 0.45, "value": 0.70, "reliability": 0.40,
      "ambiance": 0.30, "experience": 0.45}),
    ("QuickBite Catering",    "catering",       25_000,  0.08, 17,
     {"quality": 0.18, "service": 0.15, "value": 0.80, "reliability": 0.12,
      "ambiance": 0.10, "experience": 0.18}),
    ("Budget Bites",          "catering",       15_000, -0.20,  9,
     {"quality": -0.25, "service": -0.15, "value": 0.60, "reliability": -0.20,
      "ambiance": -0.10, "experience": -0.18}),

    ("Pixel Perfect Studios", "photography",    55_000,  0.88, 63,
     {"quality": 0.95, "service": 0.85, "value": 0.45, "reliability": 0.88,
      "ambiance": 0.70, "experience": 0.92}),
    ("Capture Moments",       "photography",    38_000,  0.67, 47,
     {"quality": 0.75, "service": 0.70, "value": 0.60, "reliability": 0.65,
      "ambiance": 0.50, "experience": 0.72}),
    ("Frame & Flash",         "photography",    25_000,  0.45, 29,
     {"quality": 0.55, "service": 0.50, "value": 0.68, "reliability": 0.42,
      "ambiance": 0.35, "experience": 0.48}),
    ("Snapshot Pro",          "photography",    15_000,  0.22, 18,
     {"quality": 0.30, "service": 0.25, "value": 0.75, "reliability": 0.20,
      "ambiance": 0.20, "experience": 0.28}),
    ("Economy Clicks",        "photography",     8_000, -0.05,  7,
     {"quality": -0.08, "service": -0.02, "value": 0.70, "reliability": -0.05,
      "ambiance": 0.05, "experience": 0.00}),

    ("Floral Fantasy",        "decoration",     45_000,  0.75, 38,
     {"quality": 0.85, "service": 0.72, "value": 0.40, "reliability": 0.70,
      "ambiance": 0.90, "experience": 0.80}),
    ("Bloom & Decor",         "decoration",     30_000,  0.58, 26,
     {"quality": 0.65, "service": 0.60, "value": 0.58, "reliability": 0.55,
      "ambiance": 0.72, "experience": 0.62}),
    ("Simple Touches",        "decoration",     18_000,  0.33, 14,
     {"quality": 0.40, "service": 0.35, "value": 0.70, "reliability": 0.30,
      "ambiance": 0.45, "experience": 0.38}),
    ("DIY Decor Supplies",    "decoration",      8_000, -0.10,  6,
     {"quality": -0.12, "service": -0.08, "value": 0.78, "reliability": -0.10,
      "ambiance": 0.05, "experience": -0.05}),

    ("Harmony Live Band",     "entertainment",  50_000,  0.80, 41,
     {"quality": 0.88, "service": 0.78, "value": 0.38, "reliability": 0.75,
      "ambiance": 0.82, "experience": 0.90}),
    ("DJ Maestro",            "entertainment",  25_000,  0.60, 33,
     {"quality": 0.68, "service": 0.62, "value": 0.62, "reliability": 0.58,
      "ambiance": 0.65, "experience": 0.70}),
    ("Acoustic Duo",          "entertainment",  15_000,  0.38, 22,
     {"quality": 0.45, "service": 0.40, "value": 0.72, "reliability": 0.35,
      "ambiance": 0.50, "experience": 0.42}),
    ("Karaoke Setup Rental",  "entertainment",   8_000,  0.10,  8,
     {"quality": 0.12, "service": 0.08, "value": 0.80, "reliability": 0.10,
      "ambiance": 0.20, "experience": 0.15}),
]


@dataclass
class RichVendor:
    """Extended vendor with per-aspect scores and Elo rating."""
    name:           str
    category:       str
    price:          float
    absa_score:     float
    review_count:   int
    aspect_scores:  Dict[str, float] = field(default_factory=dict)
    elo_rating:     float = 1500.0
    elo_star:       float = 3.0

    @property
    def norm_score(self) -> float:
        return float(np.clip((self.absa_score + 1) / 2, 0.0, 1.0))

    @property
    def star_rating(self) -> float:
        return float(np.clip((self.absa_score + 1) / 2 * 5, 0.0, 5.0))

    def context_score(self, weights: Dict[str, float]) -> float:
        """Dot product of aspect scores with event-template weights."""
        asp = self.aspect_scores
        if not asp:
            return self.norm_score
        total_w = sum(weights.get(a, 0.0) for a in asp)
        if total_w == 0:
            return self.norm_score
        score = sum(asp.get(a, 0.0) * w for a, w in weights.items() if a in asp)
        # Normalise to [-1, +1] -> [0, 1]
        return float(np.clip((score + 1) / 2, 0.0, 1.0))

    def keyword_match_score(self, keywords: List[str]) -> float:
        """Fraction of keyword aspects with positive sentiment (score > 0)."""
        if not keywords or not self.aspect_scores:
            return 1.0
        hits = sum(1 for k in keywords if self.aspect_scores.get(k, 0.0) > 0)
        return hits / len(keywords)


def _build_demo_vendors() -> Dict[str, List[RichVendor]]:
    vby: Dict[str, List[RichVendor]] = defaultdict(list)
    for (name, cat, price, score, rc, asp) in _DEMO_VENDORS:
        vby[cat].append(RichVendor(
            name=name, category=cat, price=price,
            absa_score=score, review_count=rc, aspect_scores=asp,
        ))
    return dict(vby)


# ---------------------------------------------------------------------------
# Elo ranker operating on RichVendor (per-category)
# ---------------------------------------------------------------------------

def _run_elo_within_category(vendors: List[RichVendor],
                              context_weights: Dict[str, float],
                              k: int = 32,
                              bayes_m: int = 8,
                              rounds: int = 5) -> List[RichVendor]:
    """Compute Elo ratings within a single category using context scores."""
    if not vendors:
        return vendors

    # Bayesian-smoothed context scores
    raw = {v.name: v.context_score(context_weights) for v in vendors}
    gm  = np.mean(list(raw.values()))
    sm  = {v.name: (v.review_count * raw[v.name] + bayes_m * gm) / (v.review_count + bayes_m)
           for v in vendors}

    ratings = {v.name: 1500.0 for v in vendors}
    names   = [v.name for v in vendors]

    for _ in range(rounds):
        for i in range(len(names)):
            for j in range(len(names)):
                if i == j: continue
                vi, vj = names[i], names[j]
                ri, rj = ratings[vi], ratings[vj]
                si, sj = sm[vi], sm[vj]
                actual   = si / (si + sj + 1e-9)
                expected = 1.0 / (1.0 + 10 ** ((rj - ri) / 400))
                ratings[vi] = ri + k * (actual - expected)

    elo_arr = np.array([ratings[v.name] for v in vendors])
    mn, mx  = elo_arr.min(), elo_arr.max()
    stars   = (np.full(len(vendors), 3.0) if mx == mn
               else 1.0 + (elo_arr - mn) / (mx - mn) * 4.0)

    for v, e, s in zip(vendors, elo_arr, stars):
        v.elo_rating = round(float(e), 1)
        v.elo_star   = round(float(np.clip(s, 1, 5)), 2)

    return vendors


# ---------------------------------------------------------------------------
# Selection engine
# ---------------------------------------------------------------------------

def _select_vendors(
    vby_cat:        Dict[str, List[RichVendor]],
    budget:         float,
    cat_budgets:    Dict[str, float],
    context_w:      Dict[str, float],
    keywords:       List[str],
    min_star:       Dict[str, float],
    excluded:       Dict[str, List[str]],
    pinned:         Dict[str, Optional[str]],
    use_elo:        bool,
) -> Tuple[Dict[str, Optional[RichVendor]], float]:
    """
    Greedy selection with Elo/context scoring, keyword filter,
    star gate, exclusions, and pins.
    """
    selection: Dict[str, Optional[RichVendor]] = {}
    total_cost = 0.0

    for cat, alloc in cat_budgets.items():
        candidates = vby_cat.get(cat, [])

        # Pin overrides everything
        pin_name = pinned.get(cat)
        if pin_name:
            pinned_v = next((v for v in candidates if v.name == pin_name), None)
            if pinned_v and pinned_v.price <= budget - total_cost + alloc:
                selection[cat] = pinned_v
                total_cost    += pinned_v.price
                continue

        # Filter affordable + not excluded + min star gate
        excl = excluded.get(cat, [])
        ms   = min_star.get(cat, 0.0)
        pool = [v for v in candidates
                if v.price <= alloc
                and v.name not in excl
                and v.star_rating >= ms]

        if not pool:
            # Relax budget constraint to global remaining budget
            remaining = budget - total_cost
            pool = [v for v in candidates
                    if v.price <= remaining
                    and v.name not in excl
                    and v.star_rating >= ms]

        if not pool:
            selection[cat] = None
            continue

        # Score: keyword match * (elo_star or context_score)
        def _score(v: RichVendor) -> float:
            km   = v.keyword_match_score(keywords)
            base = (v.elo_star / 5.0) if use_elo else v.context_score(context_w)
            return km * base

        best = max(pool, key=_score)
        selection[cat] = best
        total_cost    += best.price

    return selection, total_cost


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

_MONO = "monospace"
_BG   = "#111111"
_EDGE = "#1e1e1e"
_GRID = "#161616"
_TEXT = "#bbbbbb"
_SENT_C = {"positive": "#2ecc71", "negative": "#e74c3c",
           "neutral": "#3498db", "conflict": "#f39c12"}
_ASP_PALETTE = ["#3498db", "#2ecc71", "#f39c12", "#e74c3c", "#9b59b6", "#1abc9c"]


def _radar(vendors: List[RichVendor], aspects: List[str], title: str = ""):
    """Spider / radar chart comparing selected vendors across aspects."""
    n      = len(aspects)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(5.5, 5.5), subplot_kw=dict(polar=True))
    ax.set_facecolor(_BG)
    fig.patch.set_facecolor(_BG)
    ax.spines["polar"].set_color(_EDGE)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(aspects, fontsize=8, color=_TEXT)
    ax.set_yticklabels([]); ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.yaxis.grid(True, color=_GRID, linewidth=0.7)
    ax.xaxis.grid(True, color=_EDGE, linewidth=0.5)

    cmap = cm.get_cmap("tab10", max(len(vendors), 1))
    for i, v in enumerate(vendors):
        vals = [(v.aspect_scores.get(a, 0.0) + 1) / 2 for a in aspects]
        vals += vals[:1]
        color = cmap(i)
        ax.plot(angles, vals, color=color, linewidth=1.8, label=v.name)
        ax.fill(angles, vals, color=color, alpha=0.12)

    if title:
        ax.set_title(title, fontsize=9, pad=18, color=_TEXT)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.22),
              ncol=2, fontsize=7.5, frameon=False, labelcolor=_TEXT)
    plt.tight_layout()
    return fig


def _heatmap(vby_cat: Dict[str, List[RichVendor]],
             aspects: List[str],
             context_w: Dict[str, float]):
    """Heatmap: vendors (rows) x aspects (cols), colored by score."""
    rows, row_labels, cat_labels = [], [], []
    for cat in sorted(vby_cat.keys()):
        for v in sorted(vby_cat[cat], key=lambda x: -x.absa_score):
            row_labels.append(v.name)
            cat_labels.append(cat)
            rows.append([(v.aspect_scores.get(a, 0.0)) for a in aspects])

    if not rows:
        return None

    mat = np.array(rows, dtype=float)
    h   = max(4.0, len(rows) * 0.38)
    fig, ax = plt.subplots(figsize=(10, h))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(np.arange(len(aspects)))
    ax.set_xticklabels(aspects, fontsize=8, color=_TEXT)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=7.5, color=_TEXT)

    # Overlay event-weight importance markers on column headers
    for xi, asp in enumerate(aspects):
        w = context_w.get(asp, 0.0)
        if w >= 0.15:
            ax.annotate("*", (xi, -0.8), fontsize=10, color="#f39c12",
                        ha="center", va="center", annotation_clip=False)

    for i in range(len(row_labels)):
        for j in range(len(aspects)):
            val = mat[i, j]
            ax.text(j, i, f"{val:+.2f}", ha="center", va="center",
                    fontsize=6.5,
                    color="#000" if -0.4 < val < 0.6 else "#fff")

    plt.colorbar(im, ax=ax, shrink=0.6, pad=0.02,
                 label="Aspect Sentiment Score")
    ax.set_title("Vendor x Aspect Heatmap  (* = high event importance)",
                 fontsize=9, pad=8, color=_TEXT)
    fig.patch.set_facecolor(_BG); ax.set_facecolor(_BG)
    plt.tight_layout()
    return fig


def _elo_bar(vby_cat: Dict[str, List[RichVendor]], selected: Dict[str, Optional[RichVendor]]):
    """Horizontal Elo-star bar per category, highlighted selected vendor."""
    all_v   = [(v, cat) for cat, vlist in vby_cat.items() for v in vlist]
    all_v   = sorted(all_v, key=lambda x: (-x[0].elo_star, x[1]))
    names   = [v.name for v, _ in all_v]
    stars   = [v.elo_star for v, _ in all_v]
    cats    = [cat for _, cat in all_v]
    sel_set = {v.name for v in selected.values() if v}

    cat_list = sorted(set(cats))
    cmap     = cm.get_cmap("tab10", len(cat_list))
    cat_c    = {c: cmap(i) for i, c in enumerate(cat_list)}

    colors   = [cat_c[c] for c in cats]
    alphas   = [0.9 if n in sel_set else 0.35 for n in names]

    h = max(4.0, len(names) * 0.38)
    fig, ax = plt.subplots(figsize=(8, h))
    bars = ax.barh(range(len(names)), stars, color=colors, height=0.6,
                   alpha=1.0)
    for bar, alpha in zip(bars, alphas):
        bar.set_alpha(alpha)

    # White ring on selected
    for i, (n, s) in enumerate(zip(names, stars)):
        if n in sel_set:
            ax.barh(i, s, height=0.6, color="none",
                    edgecolor="white", linewidth=2.0)
            ax.text(s + 0.06, i, "SELECTED", va="center",
                    fontsize=6.5, color="#ffffff", fontweight="bold")

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7.5, color=_TEXT)
    ax.set_xlim(0.5, 6.5)
    ax.axvline(3.0, color="#555", linestyle="--", lw=0.8)
    ax.set_xlabel("Elo Star Rating", fontsize=8)
    ax.set_title("Elo Rankings (white ring = selected)", fontsize=9, color=_TEXT, pad=8)
    ax.spines[["top", "right"]].set_visible(False)

    handles = [mpatches.Patch(color=cat_c[c], label=c) for c in cat_list]
    ax.legend(handles=handles, loc="lower right", fontsize=7.5,
              frameon=False, labelcolor=_TEXT)

    fig.patch.set_facecolor(_BG); ax.set_facecolor(_BG)
    plt.tight_layout()
    return fig


def _context_scatter(vby_cat: Dict[str, List[RichVendor]],
                     context_w: Dict[str, float],
                     selected: Dict[str, Optional[RichVendor]]):
    """Price vs Context Score scatter with selection rings."""
    cat_list = sorted(vby_cat.keys())
    cmap     = cm.get_cmap("tab10", len(cat_list))
    cat_c    = {c: cmap(i) for i, c in enumerate(cat_list)}
    sel_set  = {v.name for v in selected.values() if v}

    fig, ax = plt.subplots(figsize=(9, 5))
    for cat in cat_list:
        vlist = vby_cat[cat]
        xs    = [v.price for v in vlist]
        ys    = [v.context_score(context_w) for v in vlist]
        ax.scatter(xs, ys, color=cat_c[cat], s=60, alpha=0.72,
                   label=cat, zorder=2)
        for v in vlist:
            ax.annotate(v.name, (v.price, v.context_score(context_w)),
                        textcoords="offset points", xytext=(5, 2),
                        fontsize=6, color="#666", clip_on=True)
        # Selection ring
        for v in vlist:
            if v.name in sel_set:
                ax.scatter([v.price], [v.context_score(context_w)],
                           s=260, facecolors="none",
                           edgecolors="white", linewidths=2.2, zorder=5)

    ax.axhline(0.5, color="#2a2a2a", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Price", fontsize=9)
    ax.set_ylabel("Event Context Score (0-1)", fontsize=9)
    ax.set_title("Event-Context Score vs Price  (ring = selected)", fontsize=9, pad=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, frameon=False, labelcolor=_TEXT, loc="lower right")
    fig.patch.set_facecolor(_BG); ax.set_facecolor(_BG)
    ax.set_facecolor(_BG)
    plt.tight_layout()
    return fig


def _match_gauge_fig(score: float, label: str = "Event Match"):
    """Half-donut gauge for event match score (0-1)."""
    fig, ax = plt.subplots(figsize=(4.5, 2.8),
                           subplot_kw=dict(aspect="equal"))
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_BG)
    ax.axis("off")

    pct   = float(np.clip(score, 0, 1))
    color = ("#2ecc71" if pct >= 0.7 else
             "#f39c12" if pct >= 0.4 else "#e74c3c")

    # Background arc
    theta_bg = np.linspace(np.pi, 0, 300)
    ax.plot(np.cos(theta_bg) * 1.0, np.sin(theta_bg) * 1.0,
            color=_GRID, linewidth=14, solid_capstyle="round")

    # Filled arc
    theta_fill = np.linspace(np.pi, np.pi - pct * np.pi, 300)
    ax.plot(np.cos(theta_fill) * 1.0, np.sin(theta_fill) * 1.0,
            color=color, linewidth=14, solid_capstyle="round")

    ax.text(0, -0.05, f"{pct*100:.0f}%", ha="center", va="center",
            fontsize=22, fontweight="bold", color=color,
            fontfamily=_MONO)
    ax.text(0, -0.35, label, ha="center", va="center",
            fontsize=8, color="#555", fontfamily=_MONO)

    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-0.6, 1.15)
    plt.tight_layout()
    return fig


def _keyword_match_bar(vendors: List[RichVendor], keywords: List[str]):
    """Bar chart of keyword match scores per vendor."""
    if not vendors or not keywords:
        return None
    names  = [v.name for v in vendors]
    scores = [v.keyword_match_score(keywords) * 100 for v in vendors]
    colors = ["#2ecc71" if s >= 70 else "#f39c12" if s >= 40 else "#e74c3c"
              for s in scores]
    fig, ax = plt.subplots(figsize=(7, max(2.5, len(names) * 0.45)))
    ax.barh(names, scores, color=colors, height=0.55)
    for i, (n, s) in enumerate(zip(names, scores)):
        ax.text(s + 0.5, i, f"{s:.0f}%", va="center", fontsize=8, color=_TEXT)
    ax.set_xlim(0, 115)
    ax.set_xlabel("Keyword Match %", fontsize=8)
    ax.set_title("Keyword / Priority Aspect Match", fontsize=9, pad=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.patch.set_facecolor(_BG); ax.set_facecolor(_BG)
    plt.tight_layout()
    return fig


def _template_weight_bar(weights: Dict[str, float], event_name: str):
    """Horizontal bar showing aspect weight distribution for chosen event."""
    aspects = list(weights.keys())
    vals    = [weights[a] * 100 for a in aspects]
    colors  = _ASP_PALETTE[:len(aspects)]
    fig, ax = plt.subplots(figsize=(6, max(2.0, len(aspects) * 0.45)))
    ax.barh(aspects, vals, color=colors, height=0.55)
    for i, (a, v) in enumerate(zip(aspects, vals)):
        ax.text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=8, color=_TEXT)
    ax.set_xlim(0, max(vals) * 1.25 if vals else 50)
    ax.set_xlabel("Importance Weight (%)", fontsize=8)
    ax.set_title(f"Aspect Importance Profile: {event_name}", fontsize=9, pad=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.patch.set_facecolor(_BG); ax.set_facecolor(_BG)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Result summary HTML table
# ---------------------------------------------------------------------------

def _selection_table_html(
    selection: Dict[str, Optional[RichVendor]],
    cat_budgets: Dict[str, float],
    context_w: Dict[str, float],
    keywords: List[str],
    stars_fn,
) -> str:
    th = ("padding:0.4rem 0.75rem;text-align:left;color:#3a3a3a;font-size:0.6rem;"
          "text-transform:uppercase;letter-spacing:0.1em;border-bottom:1px solid #161616;"
          "font-family:IBM Plex Mono,monospace;")
    td = ("padding:0.38rem 0.75rem;border-bottom:1px solid #111;"
          "font-family:IBM Plex Mono,monospace;font-size:0.8rem;")

    heads = ["Category", "Budget", "Selected Vendor", "Price",
             "Elo Stars", "Context Score", "KW Match", "Budget Used"]
    hrow  = "".join(f'<th style="{th}">{h}</th>' for h in heads)
    rows  = ""

    for cat, v in selection.items():
        alloc = cat_budgets.get(cat, 0)
        if v:
            up      = int(v.price / alloc * 100) if alloc else 0
            bc      = "#2ecc71" if up <= 100 else "#e74c3c"
            cs      = v.context_score(context_w)
            cs_c    = "#2ecc71" if cs >= 0.6 else "#f39c12" if cs >= 0.4 else "#e74c3c"
            km      = v.keyword_match_score(keywords) * 100
            km_c    = "#2ecc71" if km >= 70 else "#f39c12" if km >= 40 else "#e74c3c"
            rows += f"""<tr>
              <td style="{td}color:#aaa;">{cat}</td>
              <td style="{td}color:#666;">{alloc:,.0f}</td>
              <td style="{td}color:#e8e8e8;font-weight:600;">{v.name}</td>
              <td style="{td}color:#ccc;">{v.price:,.0f}</td>
              <td style="{td}">{stars_fn(v.elo_star, "0.85rem")}</td>
              <td style="{td}color:{cs_c};">{cs:.3f}</td>
              <td style="{td}color:{km_c};">{km:.0f}%</td>
              <td style="{td}">
                <span style="color:{bc};">{up}%</span>
                <span style="display:inline-block;width:44px;height:2px;background:#161616;
                  border-radius:2px;vertical-align:middle;margin-left:4px;overflow:hidden;">
                  <span style="display:block;width:{min(up,100)}%;height:100%;
                    background:{bc};"></span></span>
              </td></tr>"""
        else:
            rows += f"""<tr>
              <td style="{td}color:#aaa;">{cat}</td>
              <td style="{td}color:#666;">{alloc:,.0f}</td>
              <td colspan="6" style="{td}color:#e74c3c;">No vendor satisfies constraints</td>
            </tr>"""

    return (f'<div class="leaderboard-wrap"><table class="lb-table">'
            f'<thead><tr style="background:#090909;">{hrow}</tr></thead>'
            f'<tbody>{rows}</tbody></table></div>')


# ---------------------------------------------------------------------------
# Main render function
# ---------------------------------------------------------------------------

def render(sec_fn, info_fn, warn_fn, ok_fn, err_fn, stat_row_fn, stars_fn):
    """
    Render the Smart Event Matcher tab.
    Call from the main app inside `with tab8:`.
    """
    sec_fn("Smart Event Matcher")
    info_fn(
        "Select your event type to apply a context-weighted scoring model. "
        "The matcher uses Elo ratings, event-profile aspect weights, "
        "budget constraints, and keyword filters to surface the best "
        "vendor combination for your specific occasion."
    )

    # ------------------------------------------------------------------
    # Step 1: Data source
    # ------------------------------------------------------------------
    sec_fn("Step 1 - Vendor Data")

    use_demo = st.toggle(
        "Use pre-loaded demo vendors (23 vendors, 5 categories)",
        value=False, key="em_use_demo",
    )

    vby_cat: Dict[str, List[RichVendor]] = {}

    if use_demo:
        vby_cat = _build_demo_vendors()
        ok_fn(
            f"Demo loaded: {sum(len(v) for v in vby_cat.values())} vendors "
            f"across {len(vby_cat)} categories."
        )
    else:
        data_source = st.radio(
            "Data source",
            [
                "Upload CSV files (one per vendor)",
                "Upload combined CSV (vendor + text columns)",
                "Enter vendors manually",
                "Inherit from Budget Planner",
            ],
            key="em_data_source",
        )

        # helpers reused across upload paths
        def _score_reviews_em(vname, reviews):
            """Score a list of review strings; returns profile dict or None."""
            import sys as _sys
            _app             = _sys.modules.get("__main__")
            _run_pipeline    = getattr(_app, "_run", None)
            _compute_profile = getattr(_app, "compute_vendor_profile", None)
            _norm_flag       = st.session_state.get("use_norm", True)
            flat = []
            ate_mdl  = st.session_state["active_ate_model"]
            ate_tok  = st.session_state["active_ate_tok"]
            sent_mdl = st.session_state["active_sent_model"]
            sent_tok = st.session_state["active_sent_tok"]
            prog = st.progress(0, text=f"Scoring {vname}...")
            for idx, rev in enumerate(reviews):
                if _run_pipeline:
                    flat.extend(_run_pipeline(rev, ate_mdl, ate_tok, sent_mdl, sent_tok))
                prog.progress((idx + 1) / len(reviews), text=f"Scoring {vname}...")
            prog.empty()
            if _compute_profile and flat:
                return _compute_profile(flat, len(reviews), use_norm=_norm_flag)
            return None

        def _vendor_config_rows(cache_dict, key_prefix):
            """Render name/category/price/score row for each cached vendor."""
            _lbl = ("font-family:IBM Plex Mono,monospace;font-size:0.68rem;"
                    "text-transform:uppercase;letter-spacing:0.1em;color:#3a3a3a;"
                    "margin-bottom:0.5rem;")
            st.markdown(f'<div style="{_lbl}">Vendor Configuration</div>',
                        unsafe_allow_html=True)
            for vn, pd_ in cache_dict.items():
                c1, c2, c3, c4 = st.columns([2.5, 2, 1.8, 1.2])
                with c1:
                    st.markdown(
                        f'<div style="font-family:IBM Plex Mono,monospace;font-size:0.82rem;'
                        f'color:#e8e8e8;padding-top:0.55rem;">{vn}</div>',
                        unsafe_allow_html=True)
                with c2:
                    st.text_input("Category",
                                  key=f"{key_prefix}_cat_{vn}",
                                  value=pd_.get("_csv_category", ""),
                                  placeholder="e.g. venue",
                                  label_visibility="collapsed")
                with c3:
                    st.number_input("Price", min_value=0,
                                    value=int(pd_.get("_csv_price", 0)),
                                    step=1000,
                                    key=f"{key_prefix}_price_{vn}",
                                    label_visibility="collapsed")
                with c4:
                    sc = pd_["raw_score"]
                    sc_col = "#2ecc71" if sc >= 0 else "#e74c3c"
                    st.markdown(
                        f'<div style="font-family:IBM Plex Mono,monospace;font-size:0.82rem;'
                        f'color:{sc_col};padding-top:0.55rem;">ABSA {sc:+.3f}</div>',
                        unsafe_allow_html=True)

        def _build_vby_from_cache(cache_dict, key_prefix):
            result: Dict[str, List[RichVendor]] = {}
            for vn, pd_ in cache_dict.items():
                cat   = st.session_state.get(f"{key_prefix}_cat_{vn}", "").strip().lower()
                price = float(st.session_state.get(f"{key_prefix}_price_{vn}", 0) or 0)
                if cat and price > 0:
                    asp = {k: v["score"]
                           for k, v in pd_.get("aspect_profile", {}).items()}
                    result.setdefault(cat, []).append(RichVendor(
                        name=vn, category=cat, price=price,
                        absa_score=pd_["raw_score"],
                        review_count=pd_["review_count"],
                        aspect_scores=asp,
                    ))
            return result

        # --------------------------------------------------------------
        # A) One file per vendor
        # --------------------------------------------------------------
        if data_source == "Upload CSV files (one per vendor)":
            info_fn(
                "Upload one CSV (with a <code>text</code> column) or TXT "
                "file (one review per line) per vendor. "
                "Models must be loaded in the sidebar. "
                "Filename becomes the vendor name."
            )
            em_files = st.file_uploader(
                "Vendor review files", type=["csv", "txt"],
                accept_multiple_files=True, key="em_upload_multi",
            )
            models_loaded = "active_ate_model" in st.session_state
            if em_files and not models_loaded:
                warn_fn("Load models from the sidebar to score uploaded files.")
            elif em_files and models_loaded:
                import io as _io
                em_cache = st.session_state.get("em_scored_cache", {})
                for f in em_files:
                    vname = os.path.splitext(f.name)[0]
                    if vname in em_cache:
                        continue
                    raw = f.read()
                    if f.name.endswith(".txt"):
                        reviews = [l.strip() for l in
                                   raw.decode("utf-8", errors="ignore").splitlines()
                                   if l.strip()]
                    else:
                        df_f = pd.read_csv(_io.BytesIO(raw))
                        if "text" not in df_f.columns:
                            err_fn(f"{f.name}: no 'text' column. Skipping.")
                            continue
                        reviews = [r for r in
                                   df_f["text"].astype(str).str.strip().tolist()
                                   if r]
                    if not reviews:
                        continue
                    prof = _score_reviews_em(vname, reviews)
                    if prof:
                        em_cache[vname] = prof
                st.session_state["em_scored_cache"] = em_cache

            em_cache = st.session_state.get("em_scored_cache", {})
            if em_cache:
                ok_fn(f"Scored {len(em_cache)} vendor(s). Set category and price for each.")
                _vendor_config_rows(em_cache, "em")
                vby_cat = _build_vby_from_cache(em_cache, "em")

        # --------------------------------------------------------------
        # B) Combined CSV
        # --------------------------------------------------------------
        elif data_source == "Upload combined CSV (vendor + text columns)":
            info_fn(
                "Upload a CSV with <code>vendor</code> and <code>text</code> columns. "
                "Optional columns: <code>category</code>, <code>price</code>."
            )
            combo_file = st.file_uploader(
                "Combined CSV", type=["csv"], key="em_upload_combo"
            )
            models_loaded = "active_ate_model" in st.session_state
            if combo_file and not models_loaded:
                warn_fn("Load models from the sidebar to score uploaded files.")
            elif combo_file and models_loaded:
                import io as _io
                df_combo = pd.read_csv(_io.BytesIO(combo_file.read()))
                if not {"vendor", "text"} <= set(df_combo.columns):
                    err_fn("CSV must have at least 'vendor' and 'text' columns.")
                else:
                    df_combo["vendor"] = df_combo["vendor"].astype(str).str.strip()
                    df_combo["text"]   = df_combo["text"].astype(str).str.strip()
                    df_combo = df_combo[df_combo["text"].str.len() > 0]
                    has_cat   = "category" in df_combo.columns
                    has_price = "price"    in df_combo.columns
                    em_cache  = st.session_state.get("em_scored_cache", {})
                    for vn, grp in df_combo.groupby("vendor", sort=False):
                        if vn in em_cache:
                            continue
                        prof = _score_reviews_em(vn, grp["text"].tolist())
                        if prof:
                            prof["_csv_category"] = (
                                str(grp["category"].iloc[0]).strip().lower()
                                if has_cat else "")
                            prof["_csv_price"] = (
                                float(grp["price"].iloc[0])
                                if has_price else 0.0)
                            em_cache[vn] = prof
                    st.session_state["em_scored_cache"] = em_cache

            em_cache = st.session_state.get("em_scored_cache", {})
            if em_cache:
                ok_fn(f"Scored {len(em_cache)} vendor(s). Confirm category and price.")
                _vendor_config_rows(em_cache, "em")
                vby_cat = _build_vby_from_cache(em_cache, "em")

        # --------------------------------------------------------------
        # C) Manual entry
        # --------------------------------------------------------------
        elif data_source == "Enter vendors manually":
            info_fn(
                "Add vendors by filling in their details. "
                "Aspect scores range from -1.0 (very negative) to +1.0 (very positive). "
                "Leave an aspect at 0.0 if you have no data for it."
            )
            n_manual = st.number_input(
                "Number of vendors to add", min_value=1, max_value=30,
                value=int(st.session_state.get("em_n_manual", 3)),
                step=1, key="em_n_manual",
            )
            _lbl = ("font-family:IBM Plex Mono,monospace;font-size:0.68rem;"
                    "text-transform:uppercase;letter-spacing:0.1em;color:#3a3a3a;"
                    "margin-bottom:0.5rem;")
            st.markdown(f'<div style="{_lbl}">Vendor Details</div>', unsafe_allow_html=True)

            for i in range(int(n_manual)):
                vname_preview = (st.session_state.get(f"em_mv_name_{i}", "")
                                 or f"Vendor {i + 1}")
                with st.expander(vname_preview, expanded=(i == 0)):
                    r1c1, r1c2, r1c3, r1c4 = st.columns([2.5, 2, 1.8, 1.2])
                    with r1c1:
                        st.text_input("Vendor name", key=f"em_mv_name_{i}",
                                      placeholder="e.g. Grand Ballroom")
                    with r1c2:
                        st.text_input("Category", key=f"em_mv_cat_{i}",
                                      placeholder="e.g. venue")
                    with r1c3:
                        st.number_input("Price", min_value=0, value=0,
                                        step=1000, key=f"em_mv_price_{i}")
                    with r1c4:
                        st.number_input("No. of Reviews", min_value=1,
                                        value=10, step=1, key=f"em_mv_rc_{i}")

                    st.markdown(
                        '<div style="font-family:IBM Plex Mono,monospace;font-size:0.65rem;'
                        'text-transform:uppercase;letter-spacing:0.1em;color:#3a3a3a;'
                        'margin:0.6rem 0 0.3rem 0;">Aspect Scores (-1 to +1)</div>',
                        unsafe_allow_html=True,
                    )
                    asp_cols = st.columns(len(ALL_ASPECTS))
                    for j, asp in enumerate(ALL_ASPECTS):
                        with asp_cols[j]:
                            st.slider(asp, -1.0, 1.0, 0.0, 0.05,
                                      key=f"em_mv_asp_{i}_{asp}")

            for i in range(int(n_manual)):
                vn    = st.session_state.get(f"em_mv_name_{i}", "").strip()
                cat   = st.session_state.get(f"em_mv_cat_{i}", "").strip().lower()
                price = float(st.session_state.get(f"em_mv_price_{i}", 0) or 0)
                rc    = int(st.session_state.get(f"em_mv_rc_{i}", 10) or 10)
                if not vn or not cat or price <= 0:
                    continue
                asp_scores = {
                    asp: float(st.session_state.get(f"em_mv_asp_{i}_{asp}", 0.0))
                    for asp in ALL_ASPECTS
                }
                absa = float(np.mean(list(asp_scores.values())))
                vby_cat.setdefault(cat, []).append(RichVendor(
                    name=vn, category=cat, price=price,
                    absa_score=absa, review_count=rc,
                    aspect_scores=asp_scores,
                ))

        # --------------------------------------------------------------
        # D) Inherit from Budget Planner
        # --------------------------------------------------------------
        else:
            bp_cache = st.session_state.get("bp_scored_cache", {})
            if bp_cache:
                info_fn(
                    "Vendors scored in the Budget Planner are imported here. "
                    "Their categories and prices follow the Budget Planner configuration."
                )
                for vn, pd_ in bp_cache.items():
                    cat   = st.session_state.get(f"bp_cat_{vn}", "").strip().lower()
                    price = float(st.session_state.get(f"bp_price_{vn}", 0) or 0)
                    if cat and price > 0:
                        asp = {k: v["score"]
                               for k, v in pd_.get("aspect_profile", {}).items()}
                        vby_cat.setdefault(cat, []).append(RichVendor(
                            name=vn, category=cat, price=price,
                            absa_score=pd_["raw_score"],
                            review_count=pd_["review_count"],
                            aspect_scores=asp,
                        ))
                if not vby_cat:
                    warn_fn(
                        "Budget Planner vendors found but none have a category and price set. "
                        "Configure them in the Budget Planner tab first."
                    )
            else:
                warn_fn(
                    "No Budget Planner data found. Score vendors there first, "
                    "or choose a different data source above."
                )

        # Summary
        if vby_cat:
            total_v = sum(len(v) for v in vby_cat.values())
            ok_fn(
                f"{total_v} vendor(s) ready across "
                f"{len(vby_cat)} categor{'y' if len(vby_cat) == 1 else 'ies'}: "
                f"{', '.join(sorted(vby_cat.keys()))}."
            )
        else:
            warn_fn(
                "No vendors available yet. Fill in all required fields "
                "(name, category, price > 0) or choose a different data source."
            )
            return

    all_cats  = sorted(vby_cat.keys())
    all_names = [v.name for cat in all_cats for v in vby_cat[cat]]


    # ------------------------------------------------------------------
    # Step 2: Event type
    # ------------------------------------------------------------------
    sec_fn("Step 2 - Event Type and Aspect Weights")

    c1, c2 = st.columns([1, 2])
    with c1:
        event_type = st.selectbox(
            "Event type", list(EVENT_TEMPLATES.keys()), key="em_event_type"
        )

    # Build weights: start from template, allow custom override
    base_w = dict(EVENT_TEMPLATES[event_type])  # copy

    with c2:
        if event_type == "Custom" or st.checkbox(
            "Customise aspect weights", key="em_custom_w"
        ):
            st.markdown(
                '<div style="font-family:IBM Plex Mono,monospace;font-size:0.68rem;'
                'text-transform:uppercase;letter-spacing:0.1em;color:#3a3a3a;'
                'margin-bottom:0.4rem;">Adjust importance (will be normalised)</div>',
                unsafe_allow_html=True,
            )
            cols_w = st.columns(len(ALL_ASPECTS))
            custom_w: Dict[str, float] = {}
            for i, asp in enumerate(ALL_ASPECTS):
                with cols_w[i]:
                    custom_w[asp] = st.slider(
                        asp, 0.0, 1.0,
                        float(base_w.get(asp, 1.0 / len(ALL_ASPECTS))),
                        0.05, key=f"em_w_{asp}",
                    )
            s = sum(custom_w.values()) or 1.0
            context_weights = {k: v / s for k, v in custom_w.items()}
        else:
            context_weights = base_w if base_w else {a: 1 / len(ALL_ASPECTS) for a in ALL_ASPECTS}

    # Show template weight bar
    st.pyplot(_template_weight_bar(context_weights, event_type))

    # ------------------------------------------------------------------
    # Step 3: Keywords / priority aspects
    # ------------------------------------------------------------------
    sec_fn("Step 3 - Priority Keywords")
    info_fn(
        "Enter aspect keywords that must be positively rated for a vendor to be preferred. "
        "Vendors matching more keywords score higher. Leave blank to skip."
    )

    kw_input = st.text_input(
        "Keywords (comma-separated)",
        placeholder="e.g. reliability, ambiance, quality",
        key="em_keywords",
    )
    keywords = [k.strip().lower() for k in kw_input.split(",") if k.strip()]
    # Resolve synonyms to canonical aspect keys using local map
    keywords = [_KEYWORD_NORM.get(k, k) for k in keywords]
    keywords = [k for k in keywords if k in ALL_ASPECTS]

    if keywords:
        ok_fn(f"Active keyword filters: {', '.join(keywords)}")

    # ------------------------------------------------------------------
    # Step 4: Budget
    # ------------------------------------------------------------------
    sec_fn("Step 4 - Budget and Category Allocation")

    total_budget_em = st.number_input(
        "Total budget", min_value=1_000, max_value=100_000_000,
        value=250_000, step=5_000, key="em_budget",
    )

    split_mode = st.radio(
        "Budget split method",
        ["Auto (event-priority weighted)", "Equal split", "Manual"],
        key="em_split_mode", horizontal=True,
    )

    if split_mode == "Auto (event-priority weighted)":
        # Weight by average context score importance across categories
        def _cat_weight(cat):
            avg_cs = np.mean([v.context_score(context_weights) for v in vby_cat[cat]]) if vby_cat[cat] else 0.0
            return max(avg_cs, 0.01)
        raw_cw = {c: _cat_weight(c) for c in all_cats}
        s = sum(raw_cw.values()) or 1.0
        cat_budgets_em = {c: total_budget_em * raw_cw[c] / s for c in all_cats}

    elif split_mode == "Equal split":
        share = total_budget_em / max(len(all_cats), 1)
        cat_budgets_em = {c: share for c in all_cats}

    else:
        st.markdown(
            '<div style="font-family:IBM Plex Mono,monospace;font-size:0.68rem;'
            'text-transform:uppercase;letter-spacing:0.1em;color:#3a3a3a;'
            'margin-bottom:0.5rem;">Set % per category</div>',
            unsafe_allow_html=True,
        )
        cols_pct = st.columns(min(len(all_cats), 5))
        pct_em: Dict[str, float] = {}
        for i, cat in enumerate(all_cats):
            with cols_pct[i % len(cols_pct)]:
                pct_em[cat] = st.number_input(
                    cat, 0.0, 100.0,
                    round(100.0 / len(all_cats), 1),
                    1.0, key=f"em_pct_{cat}",
                )
        s = sum(pct_em.values()) or 1.0
        cat_budgets_em = {c: total_budget_em * (pct_em[c] / s) for c in all_cats}

    # Budget preview table
    preview = pd.DataFrame([
        {"category": c,
         "allocated": f"{cat_budgets_em[c]:,.0f}",
         "vendors": len(vby_cat.get(c, [])),
         "avg_context_score": round(
             np.mean([v.context_score(context_weights)
                      for v in vby_cat.get(c, [])]) if vby_cat.get(c) else 0, 3)}
        for c in all_cats
    ])
    st.dataframe(preview, use_container_width=True, hide_index=True)

    # ------------------------------------------------------------------
    # Step 5: Per-vendor customisation
    # ------------------------------------------------------------------
    sec_fn("Step 5 - Vendor Customisation (Optional)")
    info_fn(
        "Pin a specific vendor, exclude vendors you have already ruled out, "
        "or set a minimum star rating per category."
    )

    with st.expander("Open vendor customisation panel", expanded=False):
        pinned:   Dict[str, Optional[str]] = {}
        excluded: Dict[str, List[str]]     = {}
        min_star: Dict[str, float]         = {}

        for cat in all_cats:
            options = [v.name for v in vby_cat[cat]]
            st.markdown(
                f'<div style="font-family:IBM Plex Mono,monospace;font-size:0.72rem;'
                f'color:#777;text-transform:uppercase;letter-spacing:0.08em;'
                f'margin:0.6rem 0 0.2rem 0;">{cat}</div>',
                unsafe_allow_html=True,
            )
            cc1, cc2, cc3 = st.columns([2, 2, 1.5])
            with cc1:
                pin = st.selectbox(
                    "Pin vendor", ["(none)"] + options, key=f"em_pin_{cat}"
                )
                pinned[cat] = None if pin == "(none)" else pin
            with cc2:
                excl = st.multiselect(
                    "Exclude vendors", options, key=f"em_excl_{cat}"
                )
                excluded[cat] = excl
            with cc3:
                ms = st.slider(
                    "Min stars", 1.0, 5.0, 1.0, 0.5, key=f"em_ms_{cat}"
                )
                min_star[cat] = ms

    use_elo_em = st.checkbox(
        "Use Elo ratings as primary ranking signal",
        value=True, key="em_use_elo",
        help="When enabled, Elo ratings (computed within each category) "
             "drive selection. Disable to use raw context scores only.",
    )

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    if st.button("Find Best Event Combination", type="primary", key="em_run"):

        # Run Elo within each category using context weights
        for cat in all_cats:
            _run_elo_within_category(vby_cat[cat], context_weights)

        selection, total_cost = _select_vendors(
            vby_cat       = vby_cat,
            budget        = total_budget_em,
            cat_budgets   = cat_budgets_em,
            context_w     = context_weights,
            keywords      = keywords,
            min_star      = min_star,
            excluded      = excluded,
            pinned        = pinned,
            use_elo       = use_elo_em,
        )

        st.session_state["em_result"] = {
            "selection":    selection,
            "total_cost":   total_cost,
            "cat_budgets":  cat_budgets_em,
            "context_w":    context_weights,
            "keywords":     keywords,
            "vby_cat":      vby_cat,
            "event_type":   event_type,
        }

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    if "em_result" not in st.session_state:
        return

    R           = st.session_state["em_result"]
    sel         = R["selection"]
    cw          = R["context_w"]
    kws         = R["keywords"]
    vby         = R["vby_cat"]
    ev          = R["event_type"]
    cb          = R["cat_budgets"]
    total_cost  = R["total_cost"]
    remaining   = total_budget_em - total_cost

    selected_vs = [v for v in sel.values() if v]
    missing_cat = [c for c, v in sel.items() if v is None]

    if not selected_vs:
        err_fn("No vendors could be selected. Relax constraints or increase budget.")
        return

    # Aggregate event match score: mean context score of selected vendors
    avg_ctx  = float(np.mean([v.context_score(cw) for v in selected_vs]))
    avg_elo  = float(np.mean([v.elo_star for v in selected_vs]))
    avg_absa = float(np.mean([v.absa_score for v in selected_vs]))
    avg_km   = float(np.mean([v.keyword_match_score(kws) for v in selected_vs])) if kws else 1.0

    sec_fn(f"Results: {ev}")

    # Stat row
    stat_row_fn(
        ("Total cost",       f"{total_cost:,.0f}"),
        ("Remaining",        f"{remaining:,.0f}"),
        ("Vendors chosen",   len(selected_vs)),
        ("Avg context score",f"{avg_ctx:.3f}", "#3498db"),
        ("Avg Elo stars",    f"{avg_elo:.2f}",
         "#2ecc71" if avg_elo >= 3.5 else "#f39c12"),
        ("Avg ABSA score",   f"{avg_absa:+.3f}",
         "#2ecc71" if avg_absa >= 0 else "#e74c3c"),
        *([("KW match",      f"{avg_km*100:.0f}%",
            "#2ecc71" if avg_km >= 0.7 else "#f39c12")] if kws else []),
    )

    if missing_cat:
        warn_fn(
            f"No vendor matched constraints for: {', '.join(missing_cat)}. "
            "Try relaxing star gates or exclusions."
        )

    # Match gauge + top pick callout
    sec_fn("Event Match Score")
    g1, g2, g3 = st.columns([1, 1, 2])
    with g1:
        st.pyplot(_match_gauge_fig(avg_ctx, "Context Match"))
    with g2:
        st.pyplot(_match_gauge_fig(avg_km if kws else avg_elo / 5.0,
                                   "Keyword Match" if kws else "Elo Match"))
    with g3:
        top = max(selected_vs, key=lambda v: v.context_score(cw))
        sec_fn("Top Contextual Pick")
        st.markdown(f"""
        <div style="background:#111;border:1px solid #1e1e1e;border-radius:5px;
                    padding:1.2rem 1.6rem;display:inline-block;min-width:260px;">
          <div style="font-family:IBM Plex Mono,monospace;font-size:0.62rem;
                      text-transform:uppercase;letter-spacing:0.1em;
                      color:#3a3a3a;margin-bottom:0.3rem;">{top.category}</div>
          <div style="font-family:IBM Plex Mono,monospace;font-size:1.5rem;
                      font-weight:600;color:#fff;line-height:1.1;">{top.name}</div>
          <div style="margin:0.3rem 0;">{stars_fn(top.elo_star, "1.1rem")}</div>
          <div style="font-family:IBM Plex Mono,monospace;font-size:0.72rem;
                      color:#555;margin-top:0.2rem;">
            context {top.context_score(cw):.3f} &nbsp;|&nbsp;
            price {top.price:,.0f} &nbsp;|&nbsp;
            {top.review_count} reviews
          </div>
        </div>""", unsafe_allow_html=True)

    # Selection table
    sec_fn("Recommended Vendor Combination")
    st.markdown(
        _selection_table_html(sel, cb, cw, kws, stars_fn),
        unsafe_allow_html=True,
    )

    # Radar chart
    sec_fn("Aspect Radar: Selected Vendors")
    st.pyplot(_radar(selected_vs, ALL_ASPECTS, f"Selected Vendors - {ev}"))

    # Elo bar
    sec_fn("Elo Rankings Across All Vendors")
    st.pyplot(_elo_bar(vby, sel))

    # Context score scatter
    sec_fn("Event Context Score vs Price")
    st.pyplot(_context_scatter(vby, cw, sel))

    # Heatmap
    sec_fn("Full Vendor x Aspect Heatmap")
    hm = _heatmap(vby, ALL_ASPECTS, cw)
    if hm:
        st.pyplot(hm)

    # Keyword match bar (only if keywords active)
    if kws:
        sec_fn("Keyword Match by Selected Vendor")
        km_fig = _keyword_match_bar(selected_vs, kws)
        if km_fig:
            st.pyplot(km_fig)

    # Per-category radar drill-down
    sec_fn("Category Deep Dive")
    drill_cat = st.selectbox(
        "Select category to compare all candidates",
        all_cats, key="em_drill_cat",
    )
    if drill_cat and drill_cat in vby:
        drill_v = sorted(vby[drill_cat], key=lambda v: -v.elo_star)
        st.pyplot(_radar(drill_v, ALL_ASPECTS, f"{drill_cat}: All Candidates"))
        drill_df = pd.DataFrame([{
            "vendor":        v.name,
            "elo_star":      v.elo_star,
            "elo_rating":    v.elo_rating,
            "context_score": round(v.context_score(cw), 4),
            "absa_score":    round(v.absa_score, 4),
            "price":         v.price,
            "reviews":       v.review_count,
            **{f"asp_{a}": round(v.aspect_scores.get(a, 0.0), 3) for a in ALL_ASPECTS},
        } for v in drill_v])
        st.dataframe(drill_df, use_container_width=True, hide_index=True)

    # Download
    sec_fn("Download Results")
    dl_rows = []
    for cat, v in sel.items():
        dl_rows.append({
            "event_type":    ev,
            "category":      cat,
            "allocated":     round(cb.get(cat, 0), 2),
            "vendor":        v.name if v else None,
            "price":         v.price if v else None,
            "elo_star":      v.elo_star if v else None,
            "absa_score":    round(v.absa_score, 4) if v else None,
            "context_score": round(v.context_score(cw), 4) if v else None,
            "kw_match_pct":  round(v.keyword_match_score(kws) * 100, 1) if v else None,
            **({f"asp_{a}": round(v.aspect_scores.get(a, 0.0), 3)
                for a in ALL_ASPECTS} if v else {}),
        })
    st.download_button(
        "Download Event Match CSV",
        pd.DataFrame(dl_rows).to_csv(index=False).encode(),
        "event_match_recommendation.csv",
        "text/csv",
    )