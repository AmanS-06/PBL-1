"""
ABSA + ELO engine extracted verbatim from absa_event_planner_streamlit_app.py.
No logic has been changed.  Only import statements and the Streamlit-specific
session_state / st.* calls have been removed; all numerical logic is intact.
"""

import itertools
import warnings
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from scipy.special import softmax as _softmax

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SENTIMENT_LABEL_NAMES = ["positive", "negative", "neutral", "conflict"]
SENTIMENT_SCORE_MAP   = {"positive": 1.0, "negative": -1.0, "neutral": 0.0, "conflict": -0.5}
SCORE_VEC             = np.array([SENTIMENT_SCORE_MAP[l] for l in SENTIMENT_LABEL_NAMES])

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
    "planner": "service", "coordinator": "service", "organiser": "service",
    "organizer": "service", "florist": "service", "baker": "service",
    "caterer": "service", "driver": "service", "chauffeur": "service",
    "support": "service", "communication": "service", "responsiveness": "service",
    "professionalism": "service", "behaviour": "service", "behavior": "service",
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
# Rule-based ABSA (keyword + lexicon + negation window)
# Mirrors the Streamlit app's pipeline when transformer models are absent.
# ---------------------------------------------------------------------------

POSITIVE_WORDS = {
    "amazing", "excellent", "fantastic", "great", "good", "wonderful", "outstanding",
    "superb", "perfect", "beautiful", "lovely", "brilliant", "exceptional", "impressive",
    "delicious", "delightful", "professional", "friendly", "helpful", "clean", "fresh",
    "punctual", "reliable", "affordable", "reasonable", "worth", "recommend", "love",
    "best", "top", "incredible", "spectacular", "magnificent", "marvelous", "awesome",
    "happy", "pleased", "satisfied", "enjoyed", "impressed", "praise", "flawless",
}

NEGATIVE_WORDS = {
    "terrible", "awful", "horrible", "bad", "poor", "disappointing", "worst",
    "rude", "slow", "late", "dirty", "expensive", "overpriced", "unprofessional",
    "unreliable", "mediocre", "average", "bland", "tasteless", "cold", "stale",
    "broken", "damaged", "missing", "wrong", "mistake", "error", "failed", "failure",
    "disgusting", "nauseating", "offensive", "disrespectful", "ignored", "waited",
    "delayed", "cancelled", "refund", "complaint", "issue", "problem", "waste",
}

NEGATION_WORDS = {
    "not", "no", "never", "neither", "nor", "without", "lack", "lacks",
    "lacking", "barely", "hardly", "scarcely", "didn't", "wasn't",
    "isn't", "aren't", "couldn't", "wouldn't", "shouldn't",
}

ASPECT_KEYWORDS = {
    "food":        ["food", "meal", "dish", "taste", "flavor", "flavour", "cuisine",
                    "menu", "portion", "ingredient", "dessert", "cake"],
    "service":     ["service", "staff", "waiter", "waitress", "server", "host",
                    "chef", "team", "crew", "coordinator", "planner"],
    "ambiance":    ["ambiance", "ambience", "atmosphere", "venue", "decor", "decoration",
                    "setting", "place", "hall", "room", "space", "cleanliness"],
    "value":       ["price", "cost", "value", "pricing", "fee", "charge", "rate",
                    "expensive", "cheap", "affordable", "budget", "package"],
    "reliability": ["reliability", "punctuality", "timing", "delivery", "wait",
                    "delay", "speed", "schedule", "deadline", "promptness"],
    "quality":     ["quality", "photography", "photos", "video", "music", "performance",
                    "entertainment", "lighting", "equipment", "setup", "output"],
}


def _detect_aspects(text):
    text_lower = text.lower()
    found = []
    for aspect, keywords in ASPECT_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                found.append(aspect)
                break
    return list(dict.fromkeys(found)) if found else ["overall"]


def _score_aspect(text, aspect):
    text_lower  = text.lower()
    window_words = text_lower.split()
    for kw in ASPECT_KEYWORDS.get(aspect, []):
        if kw in text_lower:
            idx   = text_lower.find(kw)
            start = max(0, idx - 80)
            end   = min(len(text_lower), idx + 80)
            window_words = text_lower[start:end].split()
            break

    pos_count = neg_count = 0
    for i, w in enumerate(window_words):
        clean = w.strip(".,!?;:'\"()")
        prior = window_words[max(0, i - 2):i]
        is_negated = any(p.strip(".,!?") in NEGATION_WORDS for p in prior)
        if clean in POSITIVE_WORDS:
            neg_count += 1 if is_negated else 0
            pos_count += 0 if is_negated else 1
        elif clean in NEGATIVE_WORDS:
            pos_count += 0.5 if is_negated else 0
            neg_count += 0   if is_negated else 1

    total = pos_count + neg_count
    if total == 0:
        return "neutral", 0.55, np.array([0.1, 0.1, 0.7, 0.1]), 0.0

    ratio = pos_count / total
    if ratio >= 0.65:
        label = "positive"
        probs = np.array([ratio, 1 - ratio - 0.05, 0.05, 0.0])
    elif ratio <= 0.35:
        label = "negative"
        probs = np.array([ratio, 1 - ratio - 0.05, 0.05, 0.0])
    else:
        label = "conflict" if abs(pos_count - neg_count) <= 1 else "neutral"
        probs = np.array([0.2, 0.2, 0.4, 0.2])

    probs      = np.clip(probs, 0, 1)
    probs     /= probs.sum()
    confidence = float(probs[SENTIMENT_LABEL_NAMES.index(label)])
    weighted   = float(np.dot(probs, SCORE_VEC))
    return label, confidence, probs, weighted


def analyse_text(text):
    aspects = _detect_aspects(text)
    results = []
    for asp in aspects:
        label, conf, probs, weighted = _score_aspect(text, asp)
        results.append({
            "aspect":         asp,
            "sentiment":      label,
            "confidence":     round(conf, 4),
            "weighted_score": round(weighted, 4),
            "probs": {k: round(float(probs[i]), 4)
                      for i, k in enumerate(SENTIMENT_LABEL_NAMES)},
        })
    return results


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
        sents    = [r["sentiment"] for r in results_list
                    if (normalise_aspect(r["aspect"]) if use_norm
                        else r["aspect"].lower()) == a]
        dominant = max(set(sents), key=sents.count) if sents else "neutral"
        profile[a] = {
            "score":    round(asp_means[a], 4),
            "count":    len(asp_scores[a]),
            "weight":   round(len(asp_scores[a]) / total, 4),
            "dominant": dominant,
        }
    return {
        "raw_score":      round(raw_score, 4),
        "review_count":   review_count,
        "aspect_profile": profile,
    }


# ---------------------------------------------------------------------------
# Elo Ranker  (verbatim from Streamlit app)
# ---------------------------------------------------------------------------

class SentimentEloRanker:
    def __init__(self, k=32, initial=1500, bayes_m=10):
        self.k          = k
        self.initial    = initial
        self.bayes_m    = bayes_m
        self.ratings    = {}
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
                    vi, vj   = vendors[i], vendors[j]
                    ri, rj   = self.ratings[vi], self.ratings[vj]
                    si = (self.raw_scores[vi] + 1) / 2
                    sj = (self.raw_scores[vj] + 1) / 2
                    actual   = si / (si + sj + 1e-9)
                    expected = 1.0 / (1.0 + 10 ** ((rj - ri) / 400))
                    self.ratings[vi] = ri + self.k * (actual - expected)

    def get_rankings(self):
        if not self.ratings:
            return []
        self._update()
        names   = list(self.ratings.keys())
        elo_arr = np.array([self.ratings[n] for n in names])
        raw_arr = np.array([self.raw_scores[n] for n in names])
        mn, mx  = elo_arr.min(), elo_arr.max()
        stars   = (np.full(len(names), 3.0) if mx == mn
                   else 1.0 + (elo_arr - mn) / (mx - mn) * 4.0)
        se = np.abs(raw_arr) * 0.08 + 0.05
        rows = []
        for i, name in enumerate(names):
            rows.append({
                "vendor":      name,
                "elo_rating":  round(float(elo_arr[i]), 1),
                "raw_score":   round(float(raw_arr[i]), 4),
                "star_rating": round(float(np.clip(stars[i], 1, 5)), 2),
                "ci_lo":       round(float(np.clip(stars[i] - 1.96 * se[i], 1, 5)), 2),
                "ci_hi":       round(float(np.clip(stars[i] + 1.96 * se[i], 1, 5)), 2),
                "reviews":     self.rev_counts[name],
            })
        rows.sort(key=lambda r: r["elo_rating"], reverse=True)
        for idx, r in enumerate(rows):
            r["rank"] = idx + 1
        return rows


# ---------------------------------------------------------------------------
# Budget optimisation (verbatim from Streamlit app)
# ---------------------------------------------------------------------------

@dataclass
class _Vendor:
    name:         str
    category:     str
    price:        float
    absa_score:   float
    review_count: int = 0

    @property
    def norm_score(self):
        return float(np.clip((self.absa_score + 1) / 2, 0.0, 1.0))

    @property
    def star_rating(self):
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


def _vendor_to_dict(v):
    if v is None:
        return None
    return {
        "name":         v.name,
        "category":     v.category,
        "price":        v.price,
        "absa_score":   v.absa_score,
        "star_rating":  round(v.star_rating, 2),
        "norm_score":   round(v.norm_score, 4),
        "review_count": v.review_count,
    }


def _build_vby_cat(vendors_payload):
    vby_cat = defaultdict(list)
    for v in vendors_payload:
        vby_cat[v["category"]].append(
            _Vendor(
                name         = v["name"],
                category     = v["category"],
                price        = float(v["price"]),
                absa_score   = float(v.get("absa_score", 0.0)),
                review_count = int(v.get("review_count", 0)),
            )
        )
    return dict(vby_cat)


def recommend_auto(total_budget, priority_order, vby_cat, lam=2.0):
    splits  = _auto_split(total_budget, priority_order, lam)
    w       = _priority_weights(priority_order, lam)
    greedy  = _greedy(splits, vby_cat)
    missing = [c for c, v in greedy.items() if v is None]

    if not missing:
        sel    = greedy
        cost   = sum(v.price for v in sel.values() if v is not None)
        solver = "greedy"
    else:
        sel, cost = _exhaustive(total_budget, priority_order, vby_cat, lam)
        solver    = "exhaustive"
        if not any(sel.values()):
            sel    = greedy
            cost   = sum(v.price for v in greedy.values() if v is not None)
            solver = "greedy (fallback)"

    obj = sum(w.get(cat, 0) * v.norm_score
              for cat, v in sel.items() if v is not None)
    return {
        "selection":  {cat: _vendor_to_dict(v) for cat, v in sel.items()},
        "total_cost": round(cost, 2),
        "remaining":  round(total_budget - cost, 2),
        "splits":     {k: round(v, 2) for k, v in splits.items()},
        "weights":    {k: round(v, 4) for k, v in w.items()},
        "obj":        round(obj, 4),
        "solver":     solver,
    }


def recommend_manual(total_budget, cat_pct, vby_cat, reallocate=True):
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
            changed     = False
            sorted_cats = sorted(sel.keys(), key=lambda c: norm_pct.get(c, 0), reverse=True)
            for cat in sorted_cats:
                cur   = sel[cat]
                cands = vby_cat.get(cat, [])
                cur_p = cur.price if cur else 0.0
                ups   = [v for v in cands
                         if v.norm_score > (cur.norm_score if cur else -1)
                         and v.price <= cur_p + savings]
                if ups:
                    best     = max(ups, key=lambda v: v.norm_score)
                    delta    = best.price - cur_p
                    savings -= delta
                    cost    += delta
                    upgrades.append(
                        f"{cat}: {cur.name if cur else 'None'} -> {best.name} "
                        f"(+{delta:,.0f}, score "
                        f"{(cur.absa_score if cur else 0):+.3f} -> {best.absa_score:+.3f})"
                    )
                    sel[cat] = best
                    changed  = True
                    break

    scored = [v for v in sel.values() if v is not None]
    obj    = float(np.mean([v.norm_score for v in scored])) if scored else 0.0
    return {
        "selection":  {cat: _vendor_to_dict(v) for cat, v in sel.items()},
        "total_cost": round(cost, 2),
        "remaining":  round(total_budget - cost, 2),
        "budgets":    {k: round(v, 2) for k, v in budgets.items()},
        "pct":        {k: round(v, 2) for k, v in norm_pct.items()},
        "obj":        round(obj, 4),
        "upgrades":   upgrades,
    }


# ---------------------------------------------------------------------------
# Offline pipeline: process raw reviews -> store-ready scores
# ---------------------------------------------------------------------------

def process_reviews_offline(vendor_name, reviews):
    """
    Given a vendor name and list of review strings, run full ABSA + profile
    computation and return a dict ready to be written to vendor_scores.
    """
    flat = []
    for rev in reviews:
        flat.extend(analyse_text(str(rev)))

    profile = compute_vendor_profile(flat, len(reviews), use_norm=True)
    if not profile:
        return None

    asp = profile["aspect_profile"]
    return {
        "raw_score":           profile["raw_score"],
        "review_count":        profile["review_count"],
        "aspect_quality":      asp.get("quality",     {}).get("score"),
        "aspect_service":      asp.get("service",     {}).get("score"),
        "aspect_value":        asp.get("value",       {}).get("score"),
        "aspect_ambiance":     asp.get("ambiance",    {}).get("score"),
        "aspect_reliability":  asp.get("reliability", {}).get("score"),
        "aspect_experience":   asp.get("experience",  {}).get("score"),
    }
