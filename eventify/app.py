"""
Eventify ABSA Backend
Flask app exposing:
  OFFLINE endpoints (called from admin/cron, not the browser):
    POST /api/offline/ingest          - store raw reviews, run ABSA, store scores
    POST /api/offline/recompute-elo   - recompute Elo for all vendors in a category

  ONLINE endpoints (called by index.html):
    GET  /api/vendors                 - list vendors (with Elo star ratings from DB)
    GET  /api/vendors/<id>            - single vendor detail + scores
    GET  /api/vendors/ranked          - Elo-ranked list for a category
    POST /api/analyze                 - analyse a single review text (live)
    POST /api/recommend               - budget recommendation using DB vendors
    POST /api/vendors/<id>/reviews    - add review(s), rerun ABSA, update DB

    GET  /api/health
"""

import os
from flask import Flask, request, jsonify
from flask_cors import CORS

import db as _db
import absa_engine as absa

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

with app.app_context():
    _db.init_schema()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ASPECT_COLS = [
    "aspect_quality", "aspect_service", "aspect_value",
    "aspect_ambiance", "aspect_reliability", "aspect_experience",
]


def _row(cursor):
    cols = [d[0] for d in cursor.description]
    return [dict(zip(cols, row)) for row in cursor.fetchall()]


def _one(cursor):
    cols = [d[0] for d in cursor.description]
    row  = cursor.fetchone()
    return dict(zip(cols, row)) if row else None


def _fetch_vendors_for_elo(category=None):
    conn = _db.get_conn()
    cur  = conn.cursor()
    if category:
        cur.execute("""
            SELECT v.id, v.name, v.category,
                   COALESCE(s.raw_score, 0)      AS raw_score,
                   COALESCE(s.review_count, 0)   AS review_count,
                   COALESCE(s.elo_score, 1500)   AS elo_score,
                   COALESCE(s.star_rating, 3)    AS star_rating
            FROM vendors v
            LEFT JOIN vendor_scores s ON s.vendor_id = v.id
            WHERE v.category = %s
        """, (category,))
    else:
        cur.execute("""
            SELECT v.id, v.name, v.category,
                   COALESCE(s.raw_score, 0)      AS raw_score,
                   COALESCE(s.review_count, 0)   AS review_count,
                   COALESCE(s.elo_score, 1500)   AS elo_score,
                   COALESCE(s.star_rating, 3)    AS star_rating
            FROM vendors v
            LEFT JOIN vendor_scores s ON s.vendor_id = v.id
        """)
    rows = _row(cur)
    cur.close()
    conn.close()
    return rows


def _run_elo_and_persist(rows):
    """Run SentimentEloRanker on rows, write updated elo_score + star_rating to DB."""
    if not rows:
        return []
    ranker = absa.SentimentEloRanker(k=32, initial=1500, bayes_m=10)
    id_map = {r["name"]: r["id"] for r in rows}
    for r in rows:
        ranker.add_vendor(r["name"], {
            "raw_score":    r["raw_score"],
            "review_count": max(r["review_count"], 1),
        })
    rankings = ranker.get_rankings()

    conn = _db.get_conn()
    cur  = conn.cursor()
    for rank in rankings:
        vid = id_map.get(rank["vendor"])
        if vid is None:
            continue
        cur.execute("""
            INSERT INTO vendor_scores (vendor_id, elo_score, star_rating)
            VALUES (%s, %s, %s)
            ON DUPLICATE KEY UPDATE elo_score = VALUES(elo_score),
                                    star_rating = VALUES(star_rating)
        """, (vid, rank["elo_rating"], rank["star_rating"]))
    conn.commit()
    cur.close()
    conn.close()
    return rankings


# ---------------------------------------------------------------------------
# OFFLINE routes
# ---------------------------------------------------------------------------

@app.post("/api/offline/ingest")
def offline_ingest():
    """
    Store raw reviews for a vendor, run ABSA, persist scores to vendor_scores.

    Body:
    {
        "vendor_id": 1,
        "reviews": ["Amazing food!", "Service was slow."]
    }
    """
    data      = request.get_json(silent=True) or {}
    vendor_id = data.get("vendor_id")
    reviews   = data.get("reviews", [])

    if not vendor_id or not reviews:
        return jsonify({"error": "vendor_id and reviews required"}), 400

    conn = _db.get_conn()
    cur  = conn.cursor()

    cur.execute("SELECT name FROM vendors WHERE id = %s", (vendor_id,))
    row = cur.fetchone()
    if not row:
        cur.close(); conn.close()
        return jsonify({"error": "vendor not found"}), 404
    vendor_name = row[0]

    for rev in reviews:
        cur.execute(
            "INSERT INTO vendor_reviews (vendor_id, review) VALUES (%s, %s)",
            (vendor_id, str(rev))
        )

    scores = absa.process_reviews_offline(vendor_name, reviews)
    if scores:
        cur.execute("""
            INSERT INTO vendor_scores
                (vendor_id, raw_score, review_count,
                 aspect_quality, aspect_service, aspect_value,
                 aspect_ambiance, aspect_reliability, aspect_experience)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON DUPLICATE KEY UPDATE
                raw_score          = VALUES(raw_score),
                review_count       = review_count + VALUES(review_count),
                aspect_quality     = VALUES(aspect_quality),
                aspect_service     = VALUES(aspect_service),
                aspect_value       = VALUES(aspect_value),
                aspect_ambiance    = VALUES(aspect_ambiance),
                aspect_reliability = VALUES(aspect_reliability),
                aspect_experience  = VALUES(aspect_experience)
        """, (
            vendor_id,
            scores["raw_score"],
            scores["review_count"],
            scores.get("aspect_quality"),
            scores.get("aspect_service"),
            scores.get("aspect_value"),
            scores.get("aspect_ambiance"),
            scores.get("aspect_reliability"),
            scores.get("aspect_experience"),
        ))

    conn.commit()
    cur.close()
    conn.close()

    cur2 = _db.get_conn().cursor()
    cur2.execute("SELECT category FROM vendors WHERE id = %s", (vendor_id,))
    cat = cur2.fetchone()[0]
    cur2.close()

    cat_rows = _fetch_vendors_for_elo(category=cat)
    _run_elo_and_persist(cat_rows)

    return jsonify({"success": True, "vendor_id": vendor_id, "scores": scores})


@app.post("/api/offline/recompute-elo")
def offline_recompute_elo():
    """
    Recompute Elo ratings for all vendors in a category (or all vendors).

    Body: { "category": "catering" }   (omit for all categories)
    """
    data     = request.get_json(silent=True) or {}
    category = data.get("category")

    rows     = _fetch_vendors_for_elo(category=category)
    rankings = _run_elo_and_persist(rows)
    return jsonify({"success": True, "rankings": rankings})


# ---------------------------------------------------------------------------
# ONLINE routes
# ---------------------------------------------------------------------------

@app.get("/api/vendors")
def list_vendors():
    """
    Returns all vendors with Elo star ratings fetched from DB.
    Optional query param: ?category=catering
    """
    category = request.args.get("category")
    conn     = _db.get_conn()
    cur      = conn.cursor()

    if category:
        cur.execute("""
            SELECT v.id, v.name, v.category,
                   d.price, d.location, d.contact,
                   COALESCE(s.elo_score,   1500) AS elo_score,
                   COALESCE(s.raw_score,      0) AS raw_score,
                   COALESCE(s.star_rating,    3) AS star_rating,
                   COALESCE(s.review_count,   0) AS review_count
            FROM vendors v
            LEFT JOIN vendor_details d ON d.vendor_id = v.id
            LEFT JOIN vendor_scores  s ON s.vendor_id = v.id
            WHERE v.category = %s
            ORDER BY s.elo_score DESC
        """, (category,))
    else:
        cur.execute("""
            SELECT v.id, v.name, v.category,
                   d.price, d.location, d.contact,
                   COALESCE(s.elo_score,   1500) AS elo_score,
                   COALESCE(s.raw_score,      0) AS raw_score,
                   COALESCE(s.star_rating,    3) AS star_rating,
                   COALESCE(s.review_count,   0) AS review_count
            FROM vendors v
            LEFT JOIN vendor_details d ON d.vendor_id = v.id
            LEFT JOIN vendor_scores  s ON s.vendor_id = v.id
            ORDER BY s.elo_score DESC
        """)

    vendors = _row(cur)
    cur.close()
    conn.close()
    return jsonify(vendors)


@app.get("/api/vendors/ranked")
def ranked_vendors():
    """
    Returns Elo-ranked vendors, optionally filtered by category.
    Runs live Elo computation on current DB scores and returns ranked list.
    ?category=catering
    """
    category = request.args.get("category")
    rows     = _fetch_vendors_for_elo(category=category)

    if not rows:
        return jsonify([])

    ranker = absa.SentimentEloRanker(k=32, initial=1500, bayes_m=10)
    id_map = {r["name"]: r for r in rows}
    for r in rows:
        ranker.add_vendor(r["name"], {
            "raw_score":    r["raw_score"],
            "review_count": max(r["review_count"], 1),
        })
    rankings = ranker.get_rankings()

    conn = _db.get_conn()
    cur  = conn.cursor()
    for rank in rankings:
        meta = id_map.get(rank["vendor"], {})
        rank["vendor_id"] = meta.get("id")
        rank["category"]  = meta.get("category")
        cur.execute("""
            SELECT d.price, d.location, d.contact
            FROM vendor_details d WHERE d.vendor_id = %s
        """, (meta.get("id"),))
        detail = cur.fetchone()
        if detail:
            rank["price"]    = float(detail[0]) if detail[0] else 0
            rank["location"] = detail[1]
            rank["contact"]  = detail[2]
    cur.close()
    conn.close()
    return jsonify(rankings)


@app.get("/api/vendors/<int:vendor_id>")
def get_vendor(vendor_id):
    conn = _db.get_conn()
    cur  = conn.cursor()
    cur.execute("""
        SELECT v.id, v.name, v.category,
               d.price, d.location, d.contact,
               COALESCE(s.elo_score,          1500) AS elo_score,
               COALESCE(s.raw_score,             0) AS raw_score,
               COALESCE(s.star_rating,            3) AS star_rating,
               COALESCE(s.review_count,           0) AS review_count,
               s.aspect_quality, s.aspect_service, s.aspect_value,
               s.aspect_ambiance, s.aspect_reliability, s.aspect_experience
        FROM vendors v
        LEFT JOIN vendor_details d ON d.vendor_id = v.id
        LEFT JOIN vendor_scores  s ON s.vendor_id = v.id
        WHERE v.id = %s
    """, (vendor_id,))
    vendor = _one(cur)
    if not vendor:
        cur.close(); conn.close()
        return jsonify({"error": "not found"}), 404

    cur.execute(
        "SELECT review, created_at FROM vendor_reviews WHERE vendor_id = %s ORDER BY created_at DESC LIMIT 20",
        (vendor_id,)
    )
    vendor["recent_reviews"] = [{"review": r[0], "created_at": str(r[1])} for r in cur.fetchall()]
    cur.close()
    conn.close()
    return jsonify(vendor)


@app.post("/api/vendors/<int:vendor_id>/reviews")
def add_reviews(vendor_id):
    """
    Add one or more reviews, rerun ABSA, update scores in DB, recompute Elo.

    Body: { "reviews": ["Great venue!", "A bit pricey."] }
    """
    data    = request.get_json(silent=True) or {}
    reviews = data.get("reviews", [])
    if not reviews:
        return jsonify({"error": "reviews list required"}), 400

    conn = _db.get_conn()
    cur  = conn.cursor()
    cur.execute("SELECT name, category FROM vendors WHERE id = %s", (vendor_id,))
    row = cur.fetchone()
    if not row:
        cur.close(); conn.close()
        return jsonify({"error": "vendor not found"}), 404
    vendor_name, category = row

    for rev in reviews:
        cur.execute("INSERT INTO vendor_reviews (vendor_id, review) VALUES (%s, %s)",
                    (vendor_id, str(rev)))

    cur.execute("SELECT review FROM vendor_reviews WHERE vendor_id = %s", (vendor_id,))
    all_reviews = [r[0] for r in cur.fetchall()]

    scores = absa.process_reviews_offline(vendor_name, all_reviews)
    if scores:
        cur.execute("""
            INSERT INTO vendor_scores
                (vendor_id, raw_score, review_count,
                 aspect_quality, aspect_service, aspect_value,
                 aspect_ambiance, aspect_reliability, aspect_experience)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON DUPLICATE KEY UPDATE
                raw_score          = VALUES(raw_score),
                review_count       = VALUES(review_count),
                aspect_quality     = VALUES(aspect_quality),
                aspect_service     = VALUES(aspect_service),
                aspect_value       = VALUES(aspect_value),
                aspect_ambiance    = VALUES(aspect_ambiance),
                aspect_reliability = VALUES(aspect_reliability),
                aspect_experience  = VALUES(aspect_experience)
        """, (
            vendor_id,
            scores["raw_score"],
            scores["review_count"],
            scores.get("aspect_quality"),
            scores.get("aspect_service"),
            scores.get("aspect_value"),
            scores.get("aspect_ambiance"),
            scores.get("aspect_reliability"),
            scores.get("aspect_experience"),
        ))

    conn.commit()
    cur.close()
    conn.close()

    cat_rows = _fetch_vendors_for_elo(category=category)
    _run_elo_and_persist(cat_rows)

    return jsonify({"success": True, "scores": scores})


@app.post("/api/analyze")
def analyze_review():
    """Live single-review ABSA (no DB write)."""
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "text required"}), 400

    results = absa.analyse_text(text)
    if not results:
        return jsonify({"aspects": [], "overall_score": 0.0, "star_rating": 2.5})

    import numpy as np
    overall = float(np.mean([r["weighted_score"] for r in results]))
    star    = round(float(np.clip((overall + 1) / 2 * 5, 0, 5)), 2)
    return jsonify({"aspects": results, "overall_score": round(overall, 4), "star_rating": star})


@app.post("/api/recommend")
def recommend():
    """
    Budget-aware recommendation using vendor data from MySQL.

    Body:
    {
        "total_budget": 250000,
        "mode": "auto",
        "priority_order": ["venue","catering","photography"],
        "lambda": 2.0,
        "category_pct": { "venue": 40, "catering": 35, "photography": 25 },
        "reallocate": true
    }
    Vendors are fetched from DB automatically.
    """
    data         = request.get_json(silent=True) or {}
    total_budget = float(data.get("total_budget", 100000))
    mode         = data.get("mode", "auto")

    conn = _db.get_conn()
    cur  = conn.cursor()
    cur.execute("""
        SELECT v.name, v.category,
               COALESCE(d.price, 0)       AS price,
               COALESCE(s.raw_score, 0)   AS absa_score,
               COALESCE(s.review_count,1) AS review_count
        FROM vendors v
        LEFT JOIN vendor_details d ON d.vendor_id = v.id
        LEFT JOIN vendor_scores  s ON s.vendor_id = v.id
        WHERE d.price IS NOT NULL AND d.price > 0
    """)
    rows = _row(cur)
    cur.close()
    conn.close()

    if not rows:
        return jsonify({"error": "no vendors with price data in DB"}), 400

    vby_cat = absa._build_vby_cat(rows)

    if mode == "auto":
        priority_order = data.get("priority_order", list(vby_cat.keys()))
        lam            = float(data.get("lambda", 2.0))
        result         = absa.recommend_auto(total_budget, priority_order, vby_cat, lam)
    else:
        cats    = list(vby_cat.keys())
        cat_pct = data.get("category_pct", {c: 100 / len(cats) for c in cats})
        reallocate = bool(data.get("reallocate", True))
        result  = absa.recommend_manual(total_budget, cat_pct, vby_cat, reallocate)

    return jsonify(result)


@app.get("/api/health")
def health():
    return jsonify({"status": "ok"})


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=bool(os.environ.get("DEBUG", False)))
