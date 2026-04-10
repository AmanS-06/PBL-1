# HOW TO RUN - Eventify + ABSA Integration

## File Structure

```
eventify/
  server.js                  (Node.js backend - updated)
  index.html                 (Frontend SPA - updated)
  .env.example               (copy to .env and fill in)
  absa_service/
    app.py                   (Python ABSA microservice - new)
    requirements.txt
```

---

## 1. ABSA Python Microservice

```bash
cd absa_service

# Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt

python app.py
# Listening on http://localhost:8000
```

To verify: `curl http://localhost:8000/health`

---

## 2. Node.js Backend

```bash
# In the project root
cp .env.example .env
# Fill in DB_*, CLOUDINARY_*, STRIPE_* values

npm install
npm run dev
# Listening on http://localhost:5000
```

---

## 3. Frontend

```bash
# Serve index.html with any static file server
npx http-server -p 3000

# OR
python3 -m http.server 3000

# Access: http://localhost:3000
```

---

## Request Flow

```
Browser (index.html)
  |
  |-- POST /api/absa/analyze          --> Node proxy --> Flask :8000/analyze
  |-- POST /api/absa/analyze-vendor   --> Node proxy --> Flask :8000/analyze-vendor
  |-- POST /api/absa/rank-vendors     --> Node proxy --> Flask :8000/rank-vendors
  |-- POST /api/absa/recommend        --> Node proxy --> Flask :8000/recommend
  |
  |-- POST /api/events/:id/generate-recommendations
  |       now calls Flask /rank-vendors internally and stores Elo-ranked results
  |
  |-- All other /api/* routes         --> Node --> MySQL (unchanged)
```

---

## ABSA Endpoint Reference

### POST /api/absa/analyze
```json
{ "text": "The food was amazing but service was really slow." }
```
Returns aspect-level sentiment for each detected aspect.

### POST /api/absa/analyze-vendor
```json
{
  "vendor_name": "Elite Cuisine Co.",
  "reviews": ["Great food!", "Service was slow."],
  "use_norm": true
}
```
Returns an aggregated vendor profile (raw_score, aspect_profile, star_rating).

### POST /api/absa/rank-vendors
```json
{
  "vendors": [
    { "name": "V1", "raw_score": 0.6, "review_count": 40 },
    { "name": "V2", "raw_score": 0.3, "review_count": 20 }
  ]
}
```
Returns Elo-ranked vendor list.

### POST /api/absa/recommend
```json
{
  "mode": "auto",
  "total_budget": 250000,
  "priority_order": ["venue", "catering", "photography"],
  "lambda": 2.0,
  "vendors": [
    { "name": "Grand Hall", "category": "venue",
      "price": 80000, "absa_score": 0.75, "review_count": 30 },
    { "name": "Elite Catering", "category": "catering",
      "price": 60000, "absa_score": 0.82, "review_count": 45 }
  ]
}
```
Returns the optimal vendor selection within the budget.

---

## What Changed in index.html

- Nav bar: added "ABSA Planner" button (gradient, always visible)
- Vendor detail modal: ABSA review analyser panel appended at the bottom
  - Paste any review text, click Analyse, see per-aspect sentiment breakdown
- New ABSA Budget Planner modal (opened from nav button)
  - Enter total budget, mode (auto/manual), paste vendors as JSON
  - Returns the recommended vendor combination with cost breakdown

## What Changed in server.js

- Added `ABSA_URL` constant (reads from env, defaults to `http://localhost:8000`)
- Added proxy helper `proxyAbsa(path, body)` using native `fetch`
- Added four new routes: `/api/absa/analyze`, `/api/absa/analyze-vendor`,
  `/api/absa/rank-vendors`, `/api/absa/recommend`
- Replaced the old `/api/events/:eventId/generate-recommendations` handler
  with one that calls the ABSA service for Elo ranking before storing results

## Note on ABSA Engine

`absa_service/app.py` uses a lexicon + negation-aware rule-based engine that
runs without any model weights. It detects aspects via keyword matching and
scores sentiment using curated positive/negative word lists with a
two-word negation window. This replaces the transformer inference from
`absa_event_planner_streamlit_app.py` to keep the microservice self-contained
and dependency-light. The Elo ranker, budget optimiser (greedy + exhaustive
fallback), and aspect normalisation maps are ported exactly from the
Streamlit app.
