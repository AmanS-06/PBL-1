"""
Eventify — Flask Backend
Roles: host | attendee | vendor
Database: MYSQL via PyMySQL (eventify.db)
ABSA: rule-based engine from absa_engine.py
"""

import os
from datetime import datetime, date
from functools import wraps
# from single_review import score_single_review

from flask import Flask, request, jsonify, session, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash

from absa_engine import (
    analyse_text,
    compute_vendor_profile,
    SentimentEloRanker,
    process_reviews_offline,
    _build_vby_cat,
    recommend_auto,
    recommend_manual,
)

# ──────────────────────────────────────────────────────────
# App & DB setup
# ──────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    static_folder=os.path.join(BASE_DIR, "static"),
    template_folder=os.path.join(BASE_DIR, "templates"),
)
app.secret_key = os.environ.get("SECRET_KEY", "eventify-dev-secret-2026")

# --- RAILWAY MYSQL CONNECTION FIX ---
# This pulls the secret URL from Railway and fixes the driver name
db_url = os.environ.get("DATABASE_URL", "mysql+pymysql://eventify_user:eventify_password@localhost:3306/eventify_db")

if db_url and db_url.startswith("mysql://"):
    db_url = db_url.replace("mysql://", "mysql+pymysql://", 1)

app.config["SQLALCHEMY_DATABASE_URI"] = db_url
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_HTTPONLY"] = True

# Allow CORS for local testing and your production domain
CORS(app, supports_credentials=True, origins=["http://localhost:5000", "http://127.0.0.1:5000"])

# CRITICAL FIX: Initialize db HERE so the Models below can use it
db = SQLAlchemy(app)

# ──────────────────────────────────────────────────────────
# Models
# ──────────────────────────────────────────────────────────

class User(db.Model):
    __tablename__ = "users"
    id            = db.Column(db.Integer, primary_key=True)
    first_name    = db.Column(db.String(80), nullable=False)
    last_name     = db.Column(db.String(80), nullable=False)
    email         = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    role          = db.Column(db.String(20), nullable=False)   # host | attendee | vendor
    created_at    = db.Column(db.DateTime, default=datetime.utcnow)

    def set_password(self, pw):
        self.password_hash = generate_password_hash(pw)

    def check_password(self, pw):
        return check_password_hash(self.password_hash, pw)

    def to_dict(self):
        return {
            "id": self.id,
            "first_name": self.first_name,
            "last_name":  self.last_name,
            "email":      self.email,
            "role":       self.role,
        }


class Event(db.Model):
    __tablename__   = "events"
    id              = db.Column(db.Integer, primary_key=True)
    host_id         = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    name            = db.Column(db.String(200), nullable=False)
    event_type      = db.Column(db.String(80))
    event_date      = db.Column(db.String(20))
    location        = db.Column(db.String(200))
    description     = db.Column(db.Text)
    total_budget    = db.Column(db.Float, default=0.0)
    spent_budget    = db.Column(db.Float, default=0.0)
    # Customisation priorities (comma-separated, e.g. "quality,service,value")
    priorities      = db.Column(db.String(200), default="quality,service,value,reliability,ambiance,experience")
    created_at      = db.Column(db.DateTime, default=datetime.utcnow)

    host      = db.relationship("User", backref="events")
    guests    = db.relationship("Guest",   backref="event", cascade="all, delete-orphan")
    bookings  = db.relationship("Booking", backref="event", cascade="all, delete-orphan")

    def to_dict(self):
        return {
            "id":            self.id,
            "host_id":       self.host_id,
            "name":          self.name,
            "event_type":    self.event_type,
            "event_date":    self.event_date,
            "location":      self.location,
            "description":   self.description,
            "total_budget":  self.total_budget,
            "spent_budget":  self.spent_budget,
            "priorities":    self.priorities.split(",") if self.priorities else [],
            "created_at":    self.created_at.isoformat(),
        }


class Guest(db.Model):
    __tablename__ = "guests"
    id            = db.Column(db.Integer, primary_key=True)
    event_id      = db.Column(db.Integer, db.ForeignKey("events.id"), nullable=False)
    attendee_id   = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)
    name          = db.Column(db.String(120))
    email         = db.Column(db.String(120))
    rsvp_status   = db.Column(db.String(20), default="pending")  # pending | accepted | declined
    invited_at    = db.Column(db.DateTime, default=datetime.utcnow)
    responded_at  = db.Column(db.DateTime, nullable=True)

    attendee = db.relationship("User", foreign_keys=[attendee_id])

    def to_dict(self):
        return {
            "id":           self.id,
            "event_id":     self.event_id,
            "attendee_id":  self.attendee_id,
            "name":         self.name,
            "email":        self.email,
            "rsvp_status":  self.rsvp_status,
            "invited_at":   self.invited_at.isoformat(),
            "responded_at": self.responded_at.isoformat() if self.responded_at else None,
        }


class VendorProfile(db.Model):
    __tablename__  = "vendor_profiles"
    id             = db.Column(db.Integer, primary_key=True)
    user_id        = db.Column(db.Integer, db.ForeignKey("users.id"), unique=True, nullable=False)
    business_name  = db.Column(db.String(200))
    category       = db.Column(db.String(80))
    tags           = db.Column(db.String(300))
    description    = db.Column(db.Text, nullable=True)
    min_price      = db.Column(db.Float, default=0.0)
    max_price      = db.Column(db.Float, default=0.0)
    phone          = db.Column(db.String(20))
    description    = db.Column(db.Text)
    city           = db.Column(db.String(100))
    # ABSA scores
    raw_score         = db.Column(db.Float, default=0.0)
    star_rating       = db.Column(db.Float, default=3.0)
    review_count      = db.Column(db.Integer, default=0)
    aspect_quality    = db.Column(db.Float, nullable=True)
    aspect_service    = db.Column(db.Float, nullable=True)
    aspect_value      = db.Column(db.Float, nullable=True)
    aspect_ambiance   = db.Column(db.Float, nullable=True)
    aspect_reliability= db.Column(db.Float, nullable=True)
    aspect_experience = db.Column(db.Float, nullable=True)
    updated_at        = db.Column(db.DateTime, default=datetime.utcnow)

    user     = db.relationship("User", backref=db.backref("vendor_profile", uselist=False))
    reviews  = db.relationship("Review", backref="vendor", cascade="all, delete-orphan")
    inquiries= db.relationship("Inquiry", backref="vendor", cascade="all, delete-orphan")

    def to_dict(self):
        return {
            "id":                   self.id,
            "user_id":              self.user_id,
            "business_name":        self.business_name,
            "category":             self.category,
            "tags":                 self.tags.split(",") if self.tags else [],
            "min_price":            self.min_price,
            "max_price":            self.max_price,
            "phone":                self.phone,
            "description":          self.description,
            "city":                 self.city,
            "raw_score":            self.raw_score,
            "star_rating":          self.star_rating,
            "review_count":         self.review_count,
            "aspect_quality":       self.aspect_quality,
            "aspect_service":       self.aspect_service,
            "aspect_value":         self.aspect_value,
            "aspect_ambiance":      self.aspect_ambiance,
            "aspect_reliability":   self.aspect_reliability,
            "aspect_experience":    self.aspect_experience,
            "description":          self.description or "",
        }


class Review(db.Model):
    __tablename__  = "reviews"
    id             = db.Column(db.Integer, primary_key=True)
    vendor_id      = db.Column(db.Integer, db.ForeignKey("vendor_profiles.id"), nullable=False)
    reviewer_id    = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)
    event_id       = db.Column(db.Integer, db.ForeignKey("events.id"), nullable=True)
    review_text    = db.Column(db.Text, nullable=False)
    overall_rating = db.Column(db.Integer, default=5)
    # ABSA computed fields
    absa_score     = db.Column(db.Float, nullable=True)
    absa_aspects   = db.Column(db.Text, nullable=True)   # JSON string
    created_at     = db.Column(db.DateTime, default=datetime.utcnow)

    reviewer = db.relationship("User", foreign_keys=[reviewer_id])

    def to_dict(self):
        import json
        return {
            "id":             self.id,
            "vendor_id":      self.vendor_id,
            "reviewer_id":    self.reviewer_id,
            "event_id":       self.event_id,
            "review_text":    self.review_text,
            "overall_rating": self.overall_rating,
            "absa_score":     self.absa_score,
            "absa_aspects":   json.loads(self.absa_aspects) if self.absa_aspects else [],
            # THIS is where the fix goes:
            "created_at":     self.created_at.isoformat() if self.created_at else datetime.utcnow().isoformat(),
            "reviewer_name":  (f"{self.reviewer.first_name} {self.reviewer.last_name}"
                               if self.reviewer else "Anonymous"),
        }


class Inquiry(db.Model):
    __tablename__  = "inquiries"
    id             = db.Column(db.Integer, primary_key=True)
    vendor_id      = db.Column(db.Integer, db.ForeignKey("vendor_profiles.id"), nullable=False)
    host_id        = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    event_id       = db.Column(db.Integer, db.ForeignKey("events.id"), nullable=True)
    message        = db.Column(db.Text)
    budget         = db.Column(db.Float, nullable=True)
    status         = db.Column(db.String(20), default="pending")  # pending | accepted | declined
    created_at     = db.Column(db.DateTime, default=datetime.utcnow)

    host  = db.relationship("User", foreign_keys=[host_id])
    event = db.relationship("Event", foreign_keys=[event_id])

    def to_dict(self):
        return {
            "id":         self.id,
            "vendor_id":  self.vendor_id,
            "host_id":    self.host_id,
            "event_id":   self.event_id,
            "message":    self.message,
            "budget":     self.budget,
            "status":     self.status,
            "created_at": self.created_at.isoformat(),
            "host_name":  f"{self.host.first_name} {self.host.last_name}" if self.host else "",
            "event_name": self.event.name if self.event else "",
            "event_date": self.event.event_date if self.event else "",
        }


class Booking(db.Model):
    __tablename__  = "bookings"
    id             = db.Column(db.Integer, primary_key=True)
    event_id       = db.Column(db.Integer, db.ForeignKey("events.id"), nullable=False)
    vendor_id      = db.Column(db.Integer, db.ForeignKey("vendor_profiles.id"), nullable=False)
    category       = db.Column(db.String(80))
    amount         = db.Column(db.Float, default=0.0)
    status         = db.Column(db.String(20), default="confirmed")
    created_at     = db.Column(db.DateTime, default=datetime.utcnow)

    vendor = db.relationship("VendorProfile", foreign_keys=[vendor_id])

    def to_dict(self):
        return {
            "id":         self.id,
            "event_id":   self.event_id,
            "vendor_id":  self.vendor_id,
            "category":   self.category,
            "amount":     self.amount,
            "status":     self.status,
            "created_at": self.created_at.isoformat(),
            "vendor_name": self.vendor.business_name if self.vendor else "",
        }


# ──────────────────────────────────────────────────────────
# Auth helpers
# ──────────────────────────────────────────────────────────

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated


def role_required(*roles):
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            if "user_id" not in session:
                return jsonify({"error": "Unauthorized"}), 401
            if session.get("role") not in roles:
                return jsonify({"error": "Forbidden"}), 403
            return f(*args, **kwargs)
        return decorated
    return decorator


def current_user():
    return User.query.get(session["user_id"])


# ──────────────────────────────────────────────────────────
# ABSA helper
# ──────────────────────────────────────────────────────────

def _recompute_vendor_scores(vendor: VendorProfile):
    """Re-run ABSA over all stored reviews and update the vendor profile scores."""
    reviews = [r.review_text for r in vendor.reviews]
    if not reviews:
        return
    result = process_reviews_offline(vendor.business_name, reviews)
    if result:
        vendor.raw_score           = result["raw_score"]
        vendor.review_count        = result["review_count"]
        vendor.star_rating         = round(float(
            (result["raw_score"] + 1) / 2 * 5), 2)
        vendor.aspect_quality      = result.get("aspect_quality")
        vendor.aspect_service      = result.get("aspect_service")
        vendor.aspect_value        = result.get("aspect_value")
        vendor.aspect_ambiance     = result.get("aspect_ambiance")
        vendor.aspect_reliability  = result.get("aspect_reliability")
        vendor.aspect_experience   = result.get("aspect_experience")
        vendor.updated_at          = datetime.utcnow()
        db.session.commit()


# ══════════════════════════════════════════════════════════
# AUTH ROUTES
# ══════════════════════════════════════════════════════════

@app.route("/api/auth/signup", methods=["POST"])
def signup():
    data = request.json or {}
    required = ["first_name", "last_name", "email", "password", "role"]
    if not all(data.get(f) for f in required):
        return jsonify({"error": "All fields are required"}), 400
    if data["role"] not in ("host", "attendee", "vendor"):
        return jsonify({"error": "Invalid role"}), 400
    if User.query.filter_by(email=data["email"].lower()).first():
        return jsonify({"error": "Email already registered"}), 409

    user = User(
        first_name = data["first_name"].strip(),
        last_name  = data["last_name"].strip(),
        email      = data["email"].lower().strip(),
        role       = data["role"],
    )
    user.set_password(data["password"])
    db.session.add(user)
    db.session.flush()

    # Auto-create vendor profile stub if role is vendor
    if data["role"] == "vendor":
        vp = VendorProfile(user_id=user.id, business_name=f"{user.first_name}'s Business")
        db.session.add(vp)

    db.session.commit()
    session["user_id"] = user.id
    session["role"]    = user.role
    return jsonify({"user": user.to_dict()}), 201


@app.route("/api/auth/login", methods=["POST"])
def login():
    data = request.json or {}
    user = User.query.filter_by(email=(data.get("email") or "").lower().strip()).first()
    if not user or not user.check_password(data.get("password", "")):
        return jsonify({"error": "Invalid credentials"}), 401
    if data.get("role") and user.role != data["role"]:
        return jsonify({"error": f"This account is registered as '{user.role}', not '{data['role']}'"}), 403
    session["user_id"] = user.id
    session["role"]    = user.role
    return jsonify({"user": user.to_dict()})


@app.route('/api/logout', methods=['POST'])
def logout():
    # This completely wipes the server-side memory of this user
    session.clear() 
    return jsonify({"success": True, "message": "Successfully logged out"}), 200


@app.route("/api/auth/me")
@login_required
def me():
    return jsonify({"user": current_user().to_dict()})


# ══════════════════════════════════════════════════════════
# HOST — EVENTS
# ══════════════════════════════════════════════════════════

@app.route("/api/events", methods=["GET"])
@role_required("host")
def list_events():
    events = Event.query.filter_by(host_id=session["user_id"]).order_by(Event.created_at.desc()).all()
    return jsonify([e.to_dict() for e in events])


@app.route("/api/events", methods=["POST"])
@role_required("host")
def create_event():
    data = request.json or {}
    if not data.get("name"):
        return jsonify({"error": "Event name is required"}), 400
    
    total_budget = float(data.get("total_budget", 0))
    if total_budget < 0:
        return jsonify({"error": "Total budget cannot be negative"}), 400

    evt = Event(
        host_id      = session["user_id"],
        name         = data["name"],
        event_type   = data.get("event_type", ""),
        event_date   = data.get("event_date", ""),
        location     = data.get("location", ""),
        description  = data.get("description", ""),
        total_budget = total_budget,
        priorities   = ",".join(data.get("priorities", ["quality","service","value","reliability","ambiance","experience"])),
    )
    db.session.add(evt)
    db.session.commit()
    return jsonify(evt.to_dict()), 201


@app.route("/api/events/<int:eid>", methods=["GET"])
@role_required("host")
def get_event(eid):
    evt = Event.query.filter_by(id=eid, host_id=session["user_id"]).first_or_404()
    return jsonify(evt.to_dict())


@app.route("/api/events/<int:eid>", methods=["PUT"])
@role_required("host")
def update_event(eid):
    evt  = Event.query.filter_by(id=eid, host_id=session["user_id"]).first_or_404()
    data = request.json or {}
    for field in ["name", "event_type", "event_date", "location", "description"]:
        if field in data:
            setattr(evt, field, data[field])
    if "total_budget" in data:
        budget = float(data["total_budget"])
        if budget < 0:
            return jsonify({"error": "Total budget cannot be negative"}), 400
        evt.total_budget = budget
    if "priorities" in data:
        evt.priorities = ",".join(data["priorities"])
    db.session.commit()
    return jsonify(evt.to_dict())


@app.route("/api/events/<int:eid>", methods=["DELETE"])
@role_required("host")
def delete_event(eid):
    evt = Event.query.filter_by(id=eid, host_id=session["user_id"]).first_or_404()
    db.session.delete(evt)
    db.session.commit()
    return jsonify({"ok": True})


# ── Event overview stats ──

@app.route("/api/events/<int:eid>/stats")
@role_required("host")
def event_stats(eid):
    evt       = Event.query.filter_by(id=eid, host_id=session["user_id"]).first_or_404()
    guests    = Guest.query.filter_by(event_id=eid).all()
    bookings  = Booking.query.filter_by(event_id=eid).all()
    confirmed = sum(1 for g in guests if g.rsvp_status == "accepted")
    declined  = sum(1 for g in guests if g.rsvp_status == "declined")
    pending   = sum(1 for g in guests if g.rsvp_status == "pending")
    spent     = sum(b.amount for b in bookings)
    # Update spent budget
    evt.spent_budget = spent
    db.session.commit()

    # Days to event
    days_left = None
    try:
        ed = datetime.strptime(evt.event_date, "%Y-%m-%d").date()
        days_left = (ed - date.today()).days
    except Exception:
        pass

    return jsonify({
        "guests_total":     len(guests),
        "guests_confirmed": confirmed,
        "guests_declined":  declined,
        "guests_pending":   pending,
        "vendors_booked":   len(bookings),
        "total_budget":     evt.total_budget,
        "spent_budget":     spent,
        "budget_pct":       round(spent / evt.total_budget * 100, 1) if evt.total_budget else 0,
        "days_left":        days_left,
    })

@app.route("/api/reviews/analyze_live", methods=["POST"])
def analyze_live():
    data = request.json
    review_text = data.get("text", "")
    
    # Use the safe rule-based engine instead of the heavy ML model
    from absa_engine import analyse_text
    absa_results = analyse_text(review_text)
    
    if not absa_results:
        overall_sc = 0.0
    else:
        overall_sc = sum(r["weighted_score"] for r in absa_results) / len(absa_results)
    
    return jsonify({
        "raw_score": overall_sc,
        "detailed_results": absa_results
    })

# ══════════════════════════════════════════════════════════
# HOST — GUEST LIST
# ══════════════════════════════════════════════════════════

@app.route("/api/events/<int:eid>/guests", methods=["GET"])
@role_required("host")
def list_guests(eid):
    Event.query.filter_by(id=eid, host_id=session["user_id"]).first_or_404()
    guests = Guest.query.filter_by(event_id=eid).all()
    return jsonify([g.to_dict() for g in guests])


@app.route("/api/events/<int:eid>/guests", methods=["POST"])
@role_required("host")
def invite_guest(eid):
    Event.query.filter_by(id=eid, host_id=session["user_id"]).first_or_404()
    data = request.json or {}
    email = (data.get("email") or "").lower().strip()
    if not email:
        return jsonify({"error": "Email required"}), 400

    # Link to existing attendee account if email matches
    attendee = User.query.filter_by(email=email, role="attendee").first()

    # Prevent duplicate invite
    existing = Guest.query.filter_by(event_id=eid, email=email).first()
    if existing:
        return jsonify({"error": "Already invited"}), 409

    g = Guest(
        event_id    = eid,
        attendee_id = attendee.id if attendee else None,
        name        = data.get("name") or (f"{attendee.first_name} {attendee.last_name}" if attendee else email),
        email       = email,
        rsvp_status = "pending",
    )
    db.session.add(g)
    db.session.commit()
    return jsonify(g.to_dict()), 201


@app.route("/api/events/<int:eid>/guests/<int:gid>", methods=["DELETE"])
@role_required("host")
def remove_guest(eid, gid):
    Event.query.filter_by(id=eid, host_id=session["user_id"]).first_or_404()
    g = Guest.query.filter_by(id=gid, event_id=eid).first_or_404()
    db.session.delete(g)
    db.session.commit()
    return jsonify({"ok": True})


# ══════════════════════════════════════════════════════════
# HOST — BUDGET & BOOKINGS
# ══════════════════════════════════════════════════════════

@app.route("/api/events/<int:eid>/bookings", methods=["GET"])
@role_required("host")
def list_bookings(eid):
    Event.query.filter_by(id=eid, host_id=session["user_id"]).first_or_404()
    bookings = Booking.query.filter_by(event_id=eid).all()
    return jsonify([b.to_dict() for b in bookings])


@app.route("/api/events/<int:eid>/bookings", methods=["POST"])
@role_required("host")
def add_booking(eid):
    evt  = Event.query.filter_by(id=eid, host_id=session["user_id"]).first_or_404()
    data = request.json or {}
    b    = Booking(
        event_id  = eid,
        vendor_id = int(data["vendor_id"]),
        category  = data.get("category", ""),
        amount    = float(data.get("amount", 0)),
        status    = "confirmed",
    )
    db.session.add(b)
    evt.spent_budget = (evt.spent_budget or 0) + b.amount
    db.session.commit()
    return jsonify(b.to_dict()), 201


@app.route("/api/events/<int:eid>/bookings/<int:bid>", methods=["DELETE"])
@role_required("host")
def remove_booking(eid, bid):
    evt = Event.query.filter_by(id=eid, host_id=session["user_id"]).first_or_404()
    b   = Booking.query.filter_by(id=bid, event_id=eid).first_or_404()
    evt.spent_budget = max(0, (evt.spent_budget or 0) - b.amount)
    db.session.delete(b)
    db.session.commit()
    return jsonify({"ok": True})


# ══════════════════════════════════════════════════════════
# VENDOR DIRECTORY & PROFILES
# ══════════════════════════════════════════════════════════

@app.route("/api/vendors", methods=["GET"])
def list_vendors():
    q        = request.args.get("q", "").lower()
    category = request.args.get("category", "")
    city     = request.args.get("city", "")

    query = VendorProfile.query
    if category:
        query = query.filter(VendorProfile.category.ilike(f"%{category}%"))
    if city:
        query = query.filter(VendorProfile.city.ilike(f"%{city}%"))
    if q:
        query = query.filter(
            VendorProfile.business_name.ilike(f"%{q}%") |
            VendorProfile.description.ilike(f"%{q}%")
        )

    vendors = query.order_by(VendorProfile.star_rating.desc()).all()
    return jsonify([v.to_dict() for v in vendors])


@app.route("/api/vendors/ranked")
def vendors_ranked():
    vendors = VendorProfile.query.filter(VendorProfile.review_count > 0).all()
    if not vendors:
        return jsonify([])
    ranker = SentimentEloRanker()
    for v in vendors:
        ranker.add_vendor(v.business_name, {
            "raw_score":    v.raw_score,
            "review_count": v.review_count,
        })
    return jsonify(ranker.get_rankings())


@app.route("/api/vendors/<int:vid>", methods=["GET"])
def get_vendor(vid):
    v = VendorProfile.query.get_or_404(vid)
    return jsonify(v.to_dict())


@app.route("/api/vendors/me", methods=["GET"])
@role_required("vendor")
def my_vendor_profile():
    vp = VendorProfile.query.filter_by(user_id=session["user_id"]).first()
    if not vp:
        return jsonify({"error": "Profile not found"}), 404
    return jsonify(vp.to_dict())


@app.route("/api/vendors/me", methods=["PUT"])
@role_required("vendor")
def update_vendor_profile():
    vp   = VendorProfile.query.filter_by(user_id=session["user_id"]).first()
    if not vp:
        vp = VendorProfile(user_id=session["user_id"])
        db.session.add(vp)
    data = request.json or {}
    for field in ["business_name", "category", "phone", "description", "city"]:
        if field in data:
            setattr(vp, field, data[field])
    if "tags" in data:
        vp.tags = ",".join(data["tags"]) if isinstance(data["tags"], list) else data["tags"]
    if "min_price" in data:
        vp.min_price = float(data["min_price"])
    if "max_price" in data:
        vp.max_price = float(data["max_price"])
    vp.updated_at = datetime.utcnow()
    db.session.commit()
    return jsonify(vp.to_dict())


# ══════════════════════════════════════════════════════════
# REVIEWS
# ══════════════════════════════════════════════════════════

@app.route("/api/vendors/<int:vid>/reviews", methods=["GET"])
def get_reviews(vid):
    VendorProfile.query.get_or_404(vid)
    reviews = Review.query.filter_by(vendor_id=vid).order_by(Review.created_at.desc()).all()
    return jsonify([r.to_dict() for r in reviews])


@app.route("/api/vendors/<int:vid>/reviews", methods=["POST"])
@login_required
def post_review(vid):
    import json
    from absa_engine import analyse_text  # <-- ADD THIS LINE

    vp   = VendorProfile.query.get_or_404(vid)
    data = request.json or {}
    text = (data.get("review_text") or "").strip()
    if len(text) < 10:
        return jsonify({"error": "Review too short (min 10 chars)"}), 400

    # ABSA analysis (Using the lightweight Vercel-safe engine)
    absa_results = analyse_text(text)
    overall_sc   = sum(r["weighted_score"] for r in absa_results) / max(len(absa_results), 1)

    rev = Review(
        vendor_id      = vid,
        reviewer_id    = session["user_id"],
        event_id       = data.get("event_id"),
        review_text    = text,
        overall_rating = int(data.get("overall_rating", 5)),
        absa_score     = round(overall_sc, 4),
        absa_aspects   = json.dumps(absa_results),
    )
    db.session.add(rev)
    db.session.commit()

    # Recompute vendor ABSA profile
    _recompute_vendor_scores(vp)

    return jsonify({
        "review":       rev.to_dict(),
        "overall_score": round(overall_sc, 4),
        "star_rating":  round((overall_sc + 1) / 2 * 5, 2),
        "aspects":      absa_results,
    }), 201


@app.route("/api/vendors/me/reviews", methods=["GET"])
@role_required("vendor")
def my_reviews():
    vp = VendorProfile.query.filter_by(user_id=session["user_id"]).first()
    if not vp:
        return jsonify([])
    reviews = Review.query.filter_by(vendor_id=vp.id).order_by(Review.created_at.desc()).all()
    return jsonify([r.to_dict() for r in reviews])


# ══════════════════════════════════════════════════════════
# INQUIRIES
# ══════════════════════════════════════════════════════════

@app.route("/api/vendors/<int:vid>/inquiries", methods=["POST"])
@role_required("host")
def send_inquiry(vid):
    VendorProfile.query.get_or_404(vid)
    data = request.json or {}
    inq  = Inquiry(
        vendor_id = vid,
        host_id   = session["user_id"],
        event_id  = data.get("event_id"),
        message   = data.get("message", ""),
        budget    = float(data["budget"]) if data.get("budget") else None,
        status    = "pending",
    )
    db.session.add(inq)
    db.session.commit()
    return jsonify(inq.to_dict()), 201


@app.route("/api/vendors/me/inquiries", methods=["GET"])
@role_required("vendor")
def my_inquiries():
    vp = VendorProfile.query.filter_by(user_id=session["user_id"]).first()
    if not vp:
        return jsonify([])
    inqs = Inquiry.query.filter_by(vendor_id=vp.id).order_by(Inquiry.created_at.desc()).all()
    return jsonify([i.to_dict() for i in inqs])


@app.route("/api/inquiries/<int:iid>", methods=["PUT"])
@role_required("vendor")
def update_inquiry(iid):
    vp  = VendorProfile.query.filter_by(user_id=session["user_id"]).first_or_404()
    inq = Inquiry.query.filter_by(id=iid, vendor_id=vp.id).first_or_404()
    data = request.json or {}
    if data.get("status") in ("accepted", "declined", "pending"):
        inq.status = data["status"]
    db.session.commit()
    return jsonify(inq.to_dict())


# ══════════════════════════════════════════════════════════
# ATTENDEE
# ══════════════════════════════════════════════════════════

@app.route("/api/attendee/invites", methods=["GET"])
@role_required("attendee")
def attendee_invites():
    user   = current_user()
    guests = Guest.query.filter_by(attendee_id=user.id).all()
    result = []
    for g in guests:
        evt  = Event.query.get(g.event_id)
        host = User.query.get(evt.host_id) if evt else None
        result.append({
            **g.to_dict(),
            "event_name":   evt.name if evt else "",
            "event_date":   evt.event_date if evt else "",
            "event_location": evt.location if evt else "",
            "host_name":    f"{host.first_name} {host.last_name}" if host else "",
        })
    return jsonify(result)


@app.route("/api/attendee/invites/<int:gid>", methods=["PUT"])
@role_required("attendee")
def respond_invite(gid):
    user = current_user()
    g    = Guest.query.filter_by(id=gid, attendee_id=user.id).first_or_404()
    data = request.json or {}
    if data.get("status") in ("accepted", "declined"):
        g.rsvp_status  = data["status"]
        g.responded_at = datetime.utcnow()
    db.session.commit()
    return jsonify(g.to_dict())


@app.route("/api/attendee/schedule", methods=["GET"])
@role_required("attendee")
def attendee_schedule():
    user   = current_user()
    guests = Guest.query.filter_by(attendee_id=user.id, rsvp_status="accepted").all()
    result = []
    for g in guests:
        evt = Event.query.get(g.event_id)
        if not evt:
            continue
        host = User.query.get(evt.host_id)
        result.append({
            **evt.to_dict(),
            "host_name": f"{host.first_name} {host.last_name}" if host else "",
        })
    return jsonify(result)


# ══════════════════════════════════════════════════════════
# ABSA — standalone analyse endpoint
# ══════════════════════════════════════════════════════════

@app.route("/api/analyze", methods=["POST"])
@login_required
def analyze_text():
    data = request.json or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "text required"}), 400
    results     = analyse_text(text)
    overall_sc  = sum(r["weighted_score"] for r in results) / max(len(results), 1)
    return jsonify({
        "aspects":       results,
        "overall_score": round(overall_sc, 4),
        "star_rating":   round((overall_sc + 1) / 2 * 5, 2),
    })


# ══════════════════════════════════════════════════════════
# BUDGET PLANNER — recommend endpoint
# ══════════════════════════════════════════════════════════

@app.route("/api/budget/recommend", methods=["POST"])
@role_required("host")
def budget_recommend():
    data = request.json or {}
    mode         = data.get("mode", "auto")
    total_budget = float(data.get("total_budget", 0))
    vendors_raw  = data.get("vendors", [])
    vby_cat      = _build_vby_cat(vendors_raw)

    if mode == "auto":
        priority_order = data.get("priority_order", list(vby_cat.keys()))
        lam            = float(data.get("lam", 2.0))
        result         = recommend_auto(total_budget, priority_order, vby_cat, lam)
    else:
        cat_pct     = data.get("cat_pct", {})
        reallocate  = bool(data.get("reallocate", True))
        result      = recommend_manual(total_budget, cat_pct, vby_cat, reallocate)

    return jsonify(result)


# ══════════════════════════════════════════════════════════
# SERVE FRONTEND
# ══════════════════════════════════════════════════════════

@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "eventify_working.html")


# ══════════════════════════════════════════════════════════
# DB INIT + SEED
# ══════════════════════════════════════════════════════════

def seed_demo_data():
    """Add demo vendors if DB is empty."""
    if VendorProfile.query.count() > 0:
        return

    demo_vendors = [
        ("Priya", "Sharma",  "vendor", "priya@demovendor.com",  "Delight Caterers",  "catering",     "Catering,Veg,Non-Veg",  450, 900,  "9035290001", "Award-winning Maharashtrian catering.",          "Pune",  0.72, 4.4, 18),
        ("Arjun", "Mehta",   "vendor", "arjun@demovendor.com",  "Focus Studios",     "photography",  "Photography,Reels",     400, 800,  "9035290002", "Candid wedding photography specialists.",         "Mumbai",0.65, 4.1, 24),
        ("Neha",  "Joshi",   "vendor", "neha@demovendor.com",   "Bloom Decor",       "decoration",   "Decoration,Floral",     300, 700,  "9035290003", "Floral fantasy for every occasion.",             "Pune",  0.58, 3.9, 12),
        ("Rohan", "Patil",   "vendor", "rohan@demovendor.com",  "DJ Beats Pro",      "entertainment","DJ,Sound,Lighting",    250, 600,  "9035290004", "High-energy DJ sets for weddings and parties.",  "Nashik",0.60, 4.0, 15),
        ("Anita", "Kulkarni","vendor", "anita@demovendor.com",  "Grand Vistas Hall", "venue",        "Venue,AC,Parking",     800, 2000, "9035290005", "Banquet hall with panoramic city views.",        "Pune",  0.80, 4.6, 31),
        ("Suresh","Iyer",    "vendor", "suresh@demovendor.com", "Silver Frame Photo","photography",  "Photography,Albums",   350, 700,  "9035290006", "Timeless portraits and wedding albums.",         "Mumbai",0.50, 3.5, 9),
        ("Kavya", "Nair",    "vendor", "kavya@demovendor.com",  "Spice Route Caters","catering",     "Catering,South Indian",380, 850,  "9035290007", "Authentic South Indian wedding feasts.",         "Pune",  0.62, 4.1, 14),
    ]

    for fn, ln, role, email, biz, cat, tags, mn, mx, ph, desc, city, raw, star, rc in demo_vendors:
        user = User(first_name=fn, last_name=ln, email=email, role=role)
        user.set_password("demo1234")
        db.session.add(user)
        db.session.flush()
        vp = VendorProfile(
            user_id=user.id, business_name=biz, category=cat, tags=tags,
            min_price=mn, max_price=mx, phone=ph, description=desc, city=city,
            raw_score=raw, star_rating=star, review_count=rc,
        )
        db.session.add(vp)

    db.session.commit()
    print("[seed] Demo vendor data inserted.")


#with app.app_context():
#    db.create_all()
#    seed_demo_data()


if __name__ == "__main__":
    app.run(debug=True, port=5000)
