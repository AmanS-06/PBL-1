from datetime import datetime, timedelta
import json
from app import app, db, User, Event, Guest, Inquiry, VendorProfile, Review

def populate_attendee_and_vendor():
    with app.app_context():
        print("Preparing Attendee & Vendor Demo Data...")

        # 1. Setup the Attendee Account
        attendee_email = "attendeeuser@gmail.com"
        attendee = User.query.filter_by(email=attendee_email).first()
        if not attendee:
            attendee = User(first_name="Demo", last_name="Attendee", email=attendee_email, role="attendee")
            db.session.add(attendee)
            print(f"Created new attendee: {attendee_email}")
        
        attendee.set_password("Password")
        db.session.flush()

        # 2. Setup the Vendor Account
        vendor_email = "vendoruser@gmail.com"
        vendor_user = User.query.filter_by(email=vendor_email).first()
        if not vendor_user:
            vendor_user = User(first_name="Demo", last_name="Vendor", email=vendor_email, role="vendor")
            db.session.add(vendor_user)
            print(f"Created new vendor: {vendor_email}")
            
        vendor_user.set_password("Password")
        db.session.flush()

        # 3. Setup the Vendor Profile
        vp = VendorProfile.query.filter_by(user_id=vendor_user.id).first()
        if not vp:
            vp = VendorProfile(
                user_id=vendor_user.id,
                business_name="Elite Captured Moments",
                category="photography",
                tags="Photography,Drone,Pre-wedding",
                min_price=50000.0,
                max_price=150000.0,
                phone="9876543210",
                city="Pune",
                description="Award-winning premium photography capturing your best moments.",
                raw_score=0.85,
                star_rating=4.8,
                review_count=12,
                aspect_quality=0.9,
                aspect_service=0.8,
                aspect_value=0.7,
                aspect_reliability=0.85
            )
            db.session.add(vp)
        db.session.flush()

        # 4. Create Background Events (Owned by a dummy host) to anchor the data
        dummy_host = User.query.filter_by(role="host").first()
        if not dummy_host:
            dummy_host = User(first_name="Dummy", last_name="Host", email="dummy.host@example.com", role="host")
            dummy_host.set_password("Password")
            db.session.add(dummy_host)
            db.session.flush()

        event_future = Event(host_id=dummy_host.id, name="Aditi & Rahul Wedding", event_type="Wedding", event_date=(datetime.utcnow() + timedelta(days=30)).strftime("%Y-%m-%d"), location="Pune", total_budget=2000000)
        event_pending = Event(host_id=dummy_host.id, name="Annual Tech Meetup", event_type="Corporate", event_date=(datetime.utcnow() + timedelta(days=45)).strftime("%Y-%m-%d"), location="Mumbai", total_budget=500000)
        event_past = Event(host_id=dummy_host.id, name="Riya's Birthday Bash", event_type="Birthday", event_date=(datetime.utcnow() - timedelta(days=10)).strftime("%Y-%m-%d"), location="Pune", total_budget=100000)
        
        db.session.add_all([event_future, event_pending, event_past])
        db.session.flush()

        # 5. Populate Attendee RSVPs
        if not Guest.query.filter_by(attendee_id=attendee.id).first():
            db.session.add(Guest(event_id=event_future.id, attendee_id=attendee.id, name=f"{attendee.first_name} {attendee.last_name}", email=attendee.email, rsvp_status="accepted", responded_at=datetime.utcnow()))
            db.session.add(Guest(event_id=event_pending.id, attendee_id=attendee.id, name=f"{attendee.first_name} {attendee.last_name}", email=attendee.email, rsvp_status="pending"))
            db.session.add(Guest(event_id=event_past.id, attendee_id=attendee.id, name=f"{attendee.first_name} {attendee.last_name}", email=attendee.email, rsvp_status="accepted", responded_at=datetime.utcnow() - timedelta(days=20)))

        # 6. Populate Vendor Inquiries
        if not Inquiry.query.filter_by(vendor_id=vp.id).first():
            db.session.add(Inquiry(vendor_id=vp.id, host_id=dummy_host.id, event_id=event_future.id, message="Hi, are you available for a 2-day wedding shoot in Pune?", budget=100000, status="pending"))
            db.session.add(Inquiry(vendor_id=vp.id, host_id=dummy_host.id, event_id=event_pending.id, message="Need corporate event coverage. Please confirm.", budget=40000, status="accepted"))

        # 7. Populate Vendor Reviews (With ABSA data for the charts)
        if not Review.query.filter_by(vendor_id=vp.id).first():
            absa_mock_1 = json.dumps([{"aspect": "quality", "sentiment": "positive", "confidence": 0.95, "weighted_score": 0.95, "probs": {"positive": 0.95, "negative": 0.05, "neutral": 0.0, "conflict": 0.0}}])
            absa_mock_2 = json.dumps([{"aspect": "reliability", "sentiment": "negative", "confidence": 0.8, "weighted_score": -0.8, "probs": {"positive": 0.1, "negative": 0.8, "neutral": 0.1, "conflict": 0.0}}])
            
            db.session.add(Review(vendor_id=vp.id, reviewer_id=dummy_host.id, event_id=event_past.id, review_text="Absolutely stunning photos! The team was highly professional and captured every moment perfectly.", overall_rating=5, absa_score=0.95, absa_aspects=absa_mock_1))
            db.session.add(Review(vendor_id=vp.id, review_text="Good photos, but they arrived a bit late to the venue which caused panic.", overall_rating=3, absa_score=0.1, absa_aspects=absa_mock_2))

        db.session.commit()
        print("Success! Your Attendee and Vendor Accounts are fully populated.")

if __name__ == '__main__':
    populate_attendee_and_vendor()