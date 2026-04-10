from datetime import datetime, timedelta
import random
from app import app, db, User, Event, Guest, Booking, Inquiry, VendorProfile

def populate_userhost_data():
    with app.app_context():
        print("Preparing Hero Demo Data...")

        # 1. Setup the specific Host Account
        host_email = "userhost@gmail.com"
        host = User.query.filter_by(email=host_email).first()
        
        if not host:
            host = User(first_name="Master", last_name="Host", email=host_email, role="host")
            host.set_password("Password")
            db.session.add(host)
            db.session.flush()
            print(f"Created new host account: {host_email}")
        else:
            host.set_password("Password") # Force reset the password to exactly what you requested
            print(f"Found existing host account: {host_email}. Password updated.")
        
        db.session.commit()

        # 2. Grab a few of your REAL vendors to attach bookings/inquiries to
        vendors = VendorProfile.query.limit(5).all()
        if len(vendors) < 4:
            print("WARNING: You need at least 4 vendors in the DB for bookings to populate.")
            return

        # 3. Create Events
        event1 = Event.query.filter_by(name="The Grand Summer Wedding", host_id=host.id).first()
        if not event1:
            event1 = Event(
                host_id=host.id,
                name="The Grand Summer Wedding",
                event_type="Wedding",
                event_date=(datetime.utcnow() + timedelta(days=60)).strftime("%Y-%m-%d"),
                location="Lavale, Maharashtra",
                total_budget=1500000.0,
                description="A massive summer celebration needing top-tier catering and decor."
            )
            db.session.add(event1)

        event2 = Event.query.filter_by(name="Tech Startup Gala 2026", host_id=host.id).first()
        if not event2:
            event2 = Event(
                host_id=host.id,
                name="Tech Startup Gala 2026",
                event_type="Corporate",
                event_date=(datetime.utcnow() + timedelta(days=15)).strftime("%Y-%m-%d"),
                location="Pune City Center",
                total_budget=500000.0,
                description="Annual networking event for investors and founders."
            )
            db.session.add(event2)

        db.session.flush()

        # 4. Create Guests (Mixed RSVPs to make the dashboard charts look active)
        if not Guest.query.filter_by(event_id=event1.id).first():
            guest_data = [
                ("Aisha Patel", "aisha.p@example.com", "accepted"),
                ("Rahul Sharma", "rahul.s@example.com", "pending"),
                ("Karan Malhotra", "karan.m@example.com", "declined"),
                ("Priya Singh", "priya.singh@example.com", "accepted"),
                ("Vikram Desai", "vikram.d@example.com", "pending"),
            ]
            for g_name, g_email, g_status in guest_data:
                db.session.add(Guest(
                    event_id=event1.id, name=g_name, email=g_email, 
                    rsvp_status=g_status, invited_at=datetime.utcnow() - timedelta(days=random.randint(1, 5))
                ))
            
            # A few guests for the Gala
            db.session.add(Guest(event_id=event2.id, name="Neha Gupta", email="neha.g@example.com", rsvp_status="accepted"))
            db.session.add(Guest(event_id=event2.id, name="Rohan Joshi", email="rohan.j@example.com", rsvp_status="pending"))

        # 5. Create Bookings (Attach actual vendors to Event 1's budget)
        if not Booking.query.filter_by(event_id=event1.id).first():
            v1, v2 = vendors[0], vendors[1]
            
            b1 = Booking(event_id=event1.id, vendor_id=v1.id, category=v1.category or "venue", amount=450000.0, status="confirmed")
            b2 = Booking(event_id=event1.id, vendor_id=v2.id, category=v2.category or "catering", amount=200000.0, status="confirmed")
            db.session.add(b1)
            db.session.add(b2)
            
            # Update the event's spent budget to reflect the bookings
            event1.spent_budget = 650000.0

        # 6. Create Inquiries (Host checking out other vendors for the Gala)
        if not Inquiry.query.filter_by(host_id=host.id).first():
            v3, v4 = vendors[2], vendors[3]
            
            db.session.add(Inquiry(
                vendor_id=v3.id, host_id=host.id, event_id=event2.id, 
                message=f"Hi {v3.business_name}, we are looking for services for a corporate gala. Are you available on our dates?", 
                budget=80000.0, status="pending"
            ))
            db.session.add(Inquiry(
                vendor_id=v4.id, host_id=host.id, event_id=event1.id, 
                message=f"Hello! We need premium {v4.category} for a large wedding. We love your ABSA ratings.", 
                budget=120000.0, status="accepted"
            ))

        db.session.commit()
        print("Success! Your Hero Account is fully populated and ready for the demo.")

if __name__ == '__main__':
    populate_userhost_data()