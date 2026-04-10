import os
import mysql.connector
from mysql.connector import pooling

_pool = None

def get_pool():
    global _pool
    if _pool is None:
        _pool = pooling.MySQLConnectionPool(
            pool_name="eventify",
            pool_size=5,
            host=os.environ.get("DB_HOST", "localhost"),
            port=int(os.environ.get("DB_PORT", 3306)),
            user=os.environ.get("DB_USER", "eventify_user"),
            password=os.environ.get("DB_PASSWORD", "eventify_password"),
            database=os.environ.get("DB_NAME", "eventify_db"),
            autocommit=True,
        )
    return _pool


def get_conn():
    return get_pool().get_connection()


def init_schema():
    conn = get_conn()
    cur  = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS vendors (
            id       INT PRIMARY KEY AUTO_INCREMENT,
            name     VARCHAR(255) NOT NULL,
            category VARCHAR(100) NOT NULL,
            INDEX idx_category (category)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS vendor_details (
            vendor_id INT PRIMARY KEY,
            price     DECIMAL(12,2) DEFAULT 0,
            location  VARCHAR(255)  DEFAULT '',
            contact   VARCHAR(255)  DEFAULT '',
            FOREIGN KEY (vendor_id) REFERENCES vendors(id) ON DELETE CASCADE
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS vendor_scores (
            vendor_id        INT PRIMARY KEY,
            elo_score        FLOAT DEFAULT 1500,
            raw_score        FLOAT DEFAULT 0,
            star_rating      FLOAT DEFAULT 3,
            review_count     INT   DEFAULT 0,
            aspect_quality   FLOAT DEFAULT NULL,
            aspect_service   FLOAT DEFAULT NULL,
            aspect_value     FLOAT DEFAULT NULL,
            aspect_ambiance  FLOAT DEFAULT NULL,
            aspect_reliability FLOAT DEFAULT NULL,
            aspect_experience  FLOAT DEFAULT NULL,
            FOREIGN KEY (vendor_id) REFERENCES vendors(id) ON DELETE CASCADE
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS vendor_reviews (
            id         INT PRIMARY KEY AUTO_INCREMENT,
            vendor_id  INT  NOT NULL,
            review     TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_vendor (vendor_id),
            FOREIGN KEY (vendor_id) REFERENCES vendors(id) ON DELETE CASCADE
        )
    """)

    cur.close()
    conn.close()
