-- Run as MySQL root:
--   mysql -u root -p < setup.sql

CREATE DATABASE IF NOT EXISTS eventify_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

CREATE USER IF NOT EXISTS 'eventify_user'@'localhost' IDENTIFIED BY 'eventify_password';
GRANT ALL PRIVILEGES ON eventify_db.* TO 'eventify_user'@'localhost';
FLUSH PRIVILEGES;

USE eventify_db;

-- Core tables (also created automatically by db.py on first run)

CREATE TABLE IF NOT EXISTS vendors (
    id       INT PRIMARY KEY AUTO_INCREMENT,
    name     VARCHAR(255) NOT NULL,
    category VARCHAR(100) NOT NULL,
    INDEX idx_category (category)
);

CREATE TABLE IF NOT EXISTS vendor_details (
    vendor_id INT PRIMARY KEY,
    price     DECIMAL(12,2) DEFAULT 0,
    location  VARCHAR(255)  DEFAULT '',
    contact   VARCHAR(255)  DEFAULT '',
    FOREIGN KEY (vendor_id) REFERENCES vendors(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS vendor_scores (
    vendor_id          INT PRIMARY KEY,
    elo_score          FLOAT DEFAULT 1500,
    raw_score          FLOAT DEFAULT 0,
    star_rating        FLOAT DEFAULT 3,
    review_count       INT   DEFAULT 0,
    aspect_quality     FLOAT DEFAULT NULL,
    aspect_service     FLOAT DEFAULT NULL,
    aspect_value       FLOAT DEFAULT NULL,
    aspect_ambiance    FLOAT DEFAULT NULL,
    aspect_reliability FLOAT DEFAULT NULL,
    aspect_experience  FLOAT DEFAULT NULL,
    FOREIGN KEY (vendor_id) REFERENCES vendors(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS vendor_reviews (
    id         INT PRIMARY KEY AUTO_INCREMENT,
    vendor_id  INT  NOT NULL,
    review     TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_vendor (vendor_id),
    FOREIGN KEY (vendor_id) REFERENCES vendors(id) ON DELETE CASCADE
);

-- Sample vendor data

INSERT IGNORE INTO vendors (id, name, category) VALUES
  (1,  'Grand Ballroom',        'venue'),
  (2,  'The Garden Estate',     'venue'),
  (3,  'City View Hall',        'venue'),
  (4,  'Elite Cuisine Co.',     'catering'),
  (5,  'Spice Garden Caterers', 'catering'),
  (6,  'Homestyle Feast',       'catering'),
  (7,  'Pixel Perfect Studios', 'photography'),
  (8,  'Capture Moments',       'photography'),
  (9,  'Floral Fantasy',        'decoration'),
  (10, 'Harmony Live Band',     'entertainment');

INSERT IGNORE INTO vendor_details (vendor_id, price, location, contact) VALUES
  (1,  120000, 'Mumbai',    'grand@ballroom.com'),
  (2,   85000, 'Pune',      'garden@estate.com'),
  (3,   60000, 'Delhi',     'cityview@hall.com'),
  (4,   95000, 'Mumbai',    'elite@cuisine.com'),
  (5,   60000, 'Pune',      'spice@garden.com'),
  (6,   40000, 'Nashik',    'homestyle@feast.com'),
  (7,   55000, 'Mumbai',    'pixel@studios.com'),
  (8,   38000, 'Pune',      'capture@moments.com'),
  (9,   45000, 'Delhi',     'floral@fantasy.com'),
  (10,  50000, 'Nashik',    'harmony@band.com');

-- Seed initial ABSA scores (would normally come from offline ingestion)
INSERT IGNORE INTO vendor_scores
    (vendor_id, raw_score, review_count, aspect_quality, aspect_service,
     aspect_value, aspect_ambiance, aspect_reliability, aspect_experience)
VALUES
  (1,  0.78, 42,  0.80, 0.75, 0.60, 0.85, 0.70, 0.82),
  (2,  0.61, 35,  0.65, 0.58, 0.70, 0.72, 0.55, 0.60),
  (3,  0.44, 28,  0.45, 0.40, 0.55, 0.48, 0.42, 0.44),
  (4,  0.82, 58,  0.90, 0.80, 0.65, NULL, 0.78, 0.85),
  (5,  0.55, 44,  0.60, 0.52, 0.68, NULL, 0.50, 0.58),
  (6,  0.41, 31,  0.45, 0.38, 0.60, NULL, 0.40, 0.42),
  (7,  0.88, 63,  0.92, 0.85, 0.70, NULL, 0.88, 0.90),
  (8,  0.67, 47,  0.72, 0.65, 0.68, NULL, 0.65, 0.70),
  (9,  0.75, 38,  0.80, 0.70, 0.62, 0.82, 0.72, 0.78),
  (10, 0.80, 41,  0.85, 0.78, 0.60, NULL, 0.80, 0.83);
