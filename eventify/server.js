// ===============================================
// EVENTIFY - BACKEND SERVER
// Node.js + Express + MySQL + Stripe
// ===============================================

const express = require('express');
const cors = require('cors');
const dotenv = require('dotenv');
const mysql = require('mysql2/promise');
const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');
const multer = require('multer');
const cloudinary = require('cloudinary').v2;
const path = require('path');
const stripe = require('stripe')(process.env.STRIPE_SECRET_KEY || 'sk_test_dummy');

// Load environment variables
dotenv.config();

// ABSA microservice URL
const ABSA_URL = process.env.ABSA_URL || 'http://localhost:8000';

const app = express();

// ===============================================
// MIDDLEWARE
// ===============================================
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ limit: '50mb', extended: true }));
app.use(cors({
  origin: process.env.FRONTEND_URL || ['http://localhost:3000', 'http://localhost:5173'],
  credentials: true
}));

// ===============================================
// DATABASE CONFIGURATION
// ===============================================
const pool = mysql.createPool({
  host: process.env.DB_HOST || 'localhost',
  user: process.env.DB_USER || 'root',
  password: process.env.DB_PASSWORD || '',
  database: process.env.DB_NAME || 'eventify_db',
  waitForConnections: true,
  connectionLimit: 10,
  queueLimit: 0
});

// ===============================================
// CLOUDINARY CONFIGURATION (Image Storage)
// ===============================================
cloudinary.config({
  cloud_name: process.env.CLOUDINARY_CLOUD_NAME || 'demo',
  api_key: process.env.CLOUDINARY_API_KEY || 'demo',
  api_secret: process.env.CLOUDINARY_API_SECRET || 'demo'
});

// ===============================================
// MULTER CONFIGURATION (File Upload)
// ===============================================
const storage = multer.memoryStorage();
const upload = multer({ 
  storage,
  limits: { fileSize: 10 * 1024 * 1024 } // 10MB
});

// ===============================================
// HELPER FUNCTIONS
// ===============================================

// Generate JWT Token
const generateToken = (userId, role) => {
  return jwt.sign({ userId, role }, process.env.JWT_SECRET || 'eventify-secret-key', { 
    expiresIn: '30d' 
  });
};

// Middleware to verify JWT
const authMiddleware = async (req, res, next) => {
  try {
    const token = req.headers.authorization?.split(' ')[1];
    if (!token) return res.status(401).json({ error: 'No token provided' });
    
    const decoded = jwt.verify(token, process.env.JWT_SECRET || 'eventify-secret-key');
    req.userId = decoded.userId;
    req.userRole = decoded.role;
    next();
  } catch (error) {
    res.status(401).json({ error: 'Invalid token' });
  }
};

// Upload image to Cloudinary
const uploadToCloudinary = async (fileBuffer, folderName) => {
  return new Promise((resolve, reject) => {
    const stream = cloudinary.uploader.upload_stream(
      { folder: `eventify/${folderName}`, resource_type: 'auto' },
      (error, result) => {
        if (error) reject(error);
        else resolve(result);
      }
    );
    stream.end(fileBuffer);
  });
};

// ===============================================
// INITIALIZE DATABASE SCHEMA
// ===============================================

async function initializeDatabase() {
  const connection = await pool.getConnection();
  try {
    // Create Users Table
    await connection.execute(`
      CREATE TABLE IF NOT EXISTS users (
        user_id INT PRIMARY KEY AUTO_INCREMENT,
        email VARCHAR(255) UNIQUE NOT NULL,
        password_hash VARCHAR(255) NOT NULL,
        full_name VARCHAR(255) NOT NULL,
        phone VARCHAR(20),
        role ENUM('vendor', 'host', 'admin') NOT NULL,
        profile_image VARCHAR(500),
        business_name VARCHAR(255),
        location VARCHAR(500),
        description TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        INDEX idx_email (email),
        INDEX idx_role (role)
      )
    `);

    // Create Products Table
    await connection.execute(`
      CREATE TABLE IF NOT EXISTS products (
        product_id INT PRIMARY KEY AUTO_INCREMENT,
        vendor_id INT NOT NULL,
        name VARCHAR(255) NOT NULL,
        description TEXT,
        price DECIMAL(10, 2) NOT NULL,
        category VARCHAR(100),
        image_url VARCHAR(500),
        availability BOOLEAN DEFAULT TRUE,
        min_quantity INT DEFAULT 1,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        FOREIGN KEY (vendor_id) REFERENCES users(user_id) ON DELETE CASCADE,
        INDEX idx_vendor_id (vendor_id),
        INDEX idx_category (category)
      )
    `);

    // Create Events Table
    await connection.execute(`
      CREATE TABLE IF NOT EXISTS events (
        event_id INT PRIMARY KEY AUTO_INCREMENT,
        host_id INT NOT NULL,
        title VARCHAR(255) NOT NULL,
        description TEXT,
        event_date DATETIME NOT NULL,
        location VARCHAR(500),
        guest_count INT,
        budget DECIMAL(12, 2),
        theme VARCHAR(100),
        status ENUM('planning', 'confirmed', 'completed', 'cancelled') DEFAULT 'planning',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        FOREIGN KEY (host_id) REFERENCES users(user_id) ON DELETE CASCADE,
        INDEX idx_host_id (host_id),
        INDEX idx_event_date (event_date),
        INDEX idx_status (status)
      )
    `);

    // Create RSVPs Table
    await connection.execute(`
      CREATE TABLE IF NOT EXISTS rsvps (
        rsvp_id INT PRIMARY KEY AUTO_INCREMENT,
        event_id INT NOT NULL,
        guest_name VARCHAR(255) NOT NULL,
        guest_email VARCHAR(255),
        guest_phone VARCHAR(20),
        status ENUM('invited', 'accepted', 'declined', 'maybe') DEFAULT 'invited',
        guest_count INT DEFAULT 1,
        dietary_requirements TEXT,
        response_date DATETIME,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (event_id) REFERENCES events(event_id) ON DELETE CASCADE,
        INDEX idx_event_id (event_id),
        INDEX idx_status (status),
        UNIQUE KEY unique_rsvp (event_id, guest_email)
      )
    `);

    // Create Invites Table
    await connection.execute(`
      CREATE TABLE IF NOT EXISTS invites (
        invite_id INT PRIMARY KEY AUTO_INCREMENT,
        event_id INT NOT NULL,
        invite_token VARCHAR(255) UNIQUE NOT NULL,
        invite_email VARCHAR(255),
        status ENUM('pending', 'sent', 'opened') DEFAULT 'pending',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        expires_at DATETIME,
        FOREIGN KEY (event_id) REFERENCES events(event_id) ON DELETE CASCADE,
        INDEX idx_event_id (event_id),
        INDEX idx_token (invite_token),
        INDEX idx_expires_at (expires_at)
      )
    `);

    // Create Vendor Recommendations Table
    await connection.execute(`
      CREATE TABLE IF NOT EXISTS vendor_recommendations (
        recommendation_id INT PRIMARY KEY AUTO_INCREMENT,
        event_id INT NOT NULL,
        vendor_id INT NOT NULL,
        match_score DECIMAL(3, 2),
        reason TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (event_id) REFERENCES events(event_id) ON DELETE CASCADE,
        FOREIGN KEY (vendor_id) REFERENCES users(user_id) ON DELETE CASCADE,
        INDEX idx_event_id (event_id),
        INDEX idx_vendor_id (vendor_id),
        INDEX idx_match_score (match_score)
      )
    `);

    // Create Order Items Table
    await connection.execute(`
      CREATE TABLE IF NOT EXISTS order_items (
        order_item_id INT PRIMARY KEY AUTO_INCREMENT,
        event_id INT NOT NULL,
        product_id INT NOT NULL,
        vendor_id INT NOT NULL,
        quantity INT NOT NULL,
        unit_price DECIMAL(10, 2) NOT NULL,
        total_price DECIMAL(12, 2) NOT NULL,
        status ENUM('pending', 'confirmed', 'delivered', 'cancelled') DEFAULT 'pending',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        FOREIGN KEY (event_id) REFERENCES events(event_id) ON DELETE CASCADE,
        FOREIGN KEY (product_id) REFERENCES products(product_id) ON DELETE RESTRICT,
        FOREIGN KEY (vendor_id) REFERENCES users(user_id) ON DELETE RESTRICT,
        INDEX idx_event_id (event_id),
        INDEX idx_vendor_id (vendor_id),
        INDEX idx_status (status)
      )
    `);

    // Create To-Do Items Table
    await connection.execute(`
      CREATE TABLE IF NOT EXISTS todo_items (
        todo_id INT PRIMARY KEY AUTO_INCREMENT,
        event_id INT NOT NULL,
        title VARCHAR(255) NOT NULL,
        description TEXT,
        category VARCHAR(100),
        priority ENUM('low', 'medium', 'high', 'urgent') DEFAULT 'medium',
        completed BOOLEAN DEFAULT FALSE,
        due_date DATETIME,
        assigned_to INT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        FOREIGN KEY (event_id) REFERENCES events(event_id) ON DELETE CASCADE,
        FOREIGN KEY (assigned_to) REFERENCES users(user_id),
        INDEX idx_event_id (event_id),
        INDEX idx_completed (completed),
        INDEX idx_due_date (due_date)
      )
    `);

    // Create Payments Table
    await connection.execute(`
      CREATE TABLE IF NOT EXISTS payments (
        payment_id INT PRIMARY KEY AUTO_INCREMENT,
        event_id INT NOT NULL,
        vendor_id INT NOT NULL,
        amount DECIMAL(12, 2) NOT NULL,
        status ENUM('pending', 'completed', 'failed', 'refunded') DEFAULT 'pending',
        stripe_payment_id VARCHAR(255),
        payment_method VARCHAR(50),
        transaction_date DATETIME,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (event_id) REFERENCES events(event_id) ON DELETE CASCADE,
        FOREIGN KEY (vendor_id) REFERENCES users(user_id) ON DELETE RESTRICT,
        INDEX idx_event_id (event_id),
        INDEX idx_vendor_id (vendor_id),
        INDEX idx_status (status),
        INDEX idx_stripe_payment_id (stripe_payment_id)
      )
    `);

    // Create Budget Tracking Table
    await connection.execute(`
      CREATE TABLE IF NOT EXISTS budget_tracking (
        budget_id INT PRIMARY KEY AUTO_INCREMENT,
        event_id INT NOT NULL,
        category VARCHAR(100),
        budgeted_amount DECIMAL(12, 2),
        spent_amount DECIMAL(12, 2) DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        FOREIGN KEY (event_id) REFERENCES events(event_id) ON DELETE CASCADE,
        INDEX idx_event_id (event_id),
        UNIQUE KEY unique_category (event_id, category)
      )
    `);

    console.log('✓ Database initialized successfully');
  } catch (error) {
    console.error('Error initializing database:', error);
  } finally {
    connection.release();
  }
}

// ===============================================
// AUTHENTICATION ROUTES
// ===============================================

// Register User
app.post('/api/auth/register', async (req, res) => {
  const { email, password, fullName, phone, role, businessName, location } = req.body;
  
  try {
    const connection = await pool.getConnection();
    
    // Check if user exists
    const [existing] = await connection.execute('SELECT user_id FROM users WHERE email = ?', [email]);
    if (existing.length > 0) {
      connection.release();
      return res.status(400).json({ error: 'Email already registered' });
    }
    
    // Hash password
    const passwordHash = await bcrypt.hash(password, 10);
    
    // Create user
    const [result] = await connection.execute(
      'INSERT INTO users (email, password_hash, full_name, phone, role, business_name, location) VALUES (?, ?, ?, ?, ?, ?, ?)',
      [email, passwordHash, fullName, phone, role, businessName, location]
    );
    
    const userId = result.insertId;
    const token = generateToken(userId, role);
    
    connection.release();
    
    res.json({ 
      success: true, 
      token, 
      user: { userId, email, fullName, role } 
    });
  } catch (error) {
    console.error('Registration error:', error);
    res.status(500).json({ error: 'Registration failed' });
  }
});

// Login User
app.post('/api/auth/login', async (req, res) => {
  const { email, password } = req.body;
  
  try {
    const connection = await pool.getConnection();
    
    const [users] = await connection.execute(
      'SELECT user_id, password_hash, full_name, role FROM users WHERE email = ?',
      [email]
    );
    
    if (users.length === 0) {
      connection.release();
      return res.status(401).json({ error: 'Invalid credentials' });
    }
    
    const user = users[0];
    const isValidPassword = await bcrypt.compare(password, user.password_hash);
    
    if (!isValidPassword) {
      connection.release();
      return res.status(401).json({ error: 'Invalid credentials' });
    }
    
    const token = generateToken(user.user_id, user.role);
    
    connection.release();
    
    res.json({ 
      success: true, 
      token, 
      user: { 
        userId: user.user_id, 
        email, 
        fullName: user.full_name, 
        role: user.role 
      } 
    });
  } catch (error) {
    console.error('Login error:', error);
    res.status(500).json({ error: 'Login failed' });
  }
});

// Get Current User
app.get('/api/auth/me', authMiddleware, async (req, res) => {
  try {
    const connection = await pool.getConnection();
    const [users] = await connection.execute(
      'SELECT user_id, email, full_name, phone, role, business_name, location, profile_image FROM users WHERE user_id = ?',
      [req.userId]
    );
    
    connection.release();
    
    if (users.length === 0) return res.status(404).json({ error: 'User not found' });
    
    res.json(users[0]);
  } catch (error) {
    res.status(500).json({ error: 'Failed to fetch user' });
  }
});

// ===============================================
// VENDOR ROUTES
// ===============================================

// Get All Vendors
app.get('/api/vendors', async (req, res) => {
  try {
    const connection = await pool.getConnection();
    const [vendors] = await connection.execute(
      `SELECT user_id, full_name, business_name, location, description, profile_image 
       FROM users WHERE role = 'vendor' ORDER BY full_name`
    );
    connection.release();
    res.json(vendors);
  } catch (error) {
    res.status(500).json({ error: 'Failed to fetch vendors' });
  }
});

// Get Vendor Profile
app.get('/api/vendors/:vendorId', async (req, res) => {
  try {
    const connection = await pool.getConnection();
    const [vendors] = await connection.execute(
      'SELECT * FROM users WHERE user_id = ? AND role = "vendor"',
      [req.params.vendorId]
    );
    
    if (vendors.length === 0) {
      connection.release();
      return res.status(404).json({ error: 'Vendor not found' });
    }
    
    const vendor = vendors[0];
    
    // Fetch vendor's products
    const [products] = await connection.execute(
      'SELECT * FROM products WHERE vendor_id = ? ORDER BY created_at DESC',
      [req.params.vendorId]
    );
    
    connection.release();
    
    res.json({ ...vendor, products });
  } catch (error) {
    res.status(500).json({ error: 'Failed to fetch vendor' });
  }
});

// Update Vendor Profile
app.put('/api/vendors/:vendorId', authMiddleware, upload.single('profileImage'), async (req, res) => {
  if (req.userId !== parseInt(req.params.vendorId) && req.userRole !== 'admin') {
    return res.status(403).json({ error: 'Unauthorized' });
  }
  
  try {
    const { businessName, location, description } = req.body;
    let profileImage = req.body.profileImage;
    
    if (req.file) {
      const uploadResult = await uploadToCloudinary(req.file.buffer, 'vendors');
      profileImage = uploadResult.secure_url;
    }
    
    const connection = await pool.getConnection();
    await connection.execute(
      'UPDATE users SET business_name = ?, location = ?, description = ?, profile_image = ? WHERE user_id = ?',
      [businessName, location, description, profileImage, req.params.vendorId]
    );
    
    const [updated] = await connection.execute(
      'SELECT * FROM users WHERE user_id = ?',
      [req.params.vendorId]
    );
    
    connection.release();
    res.json(updated[0]);
  } catch (error) {
    res.status(500).json({ error: 'Failed to update vendor' });
  }
});

// ===============================================
// PRODUCT ROUTES
// ===============================================

// Create Product (Vendor Only)
app.post('/api/products', authMiddleware, upload.single('image'), async (req, res) => {
  if (req.userRole !== 'vendor') {
    return res.status(403).json({ error: 'Only vendors can create products' });
  }
  
  const { name, description, price, category, minQuantity } = req.body;
  
  try {
    let imageUrl = null;
    if (req.file) {
      const uploadResult = await uploadToCloudinary(req.file.buffer, 'products');
      imageUrl = uploadResult.secure_url;
    }
    
    const connection = await pool.getConnection();
    const [result] = await connection.execute(
      'INSERT INTO products (vendor_id, name, description, price, category, image_url, min_quantity) VALUES (?, ?, ?, ?, ?, ?, ?)',
      [req.userId, name, description, price, category, imageUrl, minQuantity || 1]
    );
    
    const [product] = await connection.execute(
      'SELECT * FROM products WHERE product_id = ?',
      [result.insertId]
    );
    
    connection.release();
    res.status(201).json(product[0]);
  } catch (error) {
    console.error('Product creation error:', error);
    res.status(500).json({ error: 'Failed to create product' });
  }
});

// Get Products by Vendor
app.get('/api/vendors/:vendorId/products', async (req, res) => {
  try {
    const connection = await pool.getConnection();
    const [products] = await connection.execute(
      'SELECT * FROM products WHERE vendor_id = ? ORDER BY created_at DESC',
      [req.params.vendorId]
    );
    connection.release();
    res.json(products);
  } catch (error) {
    res.status(500).json({ error: 'Failed to fetch products' });
  }
});

// Update Product
app.put('/api/products/:productId', authMiddleware, upload.single('image'), async (req, res) => {
  try {
    const connection = await pool.getConnection();
    
    // Check authorization
    const [products] = await connection.execute(
      'SELECT vendor_id FROM products WHERE product_id = ?',
      [req.params.productId]
    );
    
    if (products.length === 0) {
      connection.release();
      return res.status(404).json({ error: 'Product not found' });
    }
    
    if (products[0].vendor_id !== req.userId && req.userRole !== 'admin') {
      connection.release();
      return res.status(403).json({ error: 'Unauthorized' });
    }
    
    const { name, description, price, category, minQuantity, imageUrl } = req.body;
    
    let finalImageUrl = imageUrl;
    if (req.file) {
      const uploadResult = await uploadToCloudinary(req.file.buffer, 'products');
      finalImageUrl = uploadResult.secure_url;
    }
    
    await connection.execute(
      'UPDATE products SET name = ?, description = ?, price = ?, category = ?, image_url = ?, min_quantity = ? WHERE product_id = ?',
      [name, description, price, category, finalImageUrl, minQuantity || 1, req.params.productId]
    );
    
    const [updated] = await connection.execute(
      'SELECT * FROM products WHERE product_id = ?',
      [req.params.productId]
    );
    
    connection.release();
    res.json(updated[0]);
  } catch (error) {
    res.status(500).json({ error: 'Failed to update product' });
  }
});

// Delete Product
app.delete('/api/products/:productId', authMiddleware, async (req, res) => {
  try {
    const connection = await pool.getConnection();
    
    const [products] = await connection.execute(
      'SELECT vendor_id FROM products WHERE product_id = ?',
      [req.params.productId]
    );
    
    if (products.length === 0) {
      connection.release();
      return res.status(404).json({ error: 'Product not found' });
    }
    
    if (products[0].vendor_id !== req.userId && req.userRole !== 'admin') {
      connection.release();
      return res.status(403).json({ error: 'Unauthorized' });
    }
    
    await connection.execute('DELETE FROM products WHERE product_id = ?', [req.params.productId]);
    
    connection.release();
    res.json({ success: true });
  } catch (error) {
    res.status(500).json({ error: 'Failed to delete product' });
  }
});

// ===============================================
// EVENT ROUTES
// ===============================================

// Create Event (Host Only)
app.post('/api/events', authMiddleware, async (req, res) => {
  if (req.userRole !== 'host' && req.userRole !== 'admin') {
    return res.status(403).json({ error: 'Only hosts can create events' });
  }
  
  const { title, description, eventDate, location, guestCount, budget, theme } = req.body;
  
  try {
    const connection = await pool.getConnection();
    const [result] = await connection.execute(
      'INSERT INTO events (host_id, title, description, event_date, location, guest_count, budget, theme) VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
      [req.userId, title, description, eventDate, location, guestCount, budget, theme]
    );
    
    // Initialize budget categories
    const categories = ['Food & Catering', 'Venue', 'Decorations', 'Entertainment', 'Transportation', 'Miscellaneous'];
    for (const category of categories) {
      await connection.execute(
        'INSERT INTO budget_tracking (event_id, category, budgeted_amount) VALUES (?, ?, ?)',
        [result.insertId, category, 0]
      );
    }
    
    const [event] = await connection.execute(
      'SELECT * FROM events WHERE event_id = ?',
      [result.insertId]
    );
    
    connection.release();
    res.status(201).json(event[0]);
  } catch (error) {
    console.error('Event creation error:', error);
    res.status(500).json({ error: 'Failed to create event' });
  }
});

// Get All Events for Host
app.get('/api/events', authMiddleware, async (req, res) => {
  try {
    const connection = await pool.getConnection();
    const [events] = await connection.execute(
      'SELECT * FROM events WHERE host_id = ? ORDER BY event_date DESC',
      [req.userId]
    );
    connection.release();
    res.json(events);
  } catch (error) {
    res.status(500).json({ error: 'Failed to fetch events' });
  }
});

// Get Event Details
app.get('/api/events/:eventId', authMiddleware, async (req, res) => {
  try {
    const connection = await pool.getConnection();
    
    const [events] = await connection.execute(
      'SELECT * FROM events WHERE event_id = ?',
      [req.params.eventId]
    );
    
    if (events.length === 0) {
      connection.release();
      return res.status(404).json({ error: 'Event not found' });
    }
    
    const event = events[0];
    
    // Check authorization
    if (event.host_id !== req.userId && req.userRole !== 'admin') {
      connection.release();
      return res.status(403).json({ error: 'Unauthorized' });
    }
    
    // Get RSVPs
    const [rsvps] = await connection.execute(
      'SELECT * FROM rsvps WHERE event_id = ?',
      [req.params.eventId]
    );
    
    // Get order items
    const [items] = await connection.execute(
      `SELECT oi.*, p.name as product_name, u.business_name as vendor_name 
       FROM order_items oi 
       JOIN products p ON oi.product_id = p.product_id 
       JOIN users u ON oi.vendor_id = u.user_id 
       WHERE oi.event_id = ?`,
      [req.params.eventId]
    );
    
    // Get to-do items
    const [todos] = await connection.execute(
      'SELECT * FROM todo_items WHERE event_id = ? ORDER BY due_date ASC',
      [req.params.eventId]
    );
    
    // Get budget tracking
    const [budgets] = await connection.execute(
      'SELECT * FROM budget_tracking WHERE event_id = ?',
      [req.params.eventId]
    );
    
    // Get vendor recommendations
    const [recommendations] = await connection.execute(
      `SELECT vr.*, u.business_name, u.profile_image 
       FROM vendor_recommendations vr 
       JOIN users u ON vr.vendor_id = u.user_id 
       WHERE vr.event_id = ? 
       ORDER BY vr.match_score DESC`,
      [req.params.eventId]
    );
    
    connection.release();
    
    res.json({
      ...event,
      rsvps,
      orderItems: items,
      todos,
      budgets,
      recommendations
    });
  } catch (error) {
    console.error('Event detail error:', error);
    res.status(500).json({ error: 'Failed to fetch event details' });
  }
});

// Update Event
app.put('/api/events/:eventId', authMiddleware, async (req, res) => {
  try {
    const connection = await pool.getConnection();
    
    const [events] = await connection.execute(
      'SELECT host_id FROM events WHERE event_id = ?',
      [req.params.eventId]
    );
    
    if (events.length === 0) {
      connection.release();
      return res.status(404).json({ error: 'Event not found' });
    }
    
    if (events[0].host_id !== req.userId && req.userRole !== 'admin') {
      connection.release();
      return res.status(403).json({ error: 'Unauthorized' });
    }
    
    const { title, description, eventDate, location, guestCount, budget, theme, status } = req.body;
    
    await connection.execute(
      'UPDATE events SET title = ?, description = ?, event_date = ?, location = ?, guest_count = ?, budget = ?, theme = ?, status = ? WHERE event_id = ?',
      [title, description, eventDate, location, guestCount, budget, theme, status, req.params.eventId]
    );
    
    const [updated] = await connection.execute(
      'SELECT * FROM events WHERE event_id = ?',
      [req.params.eventId]
    );
    
    connection.release();
    res.json(updated[0]);
  } catch (error) {
    res.status(500).json({ error: 'Failed to update event' });
  }
});

// ===============================================
// RSVP ROUTES
// ===============================================

// Get Event RSVPs
app.get('/api/events/:eventId/rsvps', authMiddleware, async (req, res) => {
  try {
    const connection = await pool.getConnection();
    
    const [events] = await connection.execute(
      'SELECT host_id FROM events WHERE event_id = ?',
      [req.params.eventId]
    );
    
    if (events.length === 0 || events[0].host_id !== req.userId) {
      connection.release();
      return res.status(403).json({ error: 'Unauthorized' });
    }
    
    const [rsvps] = await connection.execute(
      'SELECT * FROM rsvps WHERE event_id = ? ORDER BY created_at DESC',
      [req.params.eventId]
    );
    
    // Calculate stats
    const stats = {
      total: rsvps.length,
      accepted: rsvps.filter(r => r.status === 'accepted').length,
      declined: rsvps.filter(r => r.status === 'declined').length,
      maybe: rsvps.filter(r => r.status === 'maybe').length,
      invited: rsvps.filter(r => r.status === 'invited').length,
      totalGuests: rsvps.reduce((sum, r) => sum + r.guest_count, 0)
    };
    
    connection.release();
    res.json({ rsvps, stats });
  } catch (error) {
    res.status(500).json({ error: 'Failed to fetch RSVPs' });
  }
});

// Update RSVP Response
app.post('/api/events/:eventId/rsvps/:rsvpId', async (req, res) => {
  const { status, guestCount } = req.body;
  
  try {
    const connection = await pool.getConnection();
    
    await connection.execute(
      'UPDATE rsvps SET status = ?, guest_count = ?, response_date = NOW() WHERE rsvp_id = ? AND event_id = ?',
      [status, guestCount || 1, req.params.rsvpId, req.params.eventId]
    );
    
    const [updated] = await connection.execute(
      'SELECT * FROM rsvps WHERE rsvp_id = ?',
      [req.params.rsvpId]
    );
    
    connection.release();
    res.json(updated[0]);
  } catch (error) {
    res.status(500).json({ error: 'Failed to update RSVP' });
  }
});

// ===============================================
// INVITE ROUTES
// ===============================================

// Send Invites
app.post('/api/events/:eventId/send-invites', authMiddleware, async (req, res) => {
  const { emails } = req.body;
  
  try {
    const connection = await pool.getConnection();
    
    const [events] = await connection.execute(
      'SELECT host_id FROM events WHERE event_id = ?',
      [req.params.eventId]
    );
    
    if (events.length === 0 || events[0].host_id !== req.userId) {
      connection.release();
      return res.status(403).json({ error: 'Unauthorized' });
    }
    
    const invites = [];
    for (const email of emails) {
      const token = require('crypto').randomBytes(32).toString('hex');
      const expiresAt = new Date(Date.now() + 30 * 24 * 60 * 60 * 1000); // 30 days
      
      const [result] = await connection.execute(
        'INSERT INTO invites (event_id, invite_email, invite_token, expires_at) VALUES (?, ?, ?, ?)',
        [req.params.eventId, email, token, expiresAt]
      );
      
      invites.push({
        id: result.insertId,
        email,
        token
      });
      
      // TODO: Send email with invite link
    }
    
    connection.release();
    res.status(201).json({ success: true, invites });
  } catch (error) {
    res.status(500).json({ error: 'Failed to send invites' });
  }
});

// Accept Invite
app.post('/api/invites/:token/accept', async (req, res) => {
  const { guestName, guestEmail, guestPhone, dietaryRequirements } = req.body;
  
  try {
    const connection = await pool.getConnection();
    
    const [invites] = await connection.execute(
      'SELECT event_id FROM invites WHERE invite_token = ? AND expires_at > NOW()',
      [req.params.token]
    );
    
    if (invites.length === 0) {
      connection.release();
      return res.status(400).json({ error: 'Invalid or expired invite' });
    }
    
    const eventId = invites[0].event_id;
    
    // Create or update RSVP
    const [result] = await connection.execute(
      `INSERT INTO rsvps (event_id, guest_name, guest_email, guest_phone, status, dietary_requirements)
       VALUES (?, ?, ?, ?, 'accepted', ?)
       ON DUPLICATE KEY UPDATE status = 'accepted', guest_phone = ?, dietary_requirements = ?`,
      [eventId, guestName, guestEmail, guestPhone, dietaryRequirements, guestPhone, dietaryRequirements]
    );
    
    // Mark invite as opened
    await connection.execute(
      'UPDATE invites SET status = "opened" WHERE invite_token = ?',
      [req.params.token]
    );
    
    connection.release();
    res.json({ success: true, eventId });
  } catch (error) {
    console.error('Invite acceptance error:', error);
    res.status(500).json({ error: 'Failed to accept invite' });
  }
});

// ===============================================
// TO-DO ROUTES
// ===============================================

// Get To-Do Items
app.get('/api/events/:eventId/todos', authMiddleware, async (req, res) => {
  try {
    const connection = await pool.getConnection();
    
    const [todos] = await connection.execute(
      'SELECT * FROM todo_items WHERE event_id = ? ORDER BY due_date ASC, priority DESC',
      [req.params.eventId]
    );
    
    connection.release();
    res.json(todos);
  } catch (error) {
    res.status(500).json({ error: 'Failed to fetch to-do items' });
  }
});

// Create To-Do Item
app.post('/api/events/:eventId/todos', authMiddleware, async (req, res) => {
  const { title, description, category, priority, dueDate, assignedTo } = req.body;
  
  try {
    const connection = await pool.getConnection();
    
    const [result] = await connection.execute(
      'INSERT INTO todo_items (event_id, title, description, category, priority, due_date, assigned_to) VALUES (?, ?, ?, ?, ?, ?, ?)',
      [req.params.eventId, title, description, category, priority, dueDate, assignedTo]
    );
    
    const [todo] = await connection.execute(
      'SELECT * FROM todo_items WHERE todo_id = ?',
      [result.insertId]
    );
    
    connection.release();
    res.status(201).json(todo[0]);
  } catch (error) {
    res.status(500).json({ error: 'Failed to create to-do item' });
  }
});

// Update To-Do Item
app.put('/api/todos/:todoId', authMiddleware, async (req, res) => {
  const { title, description, category, priority, dueDate, completed } = req.body;
  
  try {
    const connection = await pool.getConnection();
    
    await connection.execute(
      'UPDATE todo_items SET title = ?, description = ?, category = ?, priority = ?, due_date = ?, completed = ? WHERE todo_id = ?',
      [title, description, category, priority, dueDate, completed, req.params.todoId]
    );
    
    const [updated] = await connection.execute(
      'SELECT * FROM todo_items WHERE todo_id = ?',
      [req.params.todoId]
    );
    
    connection.release();
    res.json(updated[0]);
  } catch (error) {
    res.status(500).json({ error: 'Failed to update to-do item' });
  }
});

// Delete To-Do Item
app.delete('/api/todos/:todoId', authMiddleware, async (req, res) => {
  try {
    const connection = await pool.getConnection();
    await connection.execute('DELETE FROM todo_items WHERE todo_id = ?', [req.params.todoId]);
    connection.release();
    res.json({ success: true });
  } catch (error) {
    res.status(500).json({ error: 'Failed to delete to-do item' });
  }
});

// ===============================================
// ORDER & PAYMENT ROUTES
// ===============================================

// Add Product to Order
app.post('/api/events/:eventId/order-items', authMiddleware, async (req, res) => {
  const { productId, quantity } = req.body;
  
  try {
    const connection = await pool.getConnection();
    
    const [products] = await connection.execute(
      'SELECT price, vendor_id FROM products WHERE product_id = ?',
      [productId]
    );
    
    if (products.length === 0) {
      connection.release();
      return res.status(404).json({ error: 'Product not found' });
    }
    
    const product = products[0];
    const totalPrice = product.price * quantity;
    
    const [result] = await connection.execute(
      'INSERT INTO order_items (event_id, product_id, vendor_id, quantity, unit_price, total_price) VALUES (?, ?, ?, ?, ?, ?)',
      [req.params.eventId, productId, product.vendor_id, quantity, product.price, totalPrice]
    );
    
    const [item] = await connection.execute(
      `SELECT oi.*, p.name, u.business_name as vendor_name 
       FROM order_items oi 
       JOIN products p ON oi.product_id = p.product_id 
       JOIN users u ON oi.vendor_id = u.user_id 
       WHERE oi.order_item_id = ?`,
      [result.insertId]
    );
    
    connection.release();
    res.status(201).json(item[0]);
  } catch (error) {
    res.status(500).json({ error: 'Failed to add item to order' });
  }
});

// Get Order Items
app.get('/api/events/:eventId/order-items', authMiddleware, async (req, res) => {
  try {
    const connection = await pool.getConnection();
    
    const [items] = await connection.execute(
      `SELECT oi.*, p.name, p.image_url, u.business_name as vendor_name 
       FROM order_items oi 
       JOIN products p ON oi.product_id = p.product_id 
       JOIN users u ON oi.vendor_id = u.user_id 
       WHERE oi.event_id = ?`,
      [req.params.eventId]
    );
    
    connection.release();
    res.json(items);
  } catch (error) {
    res.status(500).json({ error: 'Failed to fetch order items' });
  }
});

// Remove Order Item
app.delete('/api/order-items/:itemId', authMiddleware, async (req, res) => {
  try {
    const connection = await pool.getConnection();
    await connection.execute('DELETE FROM order_items WHERE order_item_id = ?', [req.params.itemId]);
    connection.release();
    res.json({ success: true });
  } catch (error) {
    res.status(500).json({ error: 'Failed to remove item' });
  }
});

// Create Payment Intent (Stripe)
app.post('/api/payments/create-intent', authMiddleware, async (req, res) => {
  const { eventId, amount } = req.body;
  
  try {
    const paymentIntent = await stripe.paymentIntents.create({
      amount: Math.round(amount * 100), // Convert to cents
      currency: 'usd',
      metadata: { eventId }
    });
    
    res.json({ clientSecret: paymentIntent.client_secret });
  } catch (error) {
    console.error('Payment intent error:', error);
    res.status(500).json({ error: 'Failed to create payment intent' });
  }
});

// Confirm Payment
app.post('/api/payments/confirm', authMiddleware, async (req, res) => {
  const { eventId, stripePaymentId, amount, vendorId } = req.body;
  
  try {
    const connection = await pool.getConnection();
    
    const [result] = await connection.execute(
      'INSERT INTO payments (event_id, vendor_id, amount, stripe_payment_id, status, payment_method, transaction_date) VALUES (?, ?, ?, ?, "completed", "stripe", NOW())',
      [eventId, vendorId, amount, stripePaymentId]
    );
    
    // Update order item status
    await connection.execute(
      'UPDATE order_items SET status = "confirmed" WHERE event_id = ? AND vendor_id = ?',
      [eventId, vendorId]
    );
    
    connection.release();
    res.json({ success: true, paymentId: result.insertId });
  } catch (error) {
    res.status(500).json({ error: 'Failed to confirm payment' });
  }
});

// Get Payments for Event
app.get('/api/events/:eventId/payments', authMiddleware, async (req, res) => {
  try {
    const connection = await pool.getConnection();
    
    const [payments] = await connection.execute(
      'SELECT * FROM payments WHERE event_id = ? ORDER BY transaction_date DESC',
      [req.params.eventId]
    );
    
    connection.release();
    res.json(payments);
  } catch (error) {
    res.status(500).json({ error: 'Failed to fetch payments' });
  }
});

// ===============================================
// VENDOR RECOMMENDATION ROUTES
// ===============================================

// Get Vendor Recommendations
app.get('/api/events/:eventId/recommendations', authMiddleware, async (req, res) => {
  try {
    const connection = await pool.getConnection();
    
    const [recs] = await connection.execute(
      `SELECT vr.*, u.user_id, u.business_name, u.profile_image, u.location, 
              (SELECT AVG(reviewer_rating) FROM reviews WHERE vendor_id = u.user_id) as avg_rating
       FROM vendor_recommendations vr 
       JOIN users u ON vr.vendor_id = u.user_id 
       WHERE vr.event_id = ? 
       ORDER BY vr.match_score DESC`,
      [req.params.eventId]
    );
    
    connection.release();
    res.json(recs);
  } catch (error) {
    res.status(500).json({ error: 'Failed to fetch recommendations' });
  }
});

// Generate Recommendations (Algorithm-based)
app.post('/api/events/:eventId/generate-recommendations', authMiddleware, async (req, res) => {
  try {
    const connection = await pool.getConnection();
    
    // Get event details
    const [events] = await connection.execute(
      'SELECT * FROM events WHERE event_id = ? AND host_id = ?',
      [req.params.eventId, req.userId]
    );
    
    if (events.length === 0) {
      connection.release();
      return res.status(403).json({ error: 'Unauthorized' });
    }
    
    const event = events[0];
    
    // Get all vendors with their products and ratings
    const [vendors] = await connection.execute(
      `SELECT u.user_id, u.business_name, u.location, COUNT(p.product_id) as product_count,
              COALESCE(AVG(r.reviewer_rating), 4.0) as avg_rating
       FROM users u 
       LEFT JOIN products p ON u.user_id = p.vendor_id 
       LEFT JOIN reviews r ON u.user_id = r.vendor_id 
       WHERE u.role = 'vendor' 
       GROUP BY u.user_id`
    );
    
    // Clear existing recommendations
    await connection.execute(
      'DELETE FROM vendor_recommendations WHERE event_id = ?',
      [req.params.eventId]
    );
    
    // Calculate match scores and save recommendations
    const recommendations = vendors.map(vendor => {
      let score = 0;
      
      // Rating factor (40%)
      score += (vendor.avg_rating / 5) * 40;
      
      // Product availability factor (30%)
      score += Math.min((vendor.product_count / 10) * 30, 30);
      
      // Random factor for variety (30%)
      score += Math.random() * 30;
      
      return { vendor_id: vendor.user_id, score: score / 100, reason: `Match: ${vendor.product_count} products, ${vendor.avg_rating.toFixed(1)}★` };
    }).sort((a, b) => b.score - a.score).slice(0, 5); // Top 5
    
    for (const rec of recommendations) {
      await connection.execute(
        'INSERT INTO vendor_recommendations (event_id, vendor_id, match_score, reason) VALUES (?, ?, ?, ?)',
        [req.params.eventId, rec.vendor_id, rec.score, rec.reason]
      );
    }
    
    connection.release();
    res.json({ success: true, recommendations });
  } catch (error) {
    console.error('Recommendation error:', error);
    res.status(500).json({ error: 'Failed to generate recommendations' });
  }
});

// ===============================================
// BUDGET TRACKING ROUTES
// ===============================================

// Get Budget
app.get('/api/events/:eventId/budget', authMiddleware, async (req, res) => {
  try {
    const connection = await pool.getConnection();
    
    const [budgets] = await connection.execute(
      'SELECT * FROM budget_tracking WHERE event_id = ? ORDER BY category',
      [req.params.eventId]
    );
    
    // Calculate totals
    const totalBudgeted = budgets.reduce((sum, b) => sum + (b.budgeted_amount || 0), 0);
    const totalSpent = budgets.reduce((sum, b) => sum + (b.spent_amount || 0), 0);
    
    connection.release();
    res.json({ budgets, totalBudgeted, totalSpent, remaining: totalBudgeted - totalSpent });
  } catch (error) {
    res.status(500).json({ error: 'Failed to fetch budget' });
  }
});

// Update Budget Category
app.put('/api/budget/:budgetId', authMiddleware, async (req, res) => {
  const { budgetedAmount } = req.body;
  
  try {
    const connection = await pool.getConnection();
    
    await connection.execute(
      'UPDATE budget_tracking SET budgeted_amount = ? WHERE budget_id = ?',
      [budgetedAmount, req.params.budgetId]
    );
    
    const [updated] = await connection.execute(
      'SELECT * FROM budget_tracking WHERE budget_id = ?',
      [req.params.budgetId]
    );
    
    connection.release();
    res.json(updated[0]);
  } catch (error) {
    res.status(500).json({ error: 'Failed to update budget' });
  }
});

// ===============================================
// ABSA MICROSERVICE PROXY ROUTES
// ===============================================

// Generic proxy helper
async function proxyAbsa(path, body) {
  const url = `${ABSA_URL}${path}`;
  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  return response.json();
}

// Analyse a single review
app.post('/api/absa/analyze', async (req, res) => {
  try {
    const result = await proxyAbsa('/analyze', req.body);
    res.json(result);
  } catch (error) {
    console.error('ABSA /analyze error:', error);
    res.status(502).json({ error: 'ABSA service unavailable' });
  }
});

// Analyse all reviews for a vendor
app.post('/api/absa/analyze-vendor', async (req, res) => {
  try {
    const result = await proxyAbsa('/analyze-vendor', req.body);
    res.json(result);
  } catch (error) {
    console.error('ABSA /analyze-vendor error:', error);
    res.status(502).json({ error: 'ABSA service unavailable' });
  }
});

// Rank vendors via Elo
app.post('/api/absa/rank-vendors', async (req, res) => {
  try {
    const result = await proxyAbsa('/rank-vendors', req.body);
    res.json(result);
  } catch (error) {
    console.error('ABSA /rank-vendors error:', error);
    res.status(502).json({ error: 'ABSA service unavailable' });
  }
});

// Budget-aware vendor recommendation
app.post('/api/absa/recommend', async (req, res) => {
  try {
    const result = await proxyAbsa('/recommend', req.body);
    res.json(result);
  } catch (error) {
    console.error('ABSA /recommend error:', error);
    res.status(502).json({ error: 'ABSA service unavailable' });
  }
});

// Override: generate-recommendations now calls ABSA then stores results
app.post('/api/events/:eventId/generate-recommendations', authMiddleware, async (req, res) => {
  try {
    const connection = await pool.getConnection();

    const [events] = await connection.execute(
      'SELECT * FROM events WHERE event_id = ? AND host_id = ?',
      [req.params.eventId, req.userId]
    );
    if (events.length === 0) {
      connection.release();
      return res.status(403).json({ error: 'Unauthorized' });
    }

    const [vendors] = await connection.execute(
      `SELECT u.user_id, u.business_name, u.location,
              COUNT(p.product_id) as product_count,
              COALESCE(AVG(r.reviewer_rating), 3.5) as avg_rating
       FROM users u
       LEFT JOIN products p ON u.user_id = p.vendor_id
       LEFT JOIN reviews r  ON u.user_id = r.vendor_id
       WHERE u.role = 'vendor'
       GROUP BY u.user_id`
    );

    // Build absa_score from avg_rating (maps 1-5 to -1..+1)
    const absa_vendors = vendors.map(v => ({
      name:         v.business_name,
      raw_score:    (v.avg_rating - 3) / 2,
      review_count: Number(v.product_count) + 1,
    }));

    let elo_rankings = [];
    if (absa_vendors.length > 0) {
      try {
        const absa_res = await proxyAbsa('/rank-vendors', { vendors: absa_vendors });
        elo_rankings = absa_res.rankings || [];
      } catch (_) {
        // fallback to raw avg_rating sort
        elo_rankings = absa_vendors
          .sort((a, b) => b.raw_score - a.raw_score)
          .map((v, i) => ({ vendor: v.name, rank: i + 1, star_rating: Math.round(((v.raw_score + 1) / 2 * 4 + 1) * 10) / 10 }));
      }
    }

    // Map vendor names back to vendor_id
    const vendorByName = {};
    vendors.forEach(v => { vendorByName[v.business_name] = v; });

    await connection.execute(
      'DELETE FROM vendor_recommendations WHERE event_id = ?',
      [req.params.eventId]
    );

    const top5 = elo_rankings.slice(0, 5);
    const inserted = [];
    for (const r of top5) {
      const v = vendorByName[r.vendor];
      if (!v) continue;
      const score  = r.star_rating / 5;
      const reason = `ABSA Elo rank #${r.rank} - ${r.star_rating} stars`;
      await connection.execute(
        'INSERT INTO vendor_recommendations (event_id, vendor_id, match_score, reason) VALUES (?, ?, ?, ?)',
        [req.params.eventId, v.user_id, score, reason]
      );
      inserted.push({ vendor_id: v.user_id, score, reason });
    }

    connection.release();
    res.json({ success: true, recommendations: inserted, elo_rankings });
  } catch (error) {
    console.error('Recommendation error:', error);
    res.status(500).json({ error: 'Failed to generate recommendations' });
  }
});

// ===============================================
// HEALTH CHECK
// ===============================================

app.get('/api/health', (req, res) => {
  res.json({ status: 'OK', timestamp: new Date().toISOString() });
});

// ===============================================
// ERROR HANDLING
// ===============================================

app.use((err, req, res, next) => {
  console.error('Error:', err);
  res.status(500).json({ error: 'Internal server error' });
});

// ===============================================
// START SERVER
// ===============================================

const PORT = process.env.PORT || 5000;

async function start() {
  try {
    await initializeDatabase();
    
    app.listen(PORT, () => {
      console.log(`\n✓ Eventify Server running on port ${PORT}`);
      console.log(`✓ API available at http://localhost:${PORT}/api`);
      console.log(`✓ Health check: http://localhost:${PORT}/api/health\n`);
    });
  } catch (error) {
    console.error('Failed to start server:', error);
    process.exit(1);
  }
}

start();

module.exports = app;
