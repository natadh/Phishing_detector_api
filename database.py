import sqlite3
from datetime import datetime

DB_NAME = "phishing_detector.db"

def init_db():
    """Initialize (or create) the database and the table."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS email_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email_text TEXT,
            cleaned_text TEXT,
            prediction INTEGER,
            recommendation TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def save_prediction(email_text, cleaned_text, prediction, recommendation):
    """Save an email prediction record to the database."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        INSERT INTO email_predictions (email_text, cleaned_text, prediction, recommendation)
        VALUES (?, ?, ?, ?)
    ''', (email_text, cleaned_text, prediction, recommendation))
    conn.commit()
    conn.close()

def fetch_predictions(limit=10):
    """Fetch the most recent predictions."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('SELECT * FROM email_predictions ORDER BY id DESC LIMIT ?', (limit,))
    rows = c.fetchall()
    conn.close()
    return rows
