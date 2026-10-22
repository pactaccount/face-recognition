import sqlite3, os
DB = os.path.join(os.path.dirname(__file__), 'attendees.db')

def init_db():
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS attendees (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, photo_path TEXT, embedding BLOB, checked_in INTEGER DEFAULT 0, check_in_time TEXT)')
    conn.commit(); conn.close()

def add_attendee(name, photo_path, embedding_bytes):
    conn = sqlite3.connect(DB); c = conn.cursor()
    c.execute('INSERT INTO attendees (name, photo_path, embedding, checked_in) VALUES (?, ?, ?, 0)', (name, photo_path, sqlite3.Binary(embedding_bytes)))
    conn.commit(); conn.close()

def get_attendees():
    conn = sqlite3.connect(DB); c = conn.cursor()
    c.execute('SELECT id, name, photo_path, checked_in FROM attendees')
    rows = c.fetchall(); conn.close(); return rows

def check_in(name):
    conn = sqlite3.connect(DB); c = conn.cursor()
    c.execute('UPDATE attendees SET checked_in=1, check_in_time=datetime(\"now\") WHERE name=?', (name,))
    conn.commit(); conn.close()
