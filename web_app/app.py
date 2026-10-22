from flask import Flask, render_template, request, jsonify
import os, sys, cv2, face_recognition, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'recognition'))
from matching import choose_identity
from db import init_db, get_attendees, check_in, add_attendee
import sqlite3

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', attendees=get_attendees())

@app.route('/add', methods=['POST'])
def add():
    name = request.form['name']
    f = request.files['photo']
    path = os.path.join('event_demo/consented', f.filename)
    f.save(path)
    img = cv2.imread(path)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    enc = face_recognition.face_encodings(rgb)[0]
    add_attendee(name, path, enc.tobytes())
    return jsonify({"status":"added","name":name})

@app.route('/check', methods=['POST'])
def check():
    f = request.files['test_image']
    path = os.path.join('web_app/uploads', f.filename)
    os.makedirs('web_app/uploads', exist_ok=True)
    f.save(path)
    img = cv2.imread(path)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    enc = face_recognition.face_encodings(rgb)[0]
    conn = sqlite3.connect('web_app/attendees.db')
    c = conn.cursor(); c.execute('SELECT name, embedding FROM attendees WHERE checked_in=0')
    names, encodings = [], []
    for row in c.fetchall():
        names.append(row[0]); encodings.append(np.frombuffer(row[1], dtype=np.float64))
    conn.close()
    if not names:
        return jsonify({"status":"no_attendees","label":"Unknown","dist":"N/A"})
    distances = face_recognition.face_distance(encodings, enc)
    best = min(range(len(distances)), key=lambda i: distances[i])
    label, dist = (names[best] if distances[best] <= 0.55 else "Unknown", float(distances[best]))
    if label != "Unknown":
        check_in(label)
    return jsonify({"status":"checked","label":label,"distance":round(dist,4)})

if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=5000, debug=True)
