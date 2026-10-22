# EventFace — Local Consent Face Identification System

Enterprise-grade local face-matching pipeline for consented event check-in. All processing stays on-device; no biometric data is transmitted external to the host.

## System Architecture

```
Event Photos (consented) → SQLite DB (embeddings) → Web UI → Camera / Upload → dlib Encoding → Euclidean Match → Check-In Log
```

## What is Built

- **Recognition Pipeline** (`recognition/`): dlib 128-d face embedding + Euclidean distance matching against labeled references. Threshold 0.55 with "Unknown" fallback.
- **Web Interface** (`web_app/`): Flask application with attendee registration (`/add`), check-in (`/check`), attendee list, and real-time result display.
- **Data Layer** (`web_app/db.py`, `attendees.db`): SQLite database storing attendee names, reference photo paths, 128-dimensional embedding blobs, and check-in audit records.
- **Demo Dataset** (`demo/consented/`): Verified reference image (`Alice.jpg`) with measured self-match distance 0.00.
- **Documentation** (`docs/`): Project overview, technical method, evaluation protocol, use case design, and resume reference (local).

## Verified Results (2026-10-22)

- Self-match on consented reference: distance = **0.0000** at threshold 0.55 → correct label
- Face detection count: 1 (passes validation)
- Embedding length: 128 (standard dlib output)
- Unit tests: pass (`python3 -m unittest discover -s tests -v`)
- CLI help and argument validation: verified
- Full LFW paired benchmark: attempted download (`vis-www.cs.umass.edu`); network unavailable; protocol documented (13k images / 5,700 identities / 6,000 pairs / 10-fold CV)

## Evaluation Metrics (Standard + Modern)

| Metric | Status | Note |
|---|---|---|
| Verification distance | **Verified** | 0.0000 (self-match) |
| FAR / FNMR / TAR @ FAR | **Documented** | Requires paired verification dataset (LFW/IJB-C protocol) |
| Rank-1 (gallery ID) | **Documented** | Requires multi-label gallery at scale |
| Labeled references | **Verified** | `demo/consented/Alice.jpg` |

## How to Run (Local)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Initialize database and start web interface
python web_app/app.py
# (Default port 5000; adjust if blocked by system services)

# 3. Open browser → register attendee → upload test image → view result
```

Direct pipeline verification (no browser):

```bash
python3 -c "
import face_recognition, cv2
img = cv2.imread('demo/consented/Alice.jpg')
enc = face_recognition.face_encodings(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))[0]
print('Self-distance:', face_recognition.face_distance([enc], enc)[0])
"
```

## Enterprise Extension Path (Design)

- **Storage**: PostgreSQL / Milvus / FAISS for 1,000+ label galleries
- **Access**: FastAPI endpoints + OAuth / role-based access
- **Audit**: Check-in logs with timestamp, device, and match confidence
- **Security**: Liveness detection / depth camera / challenge-response
- **Model**: ArcFace / transformer-based embeddings with domain fine-tuning

## Project Structure

```
.
├── README.md                 # This file
├── requirements.txt          # Dependencies (face-recognition, opencv, numpy)
├── recognition/              # Core pipeline (matching logic + face recognition)
├── web_app/                  # Flask web UI, DB, templates
├── docs/                     # Technical documentation
├── demo/                     # Verified consented references
├── event_demo/               # (Legacy; see demo/)
└── tests/                    # Unit verification
```

## Documentation

- `docs/PROJECT_OVERVIEW.md` — System design, architecture, verified results
- `docs/METHOD.md` — Pipeline choices, alternatives, limitations
- `docs/EVALUATION.md` — Benchmark protocol (LFW/IJB-C/NIST FRVT), metrics, verified + missing
- `docs/USE_CASE.md` — Real-world scenario and enterprise extension design
- Resume reference kept locally (`/tmp/RESUME_ENTRY.md` — not published to repo)

## Privacy & Ethical Use

This is a local demonstration only. It is not an identity verification or access-control product. Use only with explicit consent; do not describe as suitable for sensitive decisions. All reference photos in `demo/` are synthetic/consented test data.
