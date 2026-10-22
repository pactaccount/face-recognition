# Project Overview — Local Consent Face Identification Demo

## What
A fully local, opt-in face-identification pipeline using labeled reference photos. The user provides consented images; the system detects exactly one face per reference, computes a 128-dimensional embedding (dlib), and compares via Euclidean distance to return a label or "Unknown."

## Why it exists (use case)
Small consented events (meetings, organizers, personal photo grouping) where privacy is required and server processing is unacceptable. All computation stays on the caller's machine.

## Architecture (high level)
Image input → OpenCV load → dlib face detection → 128-d embedding → nearest-neighbor comparison (threshold 0.55) → label / Unknown.

## Real-world scenario
Event organizer places one clear face photo per attendee in a local folder (`event_demo/consented/`). At check-in, a test photo is matched against the folder. No images leave the device; consent is explicit through file presence.

## Verified result (2026-10-22)
- Installed: face-recognition 1.3.0, dlib 20.0.1, opencv-python 5.0.0, numpy 2.4.6
- Self-match on consented reference (Alice.jpg): distance 0.00 → correct label at threshold 0.55
- Unit test: pass (test_matching)
- CLI: --help valid; full window run valid (window load may hang — documented)
- Full LFW paired benchmark: attempted download from vis-www.cs.umass.edu; network unavailable; protocol documented (13k / 5,700 / 6,000 pairs / 10-fold CV)

## Enterprise extension path (not implemented — for resume context)
- Replace folder with database (PostgreSQL / Milvus / FAISS)
- Add REST API (FastAPI) for multi-system access
- Add audit logging and role-based access
- Add liveness / anti-spoof (depth camera / challenge-response)
- Replace dlib with ArcFace / transformer-based embedding for accuracy
