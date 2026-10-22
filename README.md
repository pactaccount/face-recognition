# Face Recognition Demo (Local Opt-In)

A consented, local face-identification demo. No images leave your machine. Use only labeled photos you own.

## Run
```bash
pip install -r requirements.txt
python3 "recognition/face recognition.py" --known-dir recognition/known --image recognition/known/ROCK.jpg
```
Omit `--image` for webcam (press q to quit).

## Verified
- 2026-10-22: face detected, 128-d embedding, self-distance 0.00, threshold 0.55 → match.
- Unit tests pass (`python3 -m unittest discover -s tests -v`).
- Accuracy / FAR / TAR measured against standard benchmarks; paired test metrics documented in docs/METRICS_EVIDENCE.md.

## Docs
- Build technical: [docs/BUILD_TECHNICAL.md](docs/BUILD_TECHNICAL.md)
- Metrics evidence: [docs/METRICS_EVIDENCE.md](docs/METRICS_EVIDENCE.md)
- Resume entry: [docs/RESUME_READY.md](docs/RESUME_READY.md)
- Use case: [docs/USE_CASE.md](docs/USE_CASE.md)
