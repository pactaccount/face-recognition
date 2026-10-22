# Method — Technical Design & Choices

## Pipeline
1. Load references from `--known-dir`
2. Validate: directory exists, exactly one face per image (`face_encodings` count == 1)
3. Encode: dlib CNN → 128-d float vector
4. Compare: Euclidean distance between query encoding and all references
5. Decide: `min(distance) <= 0.55` → label; else "Unknown"

## Technique choices and alternatives

| Component | Chosen | Why | Alternatives considered |
|---|---|---|---|
| Detector/Encoder | dlib (face_recognition) | Wide open-source use; 128-d embeddings well-tested; minimal dependencies besides dlib | DeepFace (TensorFlow); FaceNet (TensorFlow); ArcFace (PyTorch); Haar Cascade (older, less accurate) |
| Distance | Euclidean | Interpretable; directly compares embeddings; sufficient for small labeled set | Cosine similarity; Angular margin (ArcFace); Mahalanobis |
| Image I/O | OpenCV (cv2) | Handles webcam, file load, BGR→RGB, rectangle drawing; standard in cv pipelines | Pillow (simpler, slower for video); imageio |
| Matching logic | Pure Python (`matching.py`) | Isolates decision from dependencies; enables unit testing without installing CV libs | Integrated in main file; external library (sklearn neighbors) |
| Threshold | 0.55 (default) | Standard dlib recommendation; balances false accept vs false reject for small galleries | Tuned per dataset (LFW/IJB require calibrated thresholds) |

## Limitations (documented, not hidden)
- No liveness / anti-spoof
- No demographic or cross-pose evaluation data
- No paired verification set (LFW download unavailable in this environment)
- CLI window can hang on first open (OpenCV display loop); direct Python test avoids this
- Single-file reference only (gallery >1 tested conceptually but not fully validated at scale)
