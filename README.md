# Local face recognition demo

An opt-in example that matches one image or webcam faces against labeled reference photos. This is a demo, not an access-control system.

## Run

Install `requirements.txt` in a compatible Python environment, place one clear face per labeled image in a local directory, then run:

```bash
python3 "recognition/face recognition.py" --known-dir /path/to/consented/photos --image /path/to/test.jpg
```

Omit `--image` to use a webcam; press `q` to quit. The directory and photos are local and excluded from Git. `python3 -m unittest discover -s tests -v` runs the dependency-free matching check.

## Résumé evidence

See [résumé evidence and verification](docs/RESUME_EVIDENCE.md) for current limits and safe wording.
