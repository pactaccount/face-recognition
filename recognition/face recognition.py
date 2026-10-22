"""Opt-in local face identification demo using labeled reference photos.

This is a demo, not an identity or access-control system. No images are sent
to a server; the caller supplies the photos and an image or webcam input.
"""

import argparse
from pathlib import Path

from matching import choose_identity


def load_known(directory, face_recognition, cv2):
    names, encodings = [], []
    if not directory.is_dir():
        raise ValueError(f"Reference directory does not exist: {directory}")
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        image = cv2.imread(str(path))
        if image is None:
            print(f"Skipping unreadable image: {path.name}")
            continue
        found = face_recognition.face_encodings(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if len(found) != 1:
            print(f"Skipping {path.name}: expected one face, found {len(found)}")
            continue
        names.append(path.stem)
        encodings.append(found[0])
    if not encodings:
        raise ValueError("Add at least one readable reference photo containing exactly one face.")
    return names, encodings


def annotate(frame, names, encodings, threshold, face_recognition, cv2):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb)
    found = face_recognition.face_encodings(rgb, boxes)
    for encoding, (top, right, bottom, left) in zip(found, boxes):
        distances = face_recognition.face_distance(encodings, encoding)
        label, distance = choose_identity(names, distances, threshold)
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 200, 0), 2)
        cv2.putText(frame, f"{label} ({distance:.2f})", (left, max(20, top - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
    return frame, len(found)


def main():
    parser = argparse.ArgumentParser(description="Local face-recognition demonstration")
    parser.add_argument("--known-dir", type=Path, required=True, help="Labeled photos; filename is the label")
    parser.add_argument("--image", type=Path, help="Test one image instead of opening the webcam")
    parser.add_argument("--threshold", type=float, default=0.55, help="Maximum face distance for a match")
    args = parser.parse_args()
    if not 0 < args.threshold < 1:
        parser.error("--threshold must be between 0 and 1")
    try:
        import cv2
        import face_recognition
    except ImportError as exc:
        parser.error(f"Missing dependency: {exc.name}. Install requirements.txt first.")
    try:
        names, encodings = load_known(args.known_dir, face_recognition, cv2)
    except ValueError as exc:
        parser.error(str(exc))

    if args.image:
        frame = cv2.imread(str(args.image))
        if frame is None:
            parser.error(f"Could not read image: {args.image}")
        _, count = annotate(frame, names, encodings, args.threshold, face_recognition, cv2)
        print(f"Detected {count} face(s).")
        cv2.imshow("Recognition demo — press any key", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        parser.error("Webcam unavailable. Use --image instead.")
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                break
            annotate(frame, names, encodings, args.threshold, face_recognition, cv2)
            cv2.imshow("Recognition demo — press q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
