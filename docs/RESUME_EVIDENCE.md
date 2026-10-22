# Face recognition — résumé and demo evidence

## Problem and use case
This is an opt-in local face-identification example using labeled reference photos. It demonstrates embedding-based nearest-neighbor matching, not an identity verification or access-control product. Use only photos with permission and do not describe it as suitable for sensitive decisions.

## Engineering approach
The CLI loads images from `--known-dir`, accepts only references containing exactly one detected face, computes face embeddings, and compares distances to a query image or webcam frame. A threshold returns “Unknown” when the closest reference is too far away. The program now checks unreadable inputs, a missing camera, and an empty reference set; it releases the camera and supports `q` to quit.

## Hard parts and limits
The original one-file loop crashed on empty/invalid inputs and could index a nonexistent match. It also had no stopping control. The matcher now has an isolated unit test. The `face-recognition`/dlib runtime is not installed in this workspace, so a complete image or webcam run remains unverified. There are no measured false-accept or false-reject rates, demographic slices, or latency data.

## Proof and résumé wording
`python3 -m unittest discover -s tests -v` passed the matching test and `python3 'recognition/face recognition.py' --help` ran on 2026-10-22. Use [the source](https://github.com/pactaccount/face-recognition) only as a small code sample. For a vision résumé, first run a labeled, consented verification set and report false-accept/false-reject rates at a declared threshold. Otherwise omit this project in favor of Argus or U-Net.
