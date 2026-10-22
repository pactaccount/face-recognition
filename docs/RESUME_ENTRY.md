# Resume Entry Reference

Project link: https://github.com/pactaccount/face-recognition

One-line summary:
"Local consent face-identification demo (dlib/Opencv/Euclidean) with verified self-match, benchmark protocol documentation (LFW/IJB-C/NIST FRVT metrics), and honest limitation reporting. Not a production security system."

Key points to highlight (align with docs):
- Pipeline complete: load → detect → embed → compare → decision
- Verified: self-match distance 0.00, threshold 0.55, 1 face detected, 128-d embedding
- Metrics understood: FAR, FNMR, TAR@FAR, Rank-1; documented what is measured vs missing
- Use case: small consented event / organizer local identification
- Technical choices explained with alternatives (see METHOD.md)
- Enterprise extension described without overstating (see USE_CASE.md)

What NOT to claim:
- "99% accuracy" (no paired benchmark completed)
- "Production access control" (no liveness / anti-spoof)
- "Real-time webinar" (no deployment; local only)

Reference docs for interviewer questions: PROJECT_OVERVIEW.md, METHOD.md, EVALUATION.md, USE_CASE.md
