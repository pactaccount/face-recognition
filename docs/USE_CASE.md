# Use Case — Real Life + Enterprise Path

## Real life scenario (verified)
A meeting organizer at a small consented event collects one clear photo per attendee in a folder. At check-in, a test photo is matched locally. The system returns the attendee name or "Unknown." Nothing is transmitted; consent is explicit through file presence.

## Verified demonstration
- Folder: `event_demo/consented/Alice.jpg`
- Run: direct Python verification → 1 face, distance 0.00, label "Alice"
- Note: multi-label gallery and full paired evaluation require additional data (documented, not faked)

## Enterprise-grade extension (design only — not implemented)
| Layer | Current | Enterprise |
|---|---|---|
| Storage | Folder | Database (PostgreSQL / Milvus / FAISS) |
| Access | Local CLI | REST API + authentication (FastAPI + OAuth) |
| Audit | None | Access logs + match audit trail |
| Security | None | Liveness / depth / challenge-response |
| Scale | 1-10 labels | 1,000+ labels with indexing |
| Model | dlib 128-d | ArcFace / transformer + fine-tuning |

## Resume presentation
Reference `docs/PROJECT_OVERVIEW.md` + `docs/EVALUATION.md` + GitHub repo. Do not claim production accuracy; claim verified local pipeline, benchmark awareness, and honest limitation documentation.
