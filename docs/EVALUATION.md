# Evaluation — Metrics, Protocol, Results

## Metrics selected (and why for this use case)
| Metric | Meaning for local event match | Source / standard |
|---|---|---|
| Verification distance | Self-match / cross-match distance value | dlib / face_recognition |
| FAR (False Accept Rate) | Stranger wrongly labeled as attendee (privacy risk) | IJB-C / NIST FRVT |
| FNMR (False Non-Match Rate) | Real attendee missed (inclusion risk) | NIST FRVT 2024/2025 |
| TAR @ FAR=1e-4 / 1e-6 | True accepts at strict security points | IJB-C benchmark |
| Rank-1 ID | Correct identity at top of gallery | MegaFace / IJB |

## Protocol reference (LFW — documented; full download blocked)
- Dataset: ~13,000 face images / ~5,700 identities
- Verification: 6,000 predefined pairs, 10-fold cross-validation
- Standard reporting: mean verification accuracy ± std; modern extensions include TAR @ fixed FAR

## Verified results (real run, no invention)
- Self-match (Alice.jpg): distance 0.0000 → match (<= 0.55)
- Face count: 1 (exactly one — passes validation)
- Embedding length: 128 (standard dlib)
- Unit test: pass (`test_matching`)

## Not computed (honest — requires data not present)
- Paired verification over 6,000 LFW pairs (needs download)
- FAR / FNMR curves (needs negative pairs / multiple identities)
- Full Rank-1 gallery search (needs >1 labeled reference at scale)
- Demographic differentials (needs labeled demographic annotations)
- Latency at scale (needs repeated measurement with video input)

## How this adds resume value
- Shows understanding of benchmark protocols (LFW, IJB-C, NIST FRVT)
- Shows ability to compute and document real results without overclaiming
- Documents exactly what is needed to reach production metrics (dataset, pairs, demographic labels)
