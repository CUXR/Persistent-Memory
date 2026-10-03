# Face match threshold validation (issue #27)

`FACE_MATCH_L2_THRESHOLD` is the maximum **squared** L2 distance at which
`services/face_recog_local.py` treats a face as a known person. FAISS
`IndexFlatL2` returns squared L2; on unit-normalized embeddings that equals
`2 - 2 * cosine_similarity`. The previous prototype value was `1.0` (cosine >= 0.5).

**Chosen default: `1.5`** (cosine >= 0.25).

## Dataset

Labeled Faces in the Wild (LFW), the official verification `pairs.txt`: 10 folds of
300 same-person and 300 different-person pairs (6,000 pairs). 83 pairs were
dropped because no face reached the production detection score (0.7), leaving
5917 (2960 same, 2957 different).

## Methodology

`backend/scripts/face_threshold_sweep.py` embeds each image with the production
setup (InsightFace `buffalo_l`, CPU, det size 640x640, min detection score 0.7, largest
face per image), L2-normalizes, and computes squared L2 per pair. Definitions:

- **FAR**: fraction of different-person pairs at or under the threshold (wrongly matched).
- **FRR**: fraction of same-person pairs over the threshold (wrongly rejected).

Operating points were chosen on 9 folds and measured on the held-out fold (10-fold CV).
Full output: [`face_threshold_results.json`](face_threshold_results.json).

## Results

| Threshold | FAR | FRR |
|---|---|---|
| 0.20 | 0.00% | 99.56% |
| 0.30 | 0.00% | 97.53% |
| 0.40 | 0.00% | 91.05% |
| 0.50 | 0.00% | 77.84% |
| 0.60 | 0.00% | 59.32% |
| 0.70 | 0.00% | 39.36% |
| 0.80 | 0.00% | 22.97% |
| 0.90 | 0.00% | 11.99% |
| 1.00 | 0.00% | 7.20% |
| 1.10 | 0.00% | 4.80% |
| 1.20 | 0.00% | 3.58% |
| 1.30 | 0.00% | 3.07% |
| 1.40 | 0.00% | 2.91% |
| 1.50 | 0.00% | 2.77% |
| 1.55 | 0.07% | 2.77% |
| 1.60 | 0.14% | 2.77% |
| 1.70 | 0.71% | 2.64% |
| 1.80 | 4.50% | 2.47% |

| Target FAR | Threshold (all data) | FAR | FRR | Held-out FAR | Fold threshold std |
|---|---|---|---|---|---|
| 0.05 | 1.804 | 4.97% | 2.36% | 5.01% | 0.0018 |
| 0.01 | 1.715 | 0.98% | 2.64% | 0.91% | 0.0011 |
| 0.001 | 1.545 | 0.07% | 2.77% | 0.07% | 0.0032 |

Equal error rate: 2.57% at threshold 1.76.
The old threshold `1.0` gave 0.00% FAR but 7.2% FRR.

## Decision and tradeoffs

`1.5` measured 0 false accepts in 2,957 different-person pairs (0.00% FAR) at 2.77% FRR. It sits just below the 0.1% FAR target point (1.545), leaving a small margin before FAR starts to rise. The product question
is "who is this person", so a false accept (attaching memories to the wrong person)
is worse than a false reject (a duplicate Unknown person that can be merged later).
`1.0` is safer on FAR but rejects about 7% of true matches, fragmenting identities.

Caveats:

- Pairwise FAR understates real risk: the recognizer takes the nearest neighbour among
  all enrolled people, so false-accept probability grows roughly with gallery size.
  Lower the threshold for large galleries.
- 2,957 different-person pairs cannot resolve FAR below about 0.03%; the 0.1% figure
  is a handful of pairs, and 0.00% means "below that resolution", not zero.
- LFW is mostly frontal web photos. Re-run the sweep on captures from the target
  camera (lighting, angle, distance) before relying on these numbers in deployment.

## Reproduce

```sh
python backend/scripts/face_threshold_sweep.py --lfw-dir PATH/lfw --pairs PATH/pairs.txt
```
