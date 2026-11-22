"""FAR/FRR sweep of the face-match threshold on LFW verification pairs (issue #27).

Embeds each pair with the production model (InsightFace buffalo_l, same detector
settings and minimum detection score as services/face_recog_local.py), computes the
squared L2 distance between unit-normalized embeddings (what FAISS IndexFlatL2
returns), and reports FAR/FRR across thresholds.

    python backend/scripts/face_threshold_sweep.py --lfw-dir DIR --pairs DIR/pairs.txt

DIR contains one folder per person (the extracted lfw.tgz). Embeddings are cached
in --cache so reruns skip inference.
"""
import argparse
import json
import sys
import types
from pathlib import Path

import numpy as np

MIN_DETECTION_SCORE = 0.7
MODEL_NAME = "buffalo_l"
DET_SIZE = (640, 640)
FAR_TARGETS = (0.05, 0.01, 0.001)


def read_pairs(path: Path):
    """Return (image_a, image_b, same) for LFW's pairs.txt (10 folds x 300 same + 300 different)."""
    lines = [line.split() for line in path.read_text().splitlines() if line.strip()]
    folds, per_fold = map(int, lines[0])
    pairs = []
    for fold in range(folds):
        block = lines[1 + fold * 2 * per_fold: 1 + (fold + 1) * 2 * per_fold]
        for i, row in enumerate(block):
            if i < per_fold:
                name, a, b = row[0], int(row[1]), int(row[2])
                pairs.append((image(name, a), image(name, b), True, fold))
            else:
                (n1, a), (n2, b) = (row[0], int(row[1])), (row[2], int(row[3]))
                pairs.append((image(n1, a), image(n2, b), False, fold))
    return pairs


def image(name, index):
    return f"{name}/{name}_{index:04d}.jpg"


def embed_all(lfw_dir: Path, names, cache: Path):
    cached = dict(np.load(cache, allow_pickle=True)["data"].item()) if cache.exists() else {}
    todo = [n for n in names if n not in cached]
    if todo:
        # insightface imports matplotlib only for unused 3D visualisation helpers; stub
        # whichever pieces are missing from a broken install.
        for module in ("matplotlib.pyplot", "mpl_toolkits", "mpl_toolkits.mplot3d"):
            try:
                __import__(module)
            except ImportError:
                stub = types.ModuleType(module)
                stub.Axes3D = object
                sys.modules[module] = stub
        import cv2
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name=MODEL_NAME, providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=0, det_size=DET_SIZE)
        for i, name in enumerate(todo, 1):
            faces = [f for f in app.get(cv2.imread(str(lfw_dir / name))) if f.det_score >= MIN_DETECTION_SCORE]
            # The largest face is the subject of an LFW image.
            face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), default=None)
            cached[name] = None if face is None else np.asarray(face.embedding, dtype=np.float32)
            if i % 500 == 0:
                print(f"embedded {i}/{len(todo)}", file=sys.stderr)
        np.savez(cache, data=np.array(cached, dtype=object))
    return cached


def rates(same, diff, threshold):
    """FAR: different-person pairs matched. FRR: same-person pairs rejected."""
    return float((diff <= threshold).mean()), float((same > threshold).mean())


def threshold_for_far(diff, target):
    """Largest threshold whose FAR on `diff` does not exceed target."""
    ordered = np.sort(diff)
    allowed = int(np.floor(target * len(ordered)))
    return float(ordered[allowed - 1]) if allowed else float(ordered[0]) - 1e-6


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lfw-dir", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path("lfw_embeddings.npz"))
    parser.add_argument("--out", type=Path, default=Path("face_threshold_results.json"))
    args = parser.parse_args()

    pairs = read_pairs(args.pairs)
    embeddings = embed_all(args.lfw_dir, sorted({p[0] for p in pairs} | {p[1] for p in pairs}), args.cache)
    scored = [(p, embeddings[p[0]], embeddings[p[1]]) for p in pairs]
    usable = [(p, a / np.linalg.norm(a), b / np.linalg.norm(b)) for p, a, b in scored if a is not None and b is not None]
    distance = np.array([float(((a - b) ** 2).sum()) for _, a, b in usable])
    same_flag = np.array([p[2] for p, _, _ in usable])
    fold = np.array([p[3] for p, _, _ in usable])
    same, diff = distance[same_flag], distance[~same_flag]

    result = {
        "pairs_total": len(pairs), "pairs_usable": len(usable),
        "pairs_dropped_no_face": len(pairs) - len(usable),
        "same_pairs": len(same), "different_pairs": len(diff),
        "current_threshold_1.0": dict(zip(("far", "frr"), rates(same, diff, 1.0))),
    }
    grid = np.round(np.arange(0.2, 1.8001, 0.05), 2)
    result["sweep"] = [{"threshold": float(t), **dict(zip(("far", "frr"), rates(same, diff, t)))} for t in grid]

    fine = np.sort(distance)
    eer_t = min(fine, key=lambda t: abs(rates(same, diff, t)[0] - rates(same, diff, t)[1]))
    result["eer"] = {"threshold": float(eer_t), **dict(zip(("far", "frr"), rates(same, diff, eer_t)))}

    # Cross-validated operating points: choose on 9 folds, measure on the held-out fold.
    result["operating_points"] = {}
    for target in FAR_TARGETS:
        held_out, chosen = [], []
        for k in range(10):
            train, test = fold != k, fold == k
            t = threshold_for_far(distance[train & ~same_flag], target)
            far, frr = rates(distance[test & same_flag], distance[test & ~same_flag], t)
            chosen.append(t)
            held_out.append((far, frr))
        full_t = threshold_for_far(diff, target)
        result["operating_points"][str(target)] = {
            "threshold_all_data": full_t,
            **dict(zip(("far", "frr"), rates(same, diff, full_t))),
            "threshold_fold_mean": float(np.mean(chosen)), "threshold_fold_std": float(np.std(chosen)),
            "heldout_far_mean": float(np.mean([h[0] for h in held_out])),
            "heldout_frr_mean": float(np.mean([h[1] for h in held_out])),
        }
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "sweep"}, indent=2))


if __name__ == "__main__":
    main()
