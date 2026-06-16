"""
build_dataset.py
────────────────
Step 6 of Phase 1 — Build final processed arrays and train/val/test splits.

Responsibilities:
  - Walk dataset/raw/ and load every .npy file (original + augmented)
  - Separate held-out signers into a signer-independent pool
  - Stack all remaining sequences into X (N, 30, 1662) and y (N,)
  - Save X.npy, y.npy, label_map.json to dataset/processed/
  - Generate train / val / test splits
  - Save all splits to dataset/splits/
  - Print a full summary of shapes and class distribution

Signer-independent split:
  Any signer_id listed in HELD_OUT_SIGNERS goes directly to the
  independent test set and is never seen during training or validation.

Split strategy:
  If every class has at least 2 samples, use stratified 80/10/10.
  Otherwise fall back to a simple non-stratified split and warn the
  user that they need more data for reliable model training.
"""

import os
import sys
import json
import logging
import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
DATASET_RAW_PATH       = os.path.join("dataset", "raw")
DATASET_PROCESSED_PATH = os.path.join("dataset", "processed")
DATASET_SPLITS_PATH    = os.path.join("dataset", "splits")

# ── Constants ─────────────────────────────────────────────────────────────────
SEQUENCE_LENGTH = 30
FEATURE_SIZE    = 1662
RANDOM_STATE    = 42

# ── Signer-independent held-out set ──────────────────────────────────────────
HELD_OUT_SIGNERS: list[str] = [
    # "yt_signer_02",
]


# ── Safe split helper ────────────────────────────────────────────────────────
def _safe_split_stratified(X, y, test_size, random_state):
    """
    Wrap sklearn's train_test_split.  Falls back to a non-stratified
    split if any class has fewer than 2 members (sklearn requires
    at least 2 per class for stratification).
    """
    n_classes = len(np.unique(y))
    counts    = np.bincount(y)
    min_count = int(counts.min()) if len(counts) else 0

    if min_count < 2:
        log.warning(
            "Some classes have only %d sample(s) — "
            "falling back to non-stratified split. "
            "Collect more data for stratified splits.",
            min_count,
        )
        return train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
        )

    return train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )


# ── Build ─────────────────────────────────────────────────────────────────────
def build() -> None:
    if not os.path.exists(DATASET_RAW_PATH):
        log.error("dataset/raw/ not found. Run extract_landmarks.py first.")
        sys.exit(1)

    os.makedirs(DATASET_PROCESSED_PATH, exist_ok=True)
    os.makedirs(DATASET_SPLITS_PATH,    exist_ok=True)

    # ── Discover sign classes ─────────────────────────────────────────────────
    sign_names = sorted([
        d for d in os.listdir(DATASET_RAW_PATH)
        if os.path.isdir(os.path.join(DATASET_RAW_PATH, d))
    ])

    if not sign_names:
        log.error("No sign folders found in %s", DATASET_RAW_PATH)
        sys.exit(1)

    label_map: dict[str, int] = {name: idx for idx, name in enumerate(sign_names)}
    log.info("Found %d sign classes: %s", len(sign_names), sign_names)

    # ── Collect sequences ─────────────────────────────────────────────────────
    seqs_dep:  list[np.ndarray] = []
    lbls_dep:  list[int]        = []
    seqs_indp: list[np.ndarray] = []
    lbls_indp: list[int]        = []

    skipped         = 0
    per_class_dep:  dict[str, int] = defaultdict(int)
    per_class_indp: dict[str, int] = defaultdict(int)

    for sign in sign_names:
        label     = label_map[sign]
        sign_path = os.path.join(DATASET_RAW_PATH, sign)

        for signer_id in os.listdir(sign_path):
            signer_path = os.path.join(sign_path, signer_id)
            if not os.path.isdir(signer_path):
                continue

            is_held_out = signer_id in HELD_OUT_SIGNERS

            for seq_id in os.listdir(signer_path):
                seq_dir = os.path.join(signer_path, seq_id)
                if not os.path.isdir(seq_dir):
                    continue

                for fname in sorted(os.listdir(seq_dir)):
                    if not fname.endswith(".npy"):
                        continue

                    fpath = os.path.join(seq_dir, fname)
                    try:
                        seq = np.load(fpath).astype(np.float32)
                    except Exception as exc:
                        log.warning("Cannot load %s: %s — skipping", fpath, exc)
                        skipped += 1
                        continue

                    if seq.shape != (SEQUENCE_LENGTH, FEATURE_SIZE):
                        log.warning(
                            "Wrong shape %s in %s — skipping", seq.shape, fpath
                        )
                        skipped += 1
                        continue

                    if is_held_out:
                        seqs_indp.append(seq)
                        lbls_indp.append(label)
                        per_class_indp[sign] += 1
                    else:
                        seqs_dep.append(seq)
                        lbls_dep.append(label)
                        per_class_dep[sign] += 1

    # ── Convert to arrays (guard against empty list) ─────────────────────────
    if seqs_dep:
        X_dep = np.stack(seqs_dep, axis=0).astype(np.float32)
        y_dep = np.array(lbls_dep, dtype=np.int32)
    else:
        X_dep = np.zeros((0, SEQUENCE_LENGTH, FEATURE_SIZE), dtype=np.float32)
        y_dep = np.zeros((0,), dtype=np.int32)

    if seqs_indp:
        X_indp = np.stack(seqs_indp, axis=0).astype(np.float32)
        y_indp = np.array(lbls_indp, dtype=np.int32)
    else:
        X_indp = np.zeros((0, SEQUENCE_LENGTH, FEATURE_SIZE), dtype=np.float32)
        y_indp = np.zeros((0,), dtype=np.int32)

    log.info("─" * 60)
    log.info("Signer-dependent pool   : %s", X_dep.shape)
    log.info("Signer-independent pool : %s", X_indp.shape)
    log.info("Skipped files           : %d", skipped)

    if X_dep.shape[0] == 0:
        log.error("No valid sequences found in the dependent pool. Aborting.")
        sys.exit(1)

    # ── Save full processed arrays ────────────────────────────────────────────
    np.save(os.path.join(DATASET_PROCESSED_PATH, "X.npy"), X_dep)
    np.save(os.path.join(DATASET_PROCESSED_PATH, "y.npy"), y_dep)

    with open(os.path.join(DATASET_PROCESSED_PATH, "label_map.json"), "w") as f:
        json.dump(label_map, f, indent=2)

    log.info("Saved X.npy %s and y.npy to %s", X_dep.shape, DATASET_PROCESSED_PATH)

    # ── Signer-dependent splits ───────────────────────────────────────────────
    if X_dep.shape[0] < 3:
        log.error(
            "Need at least 3 samples in the dependent pool to split. "
            "Have %d.", X_dep.shape[0],
        )
        sys.exit(1)

    # First split: 80% train, 20% remainder
    X_train, X_tmp, y_train, y_tmp = _safe_split_stratified(
        X_dep, y_dep, test_size=0.20, random_state=RANDOM_STATE,
    )

    # Second split: half the remainder to val, half to test (= 10/10 of total)
    X_val, X_test_dep, y_val, y_test_dep = _safe_split_stratified(
        X_tmp, y_tmp, test_size=0.50, random_state=RANDOM_STATE,
    )

    # ── Save splits ───────────────────────────────────────────────────────────
    splits = {
        "X_train":          X_train,
        "y_train":          y_train,
        "X_val":            X_val,
        "y_val":            y_val,
        "X_test_dependent": X_test_dep,
        "y_test_dependent": y_test_dep,
    }

    for name, arr in splits.items():
        path = os.path.join(DATASET_SPLITS_PATH, f"{name}.npy")
        np.save(path, arr)

    if X_indp.shape[0] > 0:
        np.save(os.path.join(DATASET_SPLITS_PATH, "X_test_independent.npy"),
                X_indp)
        np.save(os.path.join(DATASET_SPLITS_PATH, "y_test_independent.npy"),
                y_indp)

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info("─" * 60)
    log.info("Split summary:")
    log.info("  Train               : %s", X_train.shape)
    log.info("  Val                 : %s", X_val.shape)
    log.info("  Test (dependent)    : %s", X_test_dep.shape)
    log.info("  Test (independent)  : %s", X_indp.shape)
    log.info("─" * 60)
    log.info("Class distribution (dependent pool):")

    max_count = max(per_class_dep.values()) if per_class_dep else 1
    for sign, count in sorted(per_class_dep.items(), key=lambda x: -x[1]):
        bar = "█" * int((count / max_count) * 30)
        log.info("  %-22s  %4d  |%s", sign, count, bar)

    log.info("─" * 60)
    log.info("Build complete. Outputs:")
    log.info("  %s", os.path.abspath(DATASET_PROCESSED_PATH))
    log.info("  %s", os.path.abspath(DATASET_SPLITS_PATH))


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    build()