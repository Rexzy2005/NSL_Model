"""
repair_dataset.py
─────────────────
Fixes two issues from the previous extractor run:

  1. Face-landmark count mismatch — old sequences have shape (30, 1692)
     because the pre-update MediaPipe produced 484 face points instead
     of 468. Re-extracts those annotations.

  2. Corrupt zero-byte .npy files left by crashed runs.

Usage:
  python data_collection/repair_dataset.py
"""

import os
import json
import hashlib
import numpy as np

ANNOTATIONS_FILE = "dataset/annotations.json"
DATASET_RAW_PATH = "dataset/raw"
DONE_LOG_FILE    = "dataset/.extracted_done.json"

EXPECTED_FEATURE_SIZE = 1662   # 132 + 1404 + 63 + 63


def fingerprint(ann: dict) -> str:
    key = (
        f"{ann['video_path']}|{ann['sign_name']}|"
        f"{ann['signer_id']}|{ann['start_sec']}|{ann['end_sec']}"
    )
    return hashlib.md5(key.encode()).hexdigest()


def main() -> None:
    # ── Step 1: delete corrupt / wrong-shape sequences ──────────────────────
    removed_corrupt = 0
    removed_wrong   = 0
    bad_fingerprints: set = set()

    for root, _dirs, files in os.walk(DATASET_RAW_PATH):
        for f in files:
            if not f.endswith(".npy"):
                continue
            path = os.path.join(root, f)
            try:
                arr = np.load(path)
                if arr.ndim != 2 or arr.shape[1] != EXPECTED_FEATURE_SIZE:
                    print(f"  wrong shape  {path}  {arr.shape}")
                    os.remove(path)
                    removed_wrong += 1
            except Exception as exc:
                print(f"  corrupt file  {path}  ({exc})")
                os.remove(path)
                removed_corrupt += 1
            finally:
                # remove the now-empty parent dir created by np.save
                pass

    # Clean up empty sequence directories
    for root, dirs, files in os.walk(DATASET_RAW_PATH, topdown=False):
        for d in dirs:
            full = os.path.join(root, d)
            if not os.listdir(full):
                os.rmdir(full)

    print(f"\nRemoved {removed_corrupt} corrupt file(s) "
          f"and {removed_wrong} wrong-shape file(s).")

    # ── Step 2: re-number sequence directories from 0 ────────────────────────
    if removed_corrupt or removed_wrong:
        for sign in os.listdir(DATASET_RAW_PATH):
            sign_dir = os.path.join(DATASET_RAW_PATH, sign)
            if not os.path.isdir(sign_dir):
                continue
            for signer in os.listdir(sign_dir):
                signer_dir = os.path.join(sign_dir, signer)
                if not os.path.isdir(signer_dir):
                    continue
                seq_dirs = sorted(
                    d for d in os.listdir(signer_dir)
                    if os.path.isdir(os.path.join(signer_dir, d))
                    and os.path.exists(
                        os.path.join(signer_dir, d, "sequence.npy"))
                )
                # renumber sequentially
                for new_idx, old_name in enumerate(seq_dirs):
                    if old_name != str(new_idx):
                        old_path = os.path.join(signer_dir, old_name)
                        new_path = os.path.join(signer_dir, str(new_idx))
                        os.rename(old_path, new_path)
                        print(f"  renumbered  {old_path}  →  {new_path}")

    # ── Step 3: remove fingerprints of annotations that no longer have
    #            any saved sequences (so they will be re-extracted) ──────────
    if os.path.exists(DONE_LOG_FILE):
        with open(DONE_LOG_FILE) as f:
            done = set(json.load(f))
    else:
        done = set()

    if not os.path.exists(ANNOTATIONS_FILE):
        print(f"\nCannot find {ANNOTATIONS_FILE} — re-run the annotator.")
        return

    with open(ANNOTATIONS_FILE) as f:
        annotations = json.load(f)

    # Build a set of (sign, signer) pairs that still have at least one seq
    have_data: set = set()
    for root, dirs, files in os.walk(DATASET_RAW_PATH):
        for d in dirs:
            if d.isdigit() and "sequence.npy" in os.listdir(
                    os.path.join(root, d)):
                # root is .../<signer_id>
                signer = os.path.basename(root)
                sign   = os.path.basename(os.path.dirname(root))
                have_data.add((sign, signer))

    cleared = 0
    for ann in annotations:
        fp = fingerprint(ann)
        if fp in done and (ann["sign_name"], ann["signer_id"]) not in have_data:
            done.discard(fp)
            cleared += 1

    with open(DONE_LOG_FILE, "w") as f:
        json.dump(list(done), f, indent=2)

    print(f"Cleared {cleared} stale fingerprint(s) from done-log.")
    print("\nNow run:  python data_collection/extract_landmarks.py")
    print("(the missing annotations will be re-extracted with shape (30, 1662))")


if __name__ == "__main__":
    main()