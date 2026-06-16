"""
verify_dataset.py
─────────────────
Step 4 of Phase 1 — Dataset validation.

Checks:
  1. Every sequence.npy exists and loads without error
  2. Shape is exactly (30, 1662)
  3. No more than 5 all-zero frames per sequence (MediaPipe dropout)
  4. No NaN or Inf values
  5. Class balance — printed as a bar chart to the terminal

Prints a full report and exits with code 1 if any hard errors are found.
"""

import os
import sys
import logging
import numpy as np
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
DATASET_RAW_PATH    = os.path.join("dataset", "raw")
SEQUENCE_LENGTH     = 30
FEATURE_SIZE        = 1662
MAX_ZERO_FRAMES     = 5       # sequences with more than this are flagged
BAR_WIDTH           = 35      # terminal bar chart width


# ── Verification ──────────────────────────────────────────────────────────────
def verify() -> bool:
    """
    Walk dataset/raw/ and validate every sequence.npy.

    Returns:
        True if no hard errors found, False otherwise.
    """
    if not os.path.exists(DATASET_RAW_PATH):
        log.error("dataset/raw/ does not exist. Run extract_landmarks.py first.")
        return False

    hard_errors:  list[str] = []
    warnings:     list[str] = []
    class_counts: dict[str, int] = defaultdict(int)
    signer_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    total_sequences = 0

    sign_names = sorted([
        d for d in os.listdir(DATASET_RAW_PATH)
        if os.path.isdir(os.path.join(DATASET_RAW_PATH, d))
    ])

    if not sign_names:
        log.error("No sign folders found in %s", DATASET_RAW_PATH)
        return False

    for sign in sign_names:
        sign_path = os.path.join(DATASET_RAW_PATH, sign)

        signers = sorted([
            d for d in os.listdir(sign_path)
            if os.path.isdir(os.path.join(sign_path, d))
        ])

        for signer_id in signers:
            signer_path = os.path.join(sign_path, signer_id)

            seq_ids = sorted([
                d for d in os.listdir(signer_path)
                if os.path.isdir(os.path.join(signer_path, d))
            ])

            for seq_id in seq_ids:
                seq_file = os.path.join(signer_path, seq_id, "sequence.npy")

                # ── 1. File exists ────────────────────────────────────────────
                if not os.path.exists(seq_file):
                    hard_errors.append(f"MISSING  {seq_file}")
                    continue

                # ── 2. Loads without error ────────────────────────────────────
                try:
                    seq = np.load(seq_file).astype(np.float32)
                except Exception as exc:
                    hard_errors.append(f"CORRUPT  {seq_file}  ({exc})")
                    continue

                # ── 3. Correct shape ──────────────────────────────────────────
                if seq.shape != (SEQUENCE_LENGTH, FEATURE_SIZE):
                    hard_errors.append(
                        f"SHAPE    {seq_file}  "
                        f"expected ({SEQUENCE_LENGTH}, {FEATURE_SIZE}), "
                        f"got {seq.shape}"
                    )
                    continue

                # ── 4. NaN / Inf check ────────────────────────────────────────
                if not np.isfinite(seq).all():
                    hard_errors.append(f"NAN/INF  {seq_file}")
                    continue

                # ── 5. All-zero frame check (MediaPipe dropout) ───────────────
                zero_frames = int(np.sum(np.all(seq == 0.0, axis=1)))
                if zero_frames > MAX_ZERO_FRAMES:
                    warnings.append(
                        f"DROPOUT  {seq_file}  "
                        f"{zero_frames}/{SEQUENCE_LENGTH} blank frames"
                    )

                class_counts[sign] += 1
                signer_counts[sign][signer_id] += 1
                total_sequences += 1

    # ── Report ────────────────────────────────────────────────────────────────
    _print_class_balance(class_counts)
    _print_signer_breakdown(signer_counts)

    log.info("─" * 60)
    log.info("Total sequences  : %d", total_sequences)
    log.info("Total signs      : %d", len(class_counts))
    log.info("Hard errors      : %d", len(hard_errors))
    log.info("Warnings         : %d", len(warnings))

    if warnings:
        log.info("─" * 60)
        log.warning("Warnings (high MediaPipe dropout):")
        for w in warnings:
            log.warning("  %s", w)

    if hard_errors:
        log.info("─" * 60)
        log.error("Hard errors found — fix these before training:")
        for e in hard_errors:
            log.error("  %s", e)
        return False

    log.info("─" * 60)
    log.info("Dataset looks clean.")
    return True


# ── Display helpers ───────────────────────────────────────────────────────────
def _print_class_balance(counts: dict[str, int]) -> None:
    if not counts:
        return

    log.info("─" * 60)
    log.info("Class balance:")

    max_count = max(counts.values())
    for sign, count in sorted(counts.items(), key=lambda x: -x[1]):
        bar = "█" * int((count / max_count) * BAR_WIDTH)
        log.info("  %-22s  %4d  |%s", sign, count, bar)


def _print_signer_breakdown(signer_counts: dict) -> None:
    log.info("─" * 60)
    log.info("Signer breakdown:")
    for sign in sorted(signer_counts):
        for signer_id, count in sorted(signer_counts[sign].items()):
            log.info("  %-22s  %-20s  %4d seqs", sign, signer_id, count)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ok = verify()
    sys.exit(0 if ok else 1)