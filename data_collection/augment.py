"""
augment.py
──────────
Step 5 of Phase 1 — Dataset augmentation.

Augmentations applied per original sequence:
  1. Gaussian noise     — adds small random noise to all landmark values
  2. Mirror flip        — flips x-axis, swaps left/right hand landmark blocks
  3. Time warp          — randomly stretches or compresses the sequence in
                          time then resamples back to 30 frames

Each augmented variant is saved alongside the original in the same folder:
  sequence_aug_noise.npy
  sequence_aug_flip.npy
  sequence_aug_warp.npy

Augmentations are never applied to already-augmented files (files whose
name contains "_aug_") to avoid compounding distortions.
Re-running is safe — existing augmented files are overwritten with fresh
variants (new random seeds each run).
"""

import os
import logging
import numpy as np
from scipy.interpolate import interp1d

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
DATASET_RAW_PATH = os.path.join("dataset", "raw")

# ── Constants ─────────────────────────────────────────────────────────────────
SEQUENCE_LENGTH = 30
FEATURE_SIZE    = 1662

# Landmark block boundaries (must match extract_landmarks.py)
POSE_END   = 33 * 4          # 132  — pose ends here
FACE_END   = POSE_END + 468 * 3   # 1536 — face ends here
LH_START   = FACE_END              # 1536 — left hand starts
LH_END     = LH_START + 21 * 3    # 1599 — left hand ends
RH_START   = LH_END                # 1599 — right hand starts
RH_END     = RH_START + 21 * 3    # 1662 — right hand ends


# ── Augmentation functions ────────────────────────────────────────────────────

def gaussian_noise(
    sequence: np.ndarray,
    std: float = 0.004,
) -> np.ndarray:
    """
    Add zero-mean Gaussian noise to every landmark value.

    std=0.004 is roughly 0.4% of the normalised coordinate range [0, 1].
    Large enough to be a meaningful perturbation, small enough not to
    distort the sign shape.
    """
    noise = np.random.normal(0.0, std, sequence.shape).astype(np.float32)
    return sequence + noise


def mirror_flip(sequence: np.ndarray) -> np.ndarray:
    """
    Horizontally mirror the signer.

    Steps:
      1. Negate x-coordinates for pose landmarks (stride 4, starting at 0)
      2. Negate x-coordinates for face landmarks (stride 3, starting at POSE_END)
      3. Swap the left-hand and right-hand landmark blocks entirely

    This simulates a signer standing on the opposite side, which doubles
    effective data diversity for asymmetric signs.
    """
    flipped = sequence.copy()

    # Pose: x is at indices 0, 4, 8, ... (every 4th starting at 0)
    flipped[:, 0:POSE_END:4] = 1.0 - flipped[:, 0:POSE_END:4]

    # Face: x is at indices POSE_END, POSE_END+3, ... (every 3rd)
    flipped[:, POSE_END:FACE_END:3] = 1.0 - flipped[:, POSE_END:FACE_END:3]

    # Swap left hand and right hand blocks
    lh = flipped[:, LH_START:LH_END].copy()
    rh = flipped[:, RH_START:RH_END].copy()
    flipped[:, LH_START:LH_END] = rh
    flipped[:, RH_START:RH_END] = lh

    return flipped


def time_warp(
    sequence:     np.ndarray,
    speed_range:  tuple[float, float] = (0.75, 1.25),
) -> np.ndarray:
    """
    Randomly speed up or slow down the sequence, then resample
    back to SEQUENCE_LENGTH frames using linear interpolation.

    speed_factor < 1.0 → slower (stretch): fewer source frames mapped to 30
    speed_factor > 1.0 → faster (compress): more source frames mapped to 30

    Range (0.75, 1.25) means the signer performs the sign in 75%–125% of
    the original time.
    """
    speed_factor = np.random.uniform(*speed_range)
    original_len = sequence.shape[0]
    warped_len   = max(2, int(original_len * speed_factor))

    t_original = np.linspace(0.0, 1.0, original_len)
    t_warped   = np.linspace(0.0, 1.0, warped_len)
    t_output   = np.linspace(0.0, 1.0, SEQUENCE_LENGTH)

    output = np.empty((SEQUENCE_LENGTH, FEATURE_SIZE), dtype=np.float32)

    for f in range(FEATURE_SIZE):
        # Step 1: interpolate to warped length
        fn1 = interp1d(t_original, sequence[:, f], kind="linear")
        warped_col = fn1(t_warped)

        # Step 2: resample warped back to SEQUENCE_LENGTH
        fn2 = interp1d(
            t_warped, warped_col, kind="linear",
            bounds_error=False,
            fill_value=(warped_col[0], warped_col[-1]),
        )
        output[:, f] = fn2(t_output)

    return output


# ── Dataset-level augmentation ────────────────────────────────────────────────
def augment_dataset(
    apply_noise: bool = True,
    apply_flip:  bool = True,
    apply_warp:  bool = True,
) -> None:
    """
    Walk dataset/raw/, find every original sequence.npy, and write
    up to three augmented variants alongside it.
    """
    if not os.path.exists(DATASET_RAW_PATH):
        log.error("dataset/raw/ not found. Run extract_landmarks.py first.")
        return

    total_processed = 0
    total_created   = 0

    for sign in sorted(os.listdir(DATASET_RAW_PATH)):
        sign_path = os.path.join(DATASET_RAW_PATH, sign)
        if not os.path.isdir(sign_path):
            continue

        for signer_id in os.listdir(sign_path):
            signer_path = os.path.join(sign_path, signer_id)
            if not os.path.isdir(signer_path):
                continue

            for seq_id in os.listdir(signer_path):
                seq_dir  = os.path.join(signer_path, seq_id)
                seq_file = os.path.join(seq_dir, "sequence.npy")

                if not os.path.exists(seq_file):
                    continue

                try:
                    seq = np.load(seq_file).astype(np.float32)
                except Exception as exc:
                    log.warning("Cannot load %s: %s", seq_file, exc)
                    continue

                if seq.shape != (SEQUENCE_LENGTH, FEATURE_SIZE):
                    log.warning(
                        "Wrong shape %s %s — skipping", seq.shape, seq_file
                    )
                    continue

                total_processed += 1

                if apply_noise:
                    np.save(
                        os.path.join(seq_dir, "sequence_aug_noise.npy"),
                        gaussian_noise(seq),
                    )
                    total_created += 1

                if apply_flip:
                    np.save(
                        os.path.join(seq_dir, "sequence_aug_flip.npy"),
                        mirror_flip(seq),
                    )
                    total_created += 1

                if apply_warp:
                    np.save(
                        os.path.join(seq_dir, "sequence_aug_warp.npy"),
                        time_warp(seq),
                    )
                    total_created += 1

    log.info("─" * 60)
    log.info(
        "Augmentation complete.  "
        "Original sequences: %d  New sequences: %d  Total: %d",
        total_processed,
        total_created,
        total_processed + total_created,
    )


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    augment_dataset(apply_noise=True, apply_flip=True, apply_warp=True)