"""
extract_landmarks.py
────────────────────
Step 3 of Phase 1 — MediaPipe landmark extraction.

Compatible with mediapipe >= 0.10 (uses mediapipe.tasks API).
Falls back gracefully to mp.solutions.holistic for older installs.

Clip handling strategy:
  - Clips with fewer than MIN_FRAMES (5) frames are skipped entirely.
  - Clips shorter than SEQUENCE_LENGTH (30) are padded to 30 frames:
      * first frame repeated at start, last frame repeated at end.
  - Clips longer than 30 frames produce overlapping sequences
    using STRIDE (default 15 = 50% overlap).
  - Resume-safe: already-extracted annotations are skipped on re-run
    via a fingerprint file.

Edge-case handling:
  - Per-frame keypoints are normalised AND validated (correct shape)
    immediately after extraction. Frames with the wrong shape are
    replaced with a zero vector of the expected size so the stack
    never fails.
  - Empty clip output (decoder failure, all-black video) is handled.
  - Negative / zero / over-large clip ranges are clipped safely.
  - np.stack is replaced with manual assembly to give a clear error
    if anything is still the wrong shape.

Output:
  dataset/raw/{sign_name}/{signer_id}/{seq_id}/sequence.npy
  shape: (30, 1662) float32
"""

import os
import json
import hashlib
import logging
import time
import pathlib
import cv2
import numpy as np
import mediapipe as mp

# suppress noisy MediaPipe / TensorFlow C++ logs
os.environ.setdefault("GLOG_minloglevel",     "3")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
ANNOTATIONS_FILE = os.path.join("dataset", "annotations.json")
DATASET_RAW_PATH = os.path.join("dataset", "raw")
DONE_LOG_FILE    = os.path.join("dataset", ".extracted_done.json")

# ── Sequence parameters ───────────────────────────────────────────────────────
SEQUENCE_LENGTH = 30
MIN_FRAMES      = 5
STRIDE          = 15

# ── Landmark geometry ─────────────────────────────────────────────────────────
N_POSE_LANDMARKS = 33
N_FACE_LANDMARKS = 468
N_HAND_LANDMARKS = 21
POSE_VALUES      = 4
XYZ_VALUES       = 3

FEATURE_SIZE = (
    N_POSE_LANDMARKS * POSE_VALUES     # 132
    + N_FACE_LANDMARKS * XYZ_VALUES    # 1404
    + N_HAND_LANDMARKS * XYZ_VALUES    # 63  left
    + N_HAND_LANDMARKS * XYZ_VALUES    # 63  right
)   # 1662

MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE  = 0.5


# ══════════════════════════════════════════════════════════════════════════════
#  RESUME TRACKING
# ══════════════════════════════════════════════════════════════════════════════
def _fingerprint(ann: dict) -> str:
    key = (
        f"{ann['video_path']}|{ann['sign_name']}|"
        f"{ann['signer_id']}|{ann['start_sec']}|{ann['end_sec']}"
    )
    return hashlib.md5(key.encode()).hexdigest()


def _load_done() -> set:
    if os.path.exists(DONE_LOG_FILE):
        try:
            with open(DONE_LOG_FILE) as f:
                return set(json.load(f))
        except (OSError, json.JSONDecodeError):
            return set()
    return set()


def _mark_done(fp: str, done: set) -> None:
    done.add(fp)
    os.makedirs("dataset", exist_ok=True)
    with open(DONE_LOG_FILE, "w") as f:
        json.dump(list(done), f, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
#  KEYPOINT NORMALISATION
#  This is the core defensive helper: every per-frame output goes through
#  here, and we are GUARANTEED to return a (1662,) float32 array
#  regardless of what the landmarker returned.
# ══════════════════════════════════════════════════════════════════════════════
def _safe_keypoint(raw, expected_size: int) -> np.ndarray:
    """
    Coerce whatever the landmarker returned into a flat float32 array
    of the exact expected length.  Anything weird → zeros.
    """
    if raw is None:
        return np.zeros(expected_size, dtype=np.float32)
    try:
        arr = np.asarray(raw, dtype=np.float32).flatten()
    except (TypeError, ValueError):
        return np.zeros(expected_size, dtype=np.float32)
    if arr.size != expected_size:
        # fix the rare case where the model returns a partial list
        out = np.zeros(expected_size, dtype=np.float32)
        n   = min(arr.size, expected_size)
        out[:n] = arr[:n]
        return out
    return arr


# ── Expected sub-sizes ────────────────────────────────────────────────────────
_POSE_SIZE = N_POSE_LANDMARKS * POSE_VALUES  # 132
_FACE_SIZE = N_FACE_LANDMARKS * XYZ_VALUES  # 1404
_HAND_SIZE = N_HAND_LANDMARKS * XYZ_VALUES  # 63


# ══════════════════════════════════════════════════════════════════════════════
#  MEDIAPIPE VERSION DETECTION
# ══════════════════════════════════════════════════════════════════════════════
def _detect_mp_version() -> str:
    try:
        _ = mp.solutions.holistic
        return "legacy"
    except AttributeError:
        return "tasks"


MP_MODE = _detect_mp_version()
log.info("MediaPipe mode: %s  (version %s)", MP_MODE, mp.__version__)


# ══════════════════════════════════════════════════════════════════════════════
#  LEGACY PATH  (mediapipe < 0.10)
# ══════════════════════════════════════════════════════════════════════════════
if MP_MODE == "legacy":
    _holistic_cls = mp.solutions.holistic.Holistic

    def _make_holistic():
        return _holistic_cls(
            static_image_mode=False,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        )

    def _process_frame_legacy(holistic, frame_bgr: np.ndarray) -> np.ndarray:
        try:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = holistic.process(rgb)
        except Exception as exc:
            log.warning("Holistic process error: %s", exc)
            return np.zeros(FEATURE_SIZE, dtype=np.float32)

        pose = (
            np.array([[lm.x, lm.y, lm.z, lm.visibility]
                      for lm in results.pose_landmarks.landmark],
                     dtype=np.float32).flatten()
            if results.pose_landmarks
            else np.zeros(_POSE_SIZE, dtype=np.float32)
        )
        face = (
            np.array([[lm.x, lm.y, lm.z]
                      for lm in results.face_landmarks.landmark],
                     dtype=np.float32).flatten()
            if results.face_landmarks
            else np.zeros(_FACE_SIZE, dtype=np.float32)
        )
        lh = (
            np.array([[lm.x, lm.y, lm.z]
                      for lm in results.left_hand_landmarks.landmark],
                     dtype=np.float32).flatten()
            if results.left_hand_landmarks
            else np.zeros(_HAND_SIZE, dtype=np.float32)
        )
        rh = (
            np.array([[lm.x, lm.y, lm.z]
                      for lm in results.right_hand_landmarks.landmark],
                     dtype=np.float32).flatten()
            if results.right_hand_landmarks
            else np.zeros(_HAND_SIZE, dtype=np.float32)
        )

        return _safe_keypoint(
            np.concatenate([pose, face, lh, rh]), FEATURE_SIZE)


# ══════════════════════════════════════════════════════════════════════════════
#  TASKS PATH  (mediapipe >= 0.10)
# ══════════════════════════════════════════════════════════════════════════════
else:
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision as mp_vision

    _MODEL_DIR = pathlib.Path("dataset") / ".mp_models"
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)

    _MODELS = {
        "pose": (
            "pose_landmarker_full.task",
            "https://storage.googleapis.com/mediapipe-models/"
            "pose_landmarker/pose_landmarker_full/float16/latest/"
            "pose_landmarker_full.task",
        ),
        "hand": (
            "hand_landmarker.task",
            "https://storage.googleapis.com/mediapipe-models/"
            "hand_landmarker/hand_landmarker/float16/latest/"
            "hand_landmarker.task",
        ),
        "face": (
            "face_landmarker.task",
            "https://storage.googleapis.com/mediapipe-models/"
            "face_landmarker/face_landmarker/float16/1/"
            "face_landmarker.task",
        ),
    }

    def _is_valid_task_file(path: pathlib.Path) -> bool:
        import zipfile
        if not path.exists() or path.stat().st_size == 0:
            return False
        try:
            with zipfile.ZipFile(path, "r") as zf:
                return zf.testzip() is None
        except Exception:
            return False

    def _get_retry_exceptions():
        import requests
        import requests.exceptions as rex
        catches = [requests.ConnectionError,
                   requests.Timeout, OSError, RuntimeError]
        if hasattr(rex, "ChunkedEncodingError"):
            catches.append(rex.ChunkedEncodingError)
        return tuple(catches)

    def _download_model(url: str, dest: pathlib.Path,
                        label: str) -> None:
        import requests
        retry_exceptions = _get_retry_exceptions()
        max_retries  = 5
        retry_delay  = 3
        chunk_size   = 1024 * 128

        for attempt in range(1, max_retries + 1):
            downloaded = dest.stat().st_size if dest.exists() else 0
            headers    = ({"Range": f"bytes={downloaded}-"}
                          if downloaded else {})

            if downloaded:
                log.info("Resuming %s from %.1f MB (attempt %d/%d)",
                         label, downloaded / 1_048_576,
                         attempt, max_retries)
            else:
                log.info("Downloading %s (attempt %d/%d) ...",
                         label, attempt, max_retries)

            try:
                resp = requests.get(url, headers=headers,
                                    stream=True, timeout=(10, 60))
                if resp.status_code == 416:
                    dest.unlink(missing_ok=True)
                    downloaded = 0
                    resp = requests.get(url, stream=True,
                                        timeout=(10, 60))
                resp.raise_for_status()

                total = (int(resp.headers.get("content-length", 0))
                         + downloaded)
                with open(dest, "ab" if downloaded else "wb") as fh:
                    for chunk in resp.iter_content(
                            chunk_size=chunk_size):
                        if not chunk:
                            continue
                        fh.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            pct    = downloaded / total * 100
                            mb     = downloaded / 1_048_576
                            tot    = total      / 1_048_576
                            filled = int(30 * downloaded / total)
                            bar    = ("█" * filled
                                      + "░" * (30 - filled))
                            print(f"\r  [{bar}] {pct:5.1f}%  "
                                  f"{mb:.2f}/{tot:.2f} MB",
                                  end="", flush=True)
                print()

                if not _is_valid_task_file(dest):
                    dest.unlink(missing_ok=True)
                    raise RuntimeError(
                        f"{label} failed ZIP integrity check.")

                log.info("Saved %s (%.2f MB)",
                         label, dest.stat().st_size / 1_048_576)
                return

            except retry_exceptions as exc:
                print()
                if attempt < max_retries:
                    log.warning("Error: %s — retrying in %ds ...",
                                exc, retry_delay)
                    time.sleep(retry_delay)
                    retry_delay = min(retry_delay * 2, 60)
                else:
                    dest.unlink(missing_ok=True)
                    raise RuntimeError(
                        f"Failed to download {label} after "
                        f"{max_retries} attempts."
                    ) from exc

    def _ensure_model(key: str) -> str:
        fname, url = _MODELS[key]
        dest = _MODEL_DIR / fname
        if _is_valid_task_file(dest):
            return str(dest)
        if dest.exists():
            dest.unlink()
        _download_model(url, dest, fname)
        return str(dest)

    def _build_pose_landmarker():
        base = mp_tasks.BaseOptions(
            model_asset_path=_ensure_model("pose"))
        opts = mp_vision.PoseLandmarkerOptions(
            base_options=base,
            running_mode=mp_vision.RunningMode.VIDEO,
            min_pose_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
            num_poses=1,
        )
        return mp_vision.PoseLandmarker.create_from_options(opts)

    def _build_hand_landmarker():
        base = mp_tasks.BaseOptions(
            model_asset_path=_ensure_model("hand"))
        opts = mp_vision.HandLandmarkerOptions(
            base_options=base,
            running_mode=mp_vision.RunningMode.VIDEO,
            min_hand_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
            num_hands=2,
        )
        return mp_vision.HandLandmarker.create_from_options(opts)

    def _build_face_landmarker():
        base = mp_tasks.BaseOptions(
            model_asset_path=_ensure_model("face"))
        opts = mp_vision.FaceLandmarkerOptions(
            base_options=base,
            running_mode=mp_vision.RunningMode.VIDEO,
            min_face_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
            num_faces=1,
        )
        return mp_vision.FaceLandmarker.create_from_options(opts)

    # ── Tasks path keypoint extraction (fully defensive) ──────────────────────
    def _extract_keypoints_tasks(pose_result, hand_result,
                                  face_result) -> np.ndarray:
        # Pose (33 × 4)
        if pose_result.pose_landmarks:
            try:
                lms = pose_result.pose_landmarks[0]
                pose = np.array(
                    [[lm.x, lm.y, lm.z,
                      getattr(lm, "visibility", 0.0)] for lm in lms],
                    dtype=np.float32).flatten()
            except Exception:
                pose = np.zeros(_POSE_SIZE, dtype=np.float32)
        else:
            pose = np.zeros(_POSE_SIZE, dtype=np.float32)

        # Face (468 × 3)
        if face_result.face_landmarks:
            try:
                lms = face_result.face_landmarks[0]
                face = np.array(
                    [[lm.x, lm.y, lm.z] for lm in lms],
                    dtype=np.float32).flatten()
            except Exception:
                face = np.zeros(_FACE_SIZE, dtype=np.float32)
        else:
            face = np.zeros(_FACE_SIZE, dtype=np.float32)

        # Hands (21 × 3 each)
        lh = np.zeros(_HAND_SIZE, dtype=np.float32)
        rh = np.zeros(_HAND_SIZE, dtype=np.float32)
        if hand_result.hand_landmarks:
            try:
                for idx, handedness_list in enumerate(
                        hand_result.handedness):
                    label = handedness_list[0].category_name
                    lms   = hand_result.hand_landmarks[idx]
                    arr   = np.array(
                        [[lm.x, lm.y, lm.z] for lm in lms],
                        dtype=np.float32).flatten()
                    if label == "Left":
                        lh = arr
                    else:
                        rh = arr
            except Exception:
                pass   # keep zeros

        # concatenate and normalise to exactly 1662 via _safe_keypoint
        combined = np.concatenate([pose, face, lh, rh])
        return _safe_keypoint(combined, FEATURE_SIZE)


# ══════════════════════════════════════════════════════════════════════════════
#  PADDING  — every element is guaranteed to be a (1662,) float32
#  after _safe_keypoint runs in the extraction function.
# ══════════════════════════════════════════════════════════════════════════════
def _pad_to_sequence_length(
    frames: list,         # list[np.ndarray], each (1662,) float32
    target: int = SEQUENCE_LENGTH,
) -> list:
    """
    Pad a list of (1662,) keypoint arrays to exactly `target` entries.
    First frame repeats at start, last frame at end.
    """
    if len(frames) == 0:
        return [np.zeros(FEATURE_SIZE, dtype=np.float32)] * target

    frames = [_safe_keypoint(f, FEATURE_SIZE) for f in frames]

    n = len(frames)
    if n >= target:
        return frames[:target]

    needed    = target - n
    pad_start = needed // 2
    pad_end   = needed - pad_start

    return (
        [frames[0]]   * pad_start
        + frames
        + [frames[-1]] * pad_end
    )


# ══════════════════════════════════════════════════════════════════════════════
#  SEQUENCE SLICING  — manual assembly, never relies on np.stack's
#  shape inference.  Every frame is re-validated before assembly.
# ══════════════════════════════════════════════════════════════════════════════
def _assemble_sequence(frames: list) -> np.ndarray:
    """
    Build a (SEQUENCE_LENGTH, FEATURE_SIZE) float32 array from a list
    of already-padded (1662,) frames.  Allocates a fixed-size buffer
    first and fills it manually — no shape inference, no ambiguity.
    """
    n = len(frames)
    out = np.zeros((n, FEATURE_SIZE), dtype=np.float32)
    for i, f in enumerate(frames):
        out[i] = _safe_keypoint(f, FEATURE_SIZE)
    return out


def _slice_sequences(keypoints: list) -> list:
    """
    Convert a flat list of per-frame keypoint arrays into a list of
    (SEQUENCE_LENGTH, FEATURE_SIZE) float32 arrays.
    """
    if not keypoints:
        return []

    # normalise every frame BEFORE any further processing
    keypoints = [_safe_keypoint(f, FEATURE_SIZE) for f in keypoints]
    n = len(keypoints)

    if n < MIN_FRAMES:
        return []

    if n < SEQUENCE_LENGTH:
        padded = _pad_to_sequence_length(keypoints)
        return [_assemble_sequence(padded)]

    # overlapping sliding window
    sequences = []
    for start in range(0, n - SEQUENCE_LENGTH + 1, STRIDE):
        chunk_frames = keypoints[start: start + SEQUENCE_LENGTH]
        sequences.append(_assemble_sequence(chunk_frames))
    return sequences


# ══════════════════════════════════════════════════════════════════════════════
#  CLIP EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════
def _extract_clip_keypoints(
    cap:         cv2.VideoCapture,
    start_frame: int,
    clip_len:    int,
    fps:         float,
) -> list:
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    keypoints = []

    if MP_MODE == "legacy":
        with _make_holistic() as holistic:
            frame_idx = start_frame
            for _ in range(clip_len):
                ret, frame = cap.read()
                if not ret:
                    break
                kp = _process_frame_legacy(holistic, frame)
                keypoints.append(_safe_keypoint(kp, FEATURE_SIZE))
                frame_idx += 1

    else:
        pose_lm = _build_pose_landmarker()
        hand_lm = _build_hand_landmarker()
        face_lm = _build_face_landmarker()
        try:
            frame_idx = start_frame
            for _ in range(clip_len):
                ret, frame = cap.read()
                if not ret:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB, data=rgb)
                ts_ms = int(frame_idx * 1000 / fps)

                try:
                    pose_res = pose_lm.detect_for_video(mp_image, ts_ms)
                except Exception as exc:
                    log.debug("pose error f%d: %s", frame_idx, exc)
                    pose_res = type("R", (), {"pose_landmarks": None})()
                try:
                    hand_res = hand_lm.detect_for_video(mp_image, ts_ms)
                except Exception as exc:
                    log.debug("hand error f%d: %s", frame_idx, exc)
                    hand_res = type("R", (), {
                        "hand_landmarks": [], "handedness": []})()
                try:
                    face_res = face_lm.detect_for_video(mp_image, ts_ms)
                except Exception as exc:
                    log.debug("face error f%d: %s", frame_idx, exc)
                    face_res = type("R", (), {"face_landmarks": None})()

                kp = _extract_keypoints_tasks(
                    pose_res, hand_res, face_res)
                keypoints.append(_safe_keypoint(kp, FEATURE_SIZE))
                frame_idx += 1
        finally:
            pose_lm.close()
            hand_lm.close()
            face_lm.close()

    return keypoints


# ══════════════════════════════════════════════════════════════════════════════
#  PER-ANNOTATION PROCESSING
# ══════════════════════════════════════════════════════════════════════════════
def process_annotation(ann: dict, done: set) -> int:
    fp = _fingerprint(ann)
    if fp in done:
        log.info("         → already extracted — skipping")
        return -1   # sentinel for "resumed"

    video_path = ann["video_path"]
    sign_name  = ann["sign_name"]
    signer_id  = ann["signer_id"]
    start_sec  = ann["start_sec"]
    end_sec    = ann["end_sec"]

    if not video_path or not os.path.exists(video_path):
        log.warning("Video not found, skipping: %s", video_path)
        return 0

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log.warning("Cannot open video, skipping: %s", video_path)
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    start_frame = int(start_sec * fps)
    end_frame   = int(end_sec   * fps)
    # defensive: clip to valid video range
    start_frame = max(0, min(start_frame, max(total_frames - 1, 0)))
    end_frame   = max(start_frame, min(end_frame, total_frames))
    clip_len    = end_frame - start_frame

    if clip_len < MIN_FRAMES:
        log.warning(
            "Clip too short (%d frames, min %d)  sign='%s'  skipping",
            clip_len, MIN_FRAMES, sign_name,
        )
        cap.release()
        return 0

    if clip_len < SEQUENCE_LENGTH:
        log.info(
            "         Clip has %d frames — padding %d to reach %d",
            clip_len, SEQUENCE_LENGTH - clip_len, SEQUENCE_LENGTH,
        )

    try:
        all_keypoints = _extract_clip_keypoints(
            cap, start_frame, clip_len, fps)
    except Exception as exc:
        log.error("Extraction failed for sign='%s': %s", sign_name, exc)
        cap.release()
        return 0
    finally:
        cap.release()

    sequences = _slice_sequences(all_keypoints)

    if not sequences:
        log.warning("No sequences produced  sign='%s'", sign_name)
        return 0

    seq_id = _next_seq_id(sign_name, signer_id)
    for chunk in sequences:
        _save_sequence(chunk, sign_name, signer_id, seq_id)
        seq_id += 1

    _mark_done(fp, done)
    return len(sequences)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _save_sequence(sequence: np.ndarray, sign_name: str,
                   signer_id: str, seq_id: int) -> None:
    save_dir = os.path.join(
        DATASET_RAW_PATH, sign_name, signer_id, str(seq_id))
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "sequence.npy"),
            sequence.astype(np.float32))


def _next_seq_id(sign_name: str, signer_id: str) -> int:
    base = os.path.join(DATASET_RAW_PATH, sign_name, signer_id)
    if not os.path.exists(base):
        return 0
    return len([
        d for d in os.listdir(base)
        if os.path.isdir(os.path.join(base, d))
    ])


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
def run() -> None:
    if not os.path.exists(ANNOTATIONS_FILE):
        log.error("annotations.json not found at %s", ANNOTATIONS_FILE)
        log.error("Run annotator.py first to create annotations.")
        return

    with open(ANNOTATIONS_FILE) as f:
        annotations = json.load(f)

    if not annotations:
        log.warning("annotations.json is empty — annotate some videos first.")
        return

    done = _load_done()

    log.info("Found %d annotation(s). Starting extraction…",
             len(annotations))
    log.info("Sequence length: %d  |  Stride: %d  |  Min frames: %d",
             SEQUENCE_LENGTH, STRIDE, MIN_FRAMES)
    if done:
        log.info("Resuming — %d annotation(s) already done.", len(done))
    log.info("─" * 60)

    total_saved   = 0
    total_skipped = 0
    total_resumed = 0

    for i, ann in enumerate(annotations, 1):
        log.info(
            "[%d/%d]  sign='%s'  signer='%s'  %.2fs – %.2fs",
            i, len(annotations),
            ann["sign_name"], ann["signer_id"],
            ann["start_sec"], ann["end_sec"],
        )
        try:
            result = process_annotation(ann, done)
        except Exception as exc:
            log.error("Unhandled error on annotation %d: %s", i, exc)
            result = 0

        if result == -1:
            total_resumed += 1
        elif result == 0:
            total_skipped += 1
        else:
            log.info("         → saved %d sequence(s)", result)
            total_saved += result

    log.info("─" * 60)
    log.info(
        "Done.  Saved: %d   Skipped: %d   Already done: %d",
        total_saved, total_skipped, total_resumed,
    )
    log.info("Output: %s", os.path.abspath(DATASET_RAW_PATH))


if __name__ == "__main__":
    run()