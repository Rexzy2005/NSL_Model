"""
download_models.py
──────────────────
Pre-downloads all three MediaPipe task model files needed by
extract_landmarks.py.  Run this once before extraction.

Usage:
  python download_models.py
"""

import pathlib
import time
import sys
import zipfile

MODELS = {
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
        "face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    ),
}

MODEL_DIR = pathlib.Path("dataset") / ".mp_models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def _is_valid_task_file(path: pathlib.Path) -> bool:
    """
    A .task file is a ZIP archive.
    Check that the file exists, is non-empty, and is a valid ZIP.
    """
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with zipfile.ZipFile(path, "r") as zf:
            bad = zf.testzip()   # returns None if all files are OK
            return bad is None
    except zipfile.BadZipFile:
        return False


def _get_retry_exceptions():
    import requests
    import requests.exceptions as rex
    catches = [
        requests.ConnectionError,
        requests.Timeout,
        OSError,
        RuntimeError,
    ]
    if hasattr(rex, "ChunkedEncodingError"):
        catches.append(rex.ChunkedEncodingError)
    return tuple(catches)


def download(key: str) -> None:
    fname       = MODELS[key][0]
    url         = MODELS[key][1]
    dest        = MODEL_DIR / fname
    max_retries = 5
    retry_delay = 3
    chunk_size  = 1024 * 128   # 128 KB

    # Already have a good file — skip
    if _is_valid_task_file(dest):
        print(f"[{key}]  Already downloaded and valid "
              f"({dest.stat().st_size / 1_048_576:.1f} MB) — skipping.")
        return

    # Corrupt or partial file present — remove it
    if dest.exists():
        print(f"[{key}]  Existing file is corrupt or incomplete — removing.")
        dest.unlink()

    import requests
    retry_exceptions = _get_retry_exceptions()

    for attempt in range(1, max_retries + 1):
        downloaded = dest.stat().st_size if dest.exists() else 0
        headers    = {}

        if downloaded:
            headers["Range"] = f"bytes={downloaded}-"
            print(f"[{key}]  Resuming from "
                  f"{downloaded / 1_048_576:.1f} MB "
                  f"(attempt {attempt}/{max_retries}) ...")
        else:
            print(f"[{key}]  Downloading {fname} "
                  f"(attempt {attempt}/{max_retries}) ...")

        try:
            resp = requests.get(
                url,
                headers=headers,
                stream=True,
                timeout=(10, 60),
            )

            # 416 = server has less than we already downloaded
            if resp.status_code == 416:
                print(f"[{key}]  Range rejected — restarting from 0.")
                dest.unlink(missing_ok=True)
                downloaded = 0
                resp = requests.get(url, stream=True, timeout=(10, 60))

            resp.raise_for_status()

            content_length = int(resp.headers.get("content-length", 0))
            total = content_length + downloaded
            mode  = "ab" if downloaded else "wb"

            with open(dest, mode) as fh:
                for chunk in resp.iter_content(chunk_size=chunk_size):
                    if not chunk:
                        continue
                    fh.write(chunk)
                    downloaded += len(chunk)

                    if total:
                        pct     = downloaded / total * 100
                        mb      = downloaded / 1_048_576
                        tot     = total      / 1_048_576
                        bar_len = 30
                        filled  = int(bar_len * downloaded / total)
                        bar     = "█" * filled + "░" * (bar_len - filled)
                        print(
                            f"\r  [{bar}] {pct:5.1f}%  "
                            f"{mb:.2f}/{tot:.2f} MB",
                            end="", flush=True,
                        )

            print()   # newline after bar

            # ── Validate ZIP integrity ─────────────────────────────────
            if not _is_valid_task_file(dest):
                dest.unlink(missing_ok=True)
                raise RuntimeError(
                    f"{fname} failed ZIP integrity check — "
                    "file is corrupt or incomplete."
                )

            final_mb = dest.stat().st_size / 1_048_576
            print(f"[{key}]  Saved and verified — "
                  f"{final_mb:.2f} MB  →  {dest}")
            return   # success

        except retry_exceptions as exc:
            print()
            if attempt < max_retries:
                print(f"[{key}]  Error: {exc}")
                print(f"[{key}]  Retrying in {retry_delay}s ...")
                time.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 60)
            else:
                dest.unlink(missing_ok=True)
                print(
                    f"\n[{key}]  FAILED after {max_retries} attempts.",
                    file=sys.stderr,
                )
                print(f"         Last error: {exc}", file=sys.stderr)
                sys.exit(1)


def main() -> None:
    try:
        import requests   # noqa: F401
    except ImportError:
        print("ERROR: 'requests' is not installed.")
        print("Run:  pip install requests")
        sys.exit(1)

    print("MediaPipe model downloader")
    print(f"Saving to: {MODEL_DIR.resolve()}")
    print("-" * 52)

    for key in ("pose", "hand", "face"):
        download(key)
        print()

    print("-" * 52)
    print("All models ready.")
    print("You can now run:  "
          "python data_collection/extract_landmarks.py")


if __name__ == "__main__":
    main()