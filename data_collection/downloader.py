"""
downloader.py
─────────────
Step 1 of Phase 1 — YouTube video download.

Responsibilities:
  - Accept one or more YouTube URLs from CLI args or the URLS list
  - Download each video as a single merged .mp4 using yt-dlp
  - Cap resolution at --quality (default 1080p) so the file is large
    enough for MediaPipe to detect landmarks reliably
  - Name the file by its YouTube video ID (avoids duplicates)
  - Skip re-download if the file already exists
  - Clean up any leftover partial files from previous failed attempts
  - Log progress and a final summary

Output:
  dataset/downloads/{video_id}.mp4

Usage:
  python data_collection/downloader.py --quality 1080 URL1 URL2 ...
  python data_collection/downloader.py            # uses URLs list below
"""

import subprocess
import os
import sys
import argparse
import logging

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
DOWNLOADS_PATH = os.path.join("dataset", "downloads")
os.makedirs(DOWNLOADS_PATH, exist_ok=True)


# ── Core download function ────────────────────────────────────────────────────
def download(url: str, max_height: int = 1080) -> str:
    """
    Download a single YouTube video to dataset/downloads/.

    Args:
        url:        Full YouTube video URL.
        max_height: Cap on video stream height in pixels.
                    1080 → 1080p Full HD (recommended)
                    720  → 720p HD
                    480  → 480p SD
                    9999 → whatever is available (may be 4K)

    Returns:
        Absolute path to the downloaded .mp4 file.

    Raises:
        subprocess.CalledProcessError: If yt-dlp exits with a non-zero code.
        ValueError: If the video ID cannot be parsed from the URL.
        FileNotFoundError: If the output file is missing after download.
    """
    video_id = _parse_video_id(url)
    out_path = os.path.abspath(
        os.path.join(DOWNLOADS_PATH, f"{video_id}.mp4")
    )

    if os.path.exists(out_path):
        log.info("Already downloaded — skipping: %s", out_path)
        return out_path

    # Clean up any partial files from a previous failed attempt
    _cleanup_partial(video_id)

    # Format selector:
    #   bestvideo[height<=N]+bestaudio  → grab separate video + audio streams
    #                                    (needed for 1080p+ since YouTube
    #                                     does not serve them as a single MP4)
    #   /best[height<=N]                → fall back to a single-file MP4
    #                                    if a merged format is not available
    quality_format = (
        f"bestvideo[height<={max_height}]+bestaudio/"
        f"best[height<={max_height}]"
    )

    log.info("Downloading: %s", url)
    log.info("Quality cap: %dp   →   format: %s", max_height, quality_format)
    log.info("Saving to  : %s", out_path)

    subprocess.run(
        [
            "yt-dlp",
            "--format",               quality_format,
            "--merge-output-format",  "mp4",   # force mp4 container after merge
            "--output",               out_path,
            "--no-playlist",
            "--retries",              "5",
            "--fragment-retries",     "5",
            "--concurrent-fragments", "4",
            url,
        ],
        check=True,
    )

    if not os.path.exists(out_path):
        raise FileNotFoundError(
            f"yt-dlp reported success but output file not found: {out_path}"
        )

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    log.info("Saved: %s  (%.1f MB)", out_path, size_mb)
    return out_path


def download_many(urls: list[str], max_height: int = 1080) -> dict:
    """
    Download a list of YouTube URLs.

    Returns:
        Dict mapping each URL to its local path (str) or the Exception
        raised if it failed.
    """
    results = {}
    for i, url in enumerate(urls, 1):
        log.info("[%d/%d]  %s", i, len(urls), url)
        try:
            results[url] = download(url, max_height=max_height)
        except Exception as exc:
            log.error("FAILED: %s — %s", url, exc)
            results[url] = exc
    return results


# ── URL parsing ───────────────────────────────────────────────────────────────
def _parse_video_id(url: str) -> str:
    """
    Extract the YouTube video ID from standard URL formats.

    Supported:
      https://www.youtube.com/watch?v=VIDEO_ID
      https://youtu.be/VIDEO_ID
      https://www.youtube.com/shorts/VIDEO_ID
      Bare 11-char video ID
    """
    url = url.strip()

    if "v=" in url:
        video_id = url.split("v=")[-1].split("&")[0].split("#")[0]
        if video_id:
            return video_id

    if "youtu.be/" in url:
        video_id = url.split("youtu.be/")[-1].split("?")[0].split("#")[0]
        if video_id:
            return video_id

    if "shorts/" in url:
        video_id = url.split("shorts/")[-1].split("?")[0].split("#")[0]
        if video_id:
            return video_id

    # bare 11-char id
    if len(url) == 11 and url.replace("-", "").replace("_", "").isalnum():
        return url

    raise ValueError(f"Could not parse YouTube video ID from URL: {url}")


# ── Partial file cleanup ──────────────────────────────────────────────────────
def _cleanup_partial(video_id: str) -> None:
    """
    Remove any leftover partial files from a previous failed download
    (e.g. VIDEO_ID.f137.mp4, VIDEO_ID.f140.m4a, VIDEO_ID.mp4.part).
    """
    for fname in os.listdir(DOWNLOADS_PATH):
        if fname.startswith(video_id) and fname != f"{video_id}.mp4":
            fpath = os.path.join(DOWNLOADS_PATH, fname)
            try:
                os.remove(fpath)
                log.info("Removed partial file: %s", fname)
            except OSError as exc:
                log.warning(
                    "Could not remove partial file %s: %s", fname, exc
                )


# ── Default URL list ──────────────────────────────────────────────────────────
# Add YouTube URLs here before running without CLI args.
URLS: list[str] = [
    "https://youtu.be/bL1nPCHVYaQ",
    "https://youtu.be/ZqtWC7Mazd0",
]


# ── CLI entry point ───────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download YouTube videos for NSL dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--quality", type=int, default=1080,
        help="Max video height in pixels (default 1080). "
             "Use 720 for smaller files, 9999 for best available.",
    )
    parser.add_argument(
        "urls", nargs="*",
        help="YouTube URLs to download. If omitted, uses URLS list.",
    )
    args = parser.parse_args()

    urls = args.urls if args.urls else URLS

    if not urls:
        log.warning("No URLs provided.")
        log.warning("Either add them to the URLS list at the top of this")
        log.warning("file, or pass them as command-line arguments:")
        log.warning("    python data_collection/downloader.py URL1 URL2 ...")
        sys.exit(0)

    if args.quality not in (480, 720, 1080, 1440, 2160, 9999):
        log.warning(
            "Unusual quality value %d. Valid common values: "
            "480, 720, 1080, 1440, 2160, 9999 (no cap).",
            args.quality,
        )

    results = download_many(urls, max_height=args.quality)

    # Summary
    succeeded = [u for u, r in results.items() if isinstance(r, str)]
    failed    = [u for u, r in results.items() if isinstance(r, Exception)]

    log.info("─" * 60)
    log.info(
        "Download complete.  Success: %d  Failed: %d",
        len(succeeded), len(failed),
    )
    if failed:
        log.error("Failed URLs:")
        for u in failed:
            log.error("  %s", u)
        sys.exit(1)


if __name__ == "__main__":
    main()