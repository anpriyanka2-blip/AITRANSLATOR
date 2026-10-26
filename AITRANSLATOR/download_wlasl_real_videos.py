import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.request import urlretrieve

BASE_DIR = Path(__file__).resolve().parent
LABELS_PATH = BASE_DIR / "sign_language_model" / "labels.txt"
RAW_DIR = BASE_DIR / "datasets" / "wlasl_raw"
OUTPUT_DIR = BASE_DIR / "static" / "real_sign_videos"
METADATA_URL = "https://raw.githubusercontent.com/dxli94/WLASL/master/start_kit/WLASL_v0.3.json"


def load_labels(limit=None):
    labels = [line.strip().lower() for line in LABELS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    return labels[:limit] if limit else labels


def ensure_metadata(metadata_path):
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    if metadata_path.exists():
        return metadata_path
    print(f"Downloading WLASL metadata: {METADATA_URL}")
    urlretrieve(METADATA_URL, metadata_path)
    return metadata_path


def run_yt_dlp(url, output_template):
    command = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--quiet",
        "--no-warnings",
        "--ignore-errors",
        "--format",
        "mp4/best[ext=mp4]/best",
        "--output",
        str(output_template),
        url,
    ]
    return subprocess.run(command, check=False).returncode == 0


def find_downloaded_file(download_dir, video_id):
    for path in download_dir.glob(f"{video_id}.*"):
        if path.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov"}:
            return path
    return None


def download_subset(max_per_sign, label_limit=None):
    metadata_path = ensure_metadata(RAW_DIR / "WLASL_v0.3.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    labels = set(load_labels(label_limit))
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    counts = {label: 0 for label in labels}
    failures = []

    for item in metadata:
        gloss = item.get("gloss", "").strip().lower()
        if gloss not in labels:
            continue
        target_dir = OUTPUT_DIR / gloss
        target_dir.mkdir(parents=True, exist_ok=True)
        for instance in item.get("instances", []):
            if counts[gloss] >= max_per_sign:
                break
            video_id = str(instance.get("video_id", "")).strip()
            url = instance.get("url")
            if not video_id or not url:
                continue
            target = target_dir / f"{video_id}.mp4"
            if target.exists():
                counts[gloss] += 1
                continue

            temp_dir = RAW_DIR / gloss
            temp_dir.mkdir(parents=True, exist_ok=True)
            output_template = temp_dir / f"{video_id}.%(ext)s"
            print(f"Downloading {gloss}: {video_id}")
            ok = run_yt_dlp(url, output_template)
            downloaded = find_downloaded_file(temp_dir, video_id)
            if ok and downloaded:
                shutil.copy2(downloaded, target)
                counts[gloss] += 1
                print(f"  saved {target}")
            else:
                failures.append(f"{gloss},{video_id},{url}")
                print(f"  failed {video_id}")

    failure_path = OUTPUT_DIR / "download_failures.csv"
    if failures:
        failure_path.write_text("label,video_id,url\n" + "\n".join(failures), encoding="utf-8")

    print("\nDownloaded real videos:")
    for label in sorted(counts):
        print(f"{label}: {counts[label]}")
    print(f"Total: {sum(counts.values())}")


def main():
    parser = argparse.ArgumentParser(description="Download real WLASL motion videos for this app's available signs.")
    parser.add_argument("--max-per-sign", type=int, default=3)
    parser.add_argument("--label-limit", type=int, default=None, help="Optional limit for quick testing.")
    args = parser.parse_args()
    download_subset(args.max_per_sign, args.label_limit)


if __name__ == "__main__":
    main()
