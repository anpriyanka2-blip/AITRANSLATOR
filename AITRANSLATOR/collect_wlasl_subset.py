import argparse
import json
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_LABELS = BASE_DIR / "sign_language_model" / "labels.txt"
DEFAULT_OUTPUT = BASE_DIR / "static" / "real_sign_videos"
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".webm", ".mkv"}


def load_labels(labels_path):
    return [line.strip().lower() for line in labels_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_wlasl_metadata(metadata_path):
    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "data" in data:
        data = data["data"]
    return data


def find_video(video_id, videos_dir):
    for extension in VIDEO_EXTENSIONS:
        candidate = videos_dir / f"{video_id}{extension}"
        if candidate.is_file():
            return candidate
    matches = list(videos_dir.rglob(f"{video_id}.*"))
    for match in matches:
        if match.suffix.lower() in VIDEO_EXTENSIONS:
            return match
    return None


def copy_subset(metadata_path, videos_dir, labels_path, output_dir, max_per_sign):
    wanted = set(load_labels(labels_path))
    metadata = read_wlasl_metadata(metadata_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = {label: 0 for label in wanted}
    missing = []

    for item in metadata:
        gloss = item.get("gloss", "").lower().strip()
        if gloss not in wanted:
            continue
        for instance in item.get("instances", []):
            if counts[gloss] >= max_per_sign:
                break
            video_id = str(instance.get("video_id", "")).strip()
            source = find_video(video_id, videos_dir)
            if source is None:
                missing.append(video_id)
                continue
            target_dir = output_dir / gloss
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / f"{video_id}{source.suffix.lower()}"
            if not target.exists():
                shutil.copy2(source, target)
            counts[gloss] += 1

    return counts, missing


def main():
    parser = argparse.ArgumentParser(description="Collect real WLASL motion videos for the app's available signs.")
    parser.add_argument("--metadata", type=Path, required=True, help="Path to WLASL_v0.3.json")
    parser.add_argument("--videos-dir", type=Path, required=True, help="Path to WLASL processed videos folder")
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-per-sign", type=int, default=25)
    args = parser.parse_args()

    if not args.metadata.is_file():
        raise FileNotFoundError(f"Metadata file not found: {args.metadata}")
    if not args.videos_dir.is_dir():
        raise FileNotFoundError(f"Videos directory not found: {args.videos_dir}")

    counts, missing = copy_subset(args.metadata, args.videos_dir, args.labels, args.output_dir, args.max_per_sign)
    copied = sum(counts.values())
    print(f"Copied {copied} real sign videos into {args.output_dir}")
    for label, count in sorted(counts.items()):
        if count:
            print(f"{label}: {count}")
    if missing:
        missing_path = args.output_dir / "missing_wlasl_video_ids.txt"
        missing_path.write_text("\n".join(sorted(set(missing))), encoding="utf-8")
        print(f"Missing video IDs written to {missing_path}")


if __name__ == "__main__":
    main()
