import argparse
from pathlib import Path

import cv2

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_FRAMES_DIR = BASE_DIR / "images for phrases"
DEFAULT_OUTPUT_DIR = BASE_DIR / "static" / "generated_sign_videos"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


def safe_video_name(label):
    return "".join(char.lower() if char.isalnum() else "_" for char in label).strip("_") + ".mp4"


def collect_frames(label_dir):
    return sorted([path for path in label_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS])


def render_video(label, frames, output_path, size=(640, 480), fps=8):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not create video: {output_path}")

    for frame_path in frames:
        image = cv2.imread(str(frame_path))
        if image is None:
            continue
        canvas = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        cv2.rectangle(canvas, (0, size[1] - 58), (size[0], size[1]), (12, 24, 38), -1)
        cv2.putText(canvas, label, (18, size[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(canvas)

    writer.release()


def main():
    parser = argparse.ArgumentParser(description="Generate downloadable MP4 sign videos from existing sign image frames.")
    parser.add_argument("--frames-dir", type=Path, default=DEFAULT_FRAMES_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()

    if not args.frames_dir.is_dir():
        raise FileNotFoundError(f"Frames directory not found: {args.frames_dir}")

    generated = 0
    for label_dir in sorted([path for path in args.frames_dir.iterdir() if path.is_dir()]):
        frames = collect_frames(label_dir)
        if not frames:
            continue
        output_path = args.output_dir / safe_video_name(label_dir.name)
        render_video(label_dir.name, frames, output_path, size=(args.width, args.height), fps=args.fps)
        generated += 1
        print(f"Generated {output_path} from {len(frames)} frames")

    print(f"Done. Generated {generated} videos in {args.output_dir}")


if __name__ == "__main__":
    main()
