from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV2

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "static" / "real_sign_videos"
MODEL_DIR = BASE_DIR / "video_sign_language_model"
INDEX_PATH = MODEL_DIR / "reference_index.npz"
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".webm", ".mkv"}
IMG_SIZE = (224, 224)
FRAMES_PER_VIDEO = 16


def build_feature_extractor():
    base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights="imagenet")
    base_model.trainable = False
    return keras.Model(inputs=base_model.input, outputs=keras.layers.GlobalAveragePooling2D()(base_model.output))


def preprocess_frame(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, IMG_SIZE)
    rgb = rgb.astype("float32")
    rgb = keras.applications.mobilenet_v2.preprocess_input(rgb)
    return tf.expand_dims(rgb, 0)


def extract_video_embedding(video_path, feature_extractor):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    indices = np.linspace(0, max(total_frames - 1, 0), FRAMES_PER_VIDEO).astype(int).tolist()
    features = []
    for index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = cap.read()
        if not ok:
            continue
        features.append(feature_extractor.predict(preprocess_frame(frame), verbose=0)[0])
    cap.release()
    if not features:
        return None
    embedding = np.mean(np.array(features, dtype=np.float32), axis=0)
    norm = np.linalg.norm(embedding)
    return embedding / norm if norm else embedding


def main():
    if not DATA_DIR.is_dir():
        raise FileNotFoundError(f"Real video dataset not found: {DATA_DIR}")
    feature_extractor = build_feature_extractor()
    embeddings = []
    labels = []
    paths = []

    for label_dir in sorted([path for path in DATA_DIR.iterdir() if path.is_dir()]):
        for video_path in sorted([path for path in label_dir.iterdir() if path.suffix.lower() in VIDEO_EXTENSIONS]):
            print(f"Indexing {label_dir.name}: {video_path.name}")
            embedding = extract_video_embedding(video_path, feature_extractor)
            if embedding is None:
                continue
            embeddings.append(embedding)
            labels.append(label_dir.name)
            paths.append(str(video_path.relative_to(BASE_DIR)))

    if not embeddings:
        raise ValueError("No videos were indexed.")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        INDEX_PATH,
        embeddings=np.array(embeddings, dtype=np.float32),
        labels=np.array(labels),
        paths=np.array(paths),
    )
    print(f"Saved {len(embeddings)} reference videos to {INDEX_PATH}")


if __name__ == "__main__":
    main()
