import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = BASE_DIR / "static" / "real_sign_videos"
DEFAULT_SAVE_DIR = BASE_DIR / "video_sign_language_model"
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".webm", ".mkv"}
IMG_SIZE = (224, 224)
FRAMES_PER_CLIP = 24


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


def extract_clip_features(video_path, feature_extractor, frames_per_clip=FRAMES_PER_CLIP):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    indices = np.linspace(0, max(total_frames - 1, 0), frames_per_clip).astype(int).tolist()
    features = []
    for index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = cap.read()
        if not ok:
            if features:
                features.append(features[-1])
            continue
        features.append(feature_extractor.predict(preprocess_frame(frame), verbose=0)[0])
    cap.release()
    if not features:
        return None
    while len(features) < frames_per_clip:
        features.append(features[-1])
    return np.array(features[:frames_per_clip], dtype=np.float32)


def load_video_dataset(data_dir, feature_extractor, max_per_class=None):
    labels = sorted([path.name for path in data_dir.iterdir() if path.is_dir()])
    X, y = [], []
    counts = {}
    for label_index, label in enumerate(labels):
        videos = sorted([path for path in (data_dir / label).iterdir() if path.suffix.lower() in VIDEO_EXTENSIONS])
        if max_per_class:
            videos = videos[:max_per_class]
        counts[label] = len(videos)
        for video in videos:
            features = extract_clip_features(video, feature_extractor)
            if features is not None:
                X.append(features)
                y.append(label_index)
    if not X:
        raise ValueError(f"No real training videos found in {data_dir}. Run collect_wlasl_subset.py first.")
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), labels, counts


def build_temporal_model(num_classes):
    model = keras.Sequential(
        [
            layers.Input(shape=(FRAMES_PER_CLIP, 1280)),
            layers.Masking(mask_value=0.0),
            layers.Bidirectional(layers.GRU(160, return_sequences=True)),
            layers.Dropout(0.35),
            layers.Bidirectional(layers.GRU(96)),
            layers.Dense(160, activation="relu"),
            layers.Dropout(0.25),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    parser = argparse.ArgumentParser(description="Train a temporal sign-language model from real motion videos.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--save-dir", type=Path, default=DEFAULT_SAVE_DIR)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-per-class", type=int, default=None)
    args = parser.parse_args()

    if not args.data_dir.is_dir():
        raise FileNotFoundError(f"Real video dataset folder not found: {args.data_dir}")

    from sklearn.model_selection import train_test_split

    feature_extractor = build_feature_extractor()
    X, y, labels, counts = load_video_dataset(args.data_dir, feature_extractor, args.max_per_class)
    stratify = y if min(np.bincount(y)) >= 2 else None
    test_size = 0.2
    if stratify is not None:
        test_size = max(test_size, len(labels) / len(y))
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=42, stratify=stratify)

    model = build_temporal_model(len(labels))
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    args.save_dir.mkdir(parents=True, exist_ok=True)
    model.save(args.save_dir / "model.keras")
    (args.save_dir / "labels.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")
    metrics = {
        "epochs": list(range(1, len(history.history.get("accuracy", [])) + 1)),
        "accuracy": [float(v) for v in history.history.get("accuracy", [])],
        "loss": [float(v) for v in history.history.get("loss", [])],
        "val_accuracy": [float(v) for v in history.history.get("val_accuracy", [])],
        "val_loss": [float(v) for v in history.history.get("val_loss", [])],
        "labels": labels,
        "class_counts": [counts[label] for label in labels],
    }
    (args.save_dir / "training_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved temporal video model to {args.save_dir}")


if __name__ == "__main__":
    main()
