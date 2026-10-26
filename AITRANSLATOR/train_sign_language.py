import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR / "images for phrases"
DEFAULT_SAVE_DIR = SCRIPT_DIR / "sign_language_model"
DEFAULT_H5_PATH = SCRIPT_DIR / "sign_language_model.h5"
IMG_SIZE = (224, 224)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif"}


def build_feature_extractor():
    base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights="imagenet")
    base_model.trainable = False
    return keras.Model(inputs=base_model.input, outputs=keras.layers.GlobalAveragePooling2D()(base_model.output))


def load_image_tensor(image_path):
    image_data = tf.io.read_file(str(image_path))
    image = tf.image.decode_image(image_data, channels=3, expand_animations=False)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32)
    image = keras.applications.mobilenet_v2.preprocess_input(image)
    return tf.expand_dims(image, axis=0)


def iter_image_paths(label_dir):
    for root, _, files in os.walk(label_dir):
        for filename in sorted(files):
            path = Path(root) / filename
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                yield path


def load_dataset(data_dir, feature_extractor, max_per_class=None):
    data_dir = Path(data_dir)
    labels = sorted([path.name for path in data_dir.iterdir() if path.is_dir()])
    if not labels:
        raise ValueError(f"No class folders found in {data_dir}")

    features = []
    targets = []
    class_counts = {}
    label_map = {label: index for index, label in enumerate(labels)}

    print("Label mapping:")
    for label, index in label_map.items():
        print(f"  {index:02d}: {label}")

    for label in labels:
        paths = list(iter_image_paths(data_dir / label))
        if max_per_class:
            paths = paths[:max_per_class]
        class_counts[label] = len(paths)
        for image_path in paths:
            image = load_image_tensor(image_path)
            extracted = feature_extractor.predict(image, verbose=0)
            features.append(extracted[0])
            targets.append(label_map[label])

    if not features:
        raise ValueError("No training images were loaded.")

    return np.array(features, dtype=np.float32), np.array(targets, dtype=np.int32), labels, class_counts


def build_classifier(num_classes):
    model = keras.Sequential(
        [
            layers.Input(shape=(1280,)),
            layers.Dense(384, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.35),
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


def save_artifacts(model, labels, history, class_counts, save_dir, h5_path):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    keras_path = save_dir / "model.keras"
    labels_path = save_dir / "labels.txt"
    metrics_path = save_dir / "training_metrics.json"

    model.save(keras_path)
    model.save(h5_path)
    labels_path.write_text("\n".join(labels) + "\n", encoding="utf-8")

    metrics = {
        "epochs": list(range(1, len(history.history.get("accuracy", [])) + 1)),
        "accuracy": [float(value) for value in history.history.get("accuracy", [])],
        "loss": [float(value) for value in history.history.get("loss", [])],
        "val_accuracy": [float(value) for value in history.history.get("val_accuracy", [])],
        "val_loss": [float(value) for value in history.history.get("val_loss", [])],
        "labels": labels[:12],
        "class_counts": [class_counts[label] for label in labels[:12]],
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Saved Keras model: {keras_path}")
    print(f"Saved H5 model: {h5_path}")
    print(f"Saved labels: {labels_path}")
    print(f"Saved dashboard metrics: {metrics_path}")


def main():
    parser = argparse.ArgumentParser(description="Train the sign-language TensorFlow CNN classifier.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--save-dir", type=Path, default=DEFAULT_SAVE_DIR)
    parser.add_argument("--h5-path", type=Path, default=DEFAULT_H5_PATH)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-per-class", type=int, default=None, help="Optional quick-demo cap for each class.")
    args = parser.parse_args()

    if not args.data_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {args.data_dir}")

    try:
        from sklearn.model_selection import train_test_split
    except ImportError:
        print("Install scikit-learn first: python -m pip install scikit-learn")
        sys.exit(1)

    print("Loading MobileNetV2 feature extractor...")
    feature_extractor = build_feature_extractor()

    print("Extracting image features...")
    X, y, labels, class_counts = load_dataset(args.data_dir, feature_extractor, args.max_per_class)
    print(f"Dataset: {X.shape[0]} samples, {len(labels)} classes")

    stratify = y if min(np.bincount(y)) >= 2 else None
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify,
    )

    model = build_classifier(len(labels))
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    save_artifacts(model, labels, history, class_counts, args.save_dir, args.h5_path)


if __name__ == "__main__":
    main()
