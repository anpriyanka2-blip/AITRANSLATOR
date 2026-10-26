import argparse
import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.applications import MobileNetV2

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif"}

# ----------------------------
# FEATURE EXTRACTOR
# ----------------------------
IMG_SIZE = (224, 224)

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False
feature_extractor = keras.Model(
    inputs=base_model.input,
    outputs=keras.layers.GlobalAveragePooling2D()(base_model.output)
)


# ----------------------------
# LOAD CLASS NAMES
# ----------------------------
def load_class_names(model_dir: str, classes_dir: str | None = None):
    labels_path = os.path.join(model_dir, "labels.txt")

    if os.path.isfile(labels_path):
        with open(labels_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    if classes_dir:
        classes_dir = os.path.abspath(classes_dir)
        if os.path.isdir(classes_dir):
            return sorted([
                d for d in os.listdir(classes_dir)
                if os.path.isdir(os.path.join(classes_dir, d))
            ])

    raise FileNotFoundError(
        "Class labels not found. Provide labels.txt or valid dataset folder."
    )


# ----------------------------
# IMAGE PREPROCESSING
# ----------------------------
def preprocess_image(image_path, image_size):
    image_data = tf.io.read_file(image_path)
    img = tf.image.decode_image(image_data, channels=3, expand_animations=False)
    img = tf.image.resize(img, image_size)
    img = tf.cast(img, tf.float32)
    img = keras.applications.mobilenet_v2.preprocess_input(img)
    img = tf.expand_dims(img, axis=0)
    return img


# ----------------------------
# LOAD MODEL SAFELY
# ----------------------------
def load_model_safe(model_dir):
    model_dir = os.path.abspath(model_dir)

    # Case 1: SavedModel folder
    if os.path.isdir(model_dir):
        try:
            return keras.models.load_model(model_dir)
        except:
            pass

    # Case 2: model.keras file
    model_file = os.path.join(model_dir, "model.keras")
    if os.path.isfile(model_file):
        return keras.models.load_model(model_file)

    raise FileNotFoundError("Model not found in given directory.")


# ----------------------------
# PREDICT FUNCTION
# ----------------------------
def predict(model, image_tensor, class_names):
    features = feature_extractor.predict(image_tensor, verbose=0)
    predictions = model.predict(features, verbose=0)

    # If output already softmax → no need to apply softmax again
    probs = np.array(predictions[0])

    index = np.argmax(probs)
    confidence = float(probs[index])

    return class_names[index], confidence


def find_image_files(data_dir):
    for root, _, files in os.walk(data_dir):
        for name in sorted(files):
            if os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS:
                yield os.path.join(root, name)


def infer_label_from_path(image_path, classes_dir):
    rel = os.path.relpath(image_path, classes_dir)
    parts = rel.split(os.sep)
    return parts[0] if len(parts) > 1 else None


def predict_all_images(model, classes_dir, class_names, image_size):
    image_paths = list(find_image_files(classes_dir))
    if not image_paths:
        raise FileNotFoundError(f"No images found in classes directory: {classes_dir}")

    correct = 0
    total = 0
    print(f"Predicting {len(image_paths)} images from {classes_dir}")

    for image_path in image_paths:
        img = preprocess_image(image_path, image_size)
        label, confidence = predict(model, img, class_names)
        actual = infer_label_from_path(image_path, classes_dir)

        if actual is not None:
            total += 1
            if actual == label:
                correct += 1

        result = f"{image_path} -> predicted={label} ({confidence * 100:.2f}%)"
        if actual is not None:
            result += f", actual={actual}, {'OK' if actual == label else 'WRONG'}"
        print(result)

    if total:
        print(f"\nAccuracy: {correct}/{total} = {correct / total * 100:.2f}%")

    return correct, total


# ----------------------------
# MAIN
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default="sign_language_model")
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--all", action="store_true", help="Predict on all images found in --classes-dir")
    parser.add_argument("--classes-dir", type=str, default="images for phrases")
    parser.add_argument("--image-size", type=int, nargs=2, default=[224, 224])

    args = parser.parse_args()

    if args.all and args.image:
        raise ValueError("Use either --all or --image, not both.")
    if not args.all and not args.image:
        raise ValueError("Provide --image for a single prediction or --all to predict all dataset images.")

    model_dir = os.path.abspath(args.model_dir)

    # ----------------------------
    # VALIDATION
    # ----------------------------
    if not os.path.exists(model_dir):
        raise FileNotFoundError("Model directory not found")

    model = load_model_safe(model_dir)
    class_names = load_class_names(model_dir, args.classes_dir)

    if args.all:
        classes_dir = os.path.abspath(args.classes_dir)
        if not os.path.isdir(classes_dir):
            raise FileNotFoundError("Classes directory not found")
        predict_all_images(model, classes_dir, class_names, tuple(args.image_size))
        return

    image_path = os.path.abspath(args.image)
    if not os.path.isfile(image_path):
        raise FileNotFoundError("Image not found")

    # ----------------------------
    # PROCESS IMAGE
    # ----------------------------
    img = preprocess_image(image_path, tuple(args.image_size))

    # ----------------------------
    # PREDICTION
    # ----------------------------
    label, confidence = predict(model, img, class_names)

    # ----------------------------
    # OUTPUT
    # ----------------------------
    print("\n======================")
    print("IMAGE:", image_path)
    print("PREDICTED:", label)
    print("CONFIDENCE:", round(confidence * 100, 2), "%")
    print("======================\n")


if __name__ == "__main__":
    main()
