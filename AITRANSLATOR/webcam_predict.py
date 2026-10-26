import os
import sys
import time

try:
    import cv2
except (ModuleNotFoundError, ImportError) as exc:
    missing = exc.name if hasattr(exc, "name") else str(exc)
    print(f"Error: required module '{missing}' is not installed.")
    print("Install OpenCV with: python -m pip install opencv-python")
    sys.exit(1)

try:
    import tensorflow as tf
    from tensorflow import keras
except (ModuleNotFoundError, ImportError) as exc:
    missing = exc.name if hasattr(exc, "name") else str(exc)
    print(f"Error: required module '{missing}' is not installed.")
    print("Install TensorFlow with: python -m pip install tensorflow")
    sys.exit(1)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "sign_language_model")
CLASSES_DIR = os.path.join(SCRIPT_DIR, "images for phrases")
IMAGE_SIZE = (224, 224)


def load_labels(model_dir: str, classes_dir: str | None = None) -> list[str]:
    labels_path = os.path.join(model_dir, "labels.txt")
    if os.path.isfile(labels_path):
        with open(labels_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    if classes_dir and os.path.isdir(classes_dir):
        return sorted(
            [name for name in os.listdir(classes_dir) if os.path.isdir(os.path.join(classes_dir, name))]
        )

    raise FileNotFoundError(
        "Could not load class labels. Provide labels.txt in the model directory or a valid classes folder."
    )


def preprocess_frame(frame, image_size):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, image_size)
    image = image.astype("float32")
    image = keras.applications.mobilenet_v2.preprocess_input(image)
    image = tf.expand_dims(image, 0)
    return image


def main():
    if not os.path.isdir(MODEL_DIR):
        raise FileNotFoundError(f"Saved model directory not found: {MODEL_DIR}")

    labels = load_labels(MODEL_DIR, CLASSES_DIR)

    # Use TFSMLayer for Keras 3 compatibility with SavedModel format
    model = keras.layers.TFSMLayer(MODEL_DIR, call_endpoint='serving_default')

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam.")
        return

    last_prediction = ""
    last_confidence = 0.0
    last_time = time.time()
    frame_count = 0

    print("Press 'q' to quit, 'p' to pause/resume predictions.")
    running = True
    predict_enabled = True

    while running:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from webcam.")
            break

        display_frame = frame.copy()
        cv2.putText(display_frame, "Press q to quit, p to pause/resume", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if predict_enabled and frame_count % 5 == 0:
            input_tensor = preprocess_frame(frame, IMAGE_SIZE)
            predictions = model(input_tensor)
            # TFSMLayer returns a dict, extract the predictions
            if isinstance(predictions, dict):
                pred_array = predictions['dense']  # or whatever the output key is
            else:
                pred_array = predictions
            scores = tf.nn.softmax(pred_array[0])
            idx = int(tf.argmax(scores).numpy())
            last_prediction = labels[idx]
            last_confidence = float(scores[idx])
            last_time = time.time()

        cv2.putText(display_frame, f"Prediction: {last_prediction}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Confidence: {last_confidence * 100:.1f}%", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Sign Language Webcam Prediction", display_frame)
        frame_count += 1

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            running = False
        elif key == ord("p"):
            predict_enabled = not predict_enabled
            status = "enabled" if predict_enabled else "paused"
            print(f"Prediction {status}.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
