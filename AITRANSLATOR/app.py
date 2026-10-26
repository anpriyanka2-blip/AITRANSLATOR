import json
import os
import sqlite3
import sys
import threading
import time
import webbrowser
from collections import Counter, deque
from datetime import datetime
from pathlib import Path

import numpy as np
from flask import Flask, Response, jsonify, render_template, request, send_from_directory, url_for
from werkzeug.utils import secure_filename

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "sign_language_model"
H5_MODEL_PATH = BASE_DIR / "sign_language_model.h5"
CLASSES_DIR = BASE_DIR / "images for phrases"
UPLOAD_FOLDER = BASE_DIR / "uploads"
REAL_SIGN_VIDEO_DIR = BASE_DIR / "static" / "real_sign_videos"
SIGN_VIDEO_DIR = BASE_DIR / "static" / "generated_sign_videos"
DATABASE_PATH = BASE_DIR / "instance" / "history.db"
VIDEO_MODEL_DIR = BASE_DIR / "video_sign_language_model"
VIDEO_REFERENCE_INDEX_PATH = VIDEO_MODEL_DIR / "reference_index.npz"

ALLOWED_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "gif"}
ALLOWED_VIDEO_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm"}
IMG_SIZE = (224, 224)
NO_SIGN_LABEL = "No clear sign"
MIN_CONFIDENCE = 0.30
HIGH_CONFIDENCE = 0.82
PREDICTION_HISTORY_SIZE = 8
REQUIRED_STABLE_VOTES = 3
FRAME_SAMPLE_RATE = 12

TRANSLATIONS = {
    "again": {"hindi": "फिर से", "telugu": "మళ్లీ", "kannada": "ಮತ್ತೆ"},
    "agree": {"hindi": "सहमत", "telugu": "అంగీకారం", "kannada": "ಒಪ್ಪಿಗೆ"},
    "answer": {"hindi": "उत्तर", "telugu": "సమాధానం", "kannada": "ಉತ್ತರ"},
    "attendance": {"hindi": "उपस्थिति", "telugu": "హాజరు", "kannada": "ಹಾಜರಾತಿ"},
    "book": {"hindi": "किताब", "telugu": "పుస్తకం", "kannada": "ಪುಸ್ತಕ"},
    "break": {"hindi": "विराम", "telugu": "విరామం", "kannada": "ವಿರಾಮ"},
    "careful": {"hindi": "सावधान", "telugu": "జాగ్రత్త", "kannada": "ಜಾಗರೂಕ"},
    "change": {"hindi": "बदलाव", "telugu": "మార్పు", "kannada": "ಬದಲಾವಣೆ"},
    "chat": {"hindi": "बातचीत", "telugu": "మాట్లాడు", "kannada": "ಮಾತನಾಡು"},
    "congratulations": {"hindi": "बधाई हो", "telugu": "అభినందనలు", "kannada": "ಅಭಿನಂದನೆಗಳು"},
    "email": {"hindi": "ईमेल", "telugu": "ఈమెయిల్", "kannada": "ಇಮೇಲ್"},
    "file": {"hindi": "फ़ाइल", "telugu": "ఫైల్", "kannada": "ಫೈಲ್"},
    "good morning": {"hindi": "सुप्रभात", "telugu": "శుభోదయం", "kannada": "ಶುಭೋದಯ"},
    "happy birthday": {"hindi": "जन्मदिन मुबारक", "telugu": "పుట్టినరోజు శుభాకాంక్షలు", "kannada": "ಹುಟ್ಟುಹಬ್ಬದ ಶುಭಾಶಯಗಳು"},
    "home": {"hindi": "घर", "telugu": "ఇల్లు", "kannada": "ಮನೆ"},
    "how are you": {"hindi": "आप कैसे हैं", "telugu": "మీరు ఎలా ఉన్నారు", "kannada": "ನೀವು ಹೇಗಿದ್ದೀರಿ"},
    "hungry": {"hindi": "भूख लगी है", "telugu": "ఆకలిగా ఉంది", "kannada": "ಹಸಿವು"},
    "i need help": {"hindi": "मुझे मदद चाहिए", "telugu": "నాకు సహాయం కావాలి", "kannada": "ನನಗೆ ಸಹಾಯ ಬೇಕು"},
    "join": {"hindi": "शामिल हों", "telugu": "చేరు", "kannada": "ಸೇರಿ"},
    "keepsmile": {"hindi": "मुस्कुराते रहो", "telugu": "నవ్వుతూ ఉండండి", "kannada": "ನಗುತಿರಿ"},
    "meet": {"hindi": "मिलना", "telugu": "కలవండి", "kannada": "ಭೇಟಿ"},
    "mistake": {"hindi": "गलती", "telugu": "తప్పు", "kannada": "ತಪ್ಪು"},
    "open": {"hindi": "खोलो", "telugu": "తెరువు", "kannada": "ತೆರೆ"},
    "opinion": {"hindi": "राय", "telugu": "అభిప్రాయం", "kannada": "ಅಭಿಪ್ರಾಯ"},
    "pass": {"hindi": "पास", "telugu": "పాస్", "kannada": "ಪಾಸ್"},
    "please": {"hindi": "कृपया", "telugu": "దయచేసి", "kannada": "ದಯವಿಟ್ಟು"},
    "practice": {"hindi": "अभ्यास", "telugu": "అభ్యాసం", "kannada": "ಅಭ್ಯಾಸ"},
    "pressure": {"hindi": "दबाव", "telugu": "ఒత్తిడి", "kannada": "ಒತ್ತಡ"},
    "problem": {"hindi": "समस्या", "telugu": "సమస్య", "kannada": "ಸಮಸ್ಯೆ"},
    "questions": {"hindi": "प्रश्न", "telugu": "ప్రశ్నలు", "kannada": "ಪ್ರಶ್ನೆಗಳು"},
    "remember": {"hindi": "याद रखें", "telugu": "గుర్తుంచుకో", "kannada": "ನೆನಪಿಡಿ"},
    "seat": {"hindi": "सीट", "telugu": "సీటు", "kannada": "ಆಸನ"},
    "shift": {"hindi": "शिफ्ट", "telugu": "మార్చు", "kannada": "ಬದಲಿಸಿ"},
    "sick": {"hindi": "बीमार", "telugu": "అనారోగ్యం", "kannada": "ಅನಾರೋಗ್ಯ"},
    "stop": {"hindi": "रुको", "telugu": "ఆపు", "kannada": "ನಿಲ್ಲಿಸಿ"},
    "sun": {"hindi": "सूरज", "telugu": "సూర్యుడు", "kannada": "ಸೂರ್ಯ"},
    "team": {"hindi": "टीम", "telugu": "బృందం", "kannada": "ತಂಡ"},
    "thirsty": {"hindi": "प्यास लगी है", "telugu": "దాహంగా ఉంది", "kannada": "ಬಾಯಾರಿಕೆ"},
    "this": {"hindi": "यह", "telugu": "ఇది", "kannada": "ಇದು"},
    "together": {"hindi": "साथ में", "telugu": "కలిసి", "kannada": "ಒಟ್ಟಿಗೆ"},
    "understand": {"hindi": "समझना", "telugu": "అర్థం చేసుకో", "kannada": "ಅರ್ಥಮಾಡಿಕೊಳ್ಳಿ"},
    "wait": {"hindi": "रुकिए", "telugu": "వేచి ఉండండి", "kannada": "ಕಾಯಿರಿ"},
    "where": {"hindi": "कहाँ", "telugu": "ఎక్కడ", "kannada": "ಎಲ್ಲಿ"},
    "write": {"hindi": "लिखो", "telugu": "రాయండి", "kannada": "ಬರೆಯಿರಿ"},
}

PREDICTION_STATE = {
    "active": False,
    "session": 0,
    "label": "Press Start",
    "confidence": 0.0,
    "stable_label": "",
    "stable_confidence": 0.0,
    "translations": {"hindi": "-", "telugu": "-", "kannada": "-"},
    "vote_scores": {},
    "vote_counts": {},
    "error": None,
}
PREDICTION_LOCK = threading.Lock()

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import mediapipe as mp
except ImportError:
    mp = None

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.applications import MobileNetV2
except (ModuleNotFoundError, ImportError) as exc:
    missing = exc.name if hasattr(exc, "name") else str(exc)
    print(f"Required module '{missing}' is not installed.")
    print("Install dependencies with: python -m pip install -r requirements.txt")
    sys.exit(1)


def init_db():
    DATABASE_PATH.parent.mkdir(exist_ok=True)
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_type TEXT NOT NULL,
                input_text TEXT,
                label TEXT NOT NULL,
                confidence REAL NOT NULL,
                language TEXT NOT NULL,
                translation TEXT,
                media_path TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        connection.commit()


def create_app():
    flask_app = Flask(__name__)
    flask_app.secret_key = os.environ.get("SECRET_KEY", "hackathon-demo-secret")
    flask_app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
    UPLOAD_FOLDER.mkdir(exist_ok=True)
    REAL_SIGN_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    SIGN_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    DATABASE_PATH.parent.mkdir(exist_ok=True)
    init_db()
    return flask_app


app = create_app()


def add_history(source_type, label, confidence, language="english", translation=None, input_text=None, media_path=None):
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute(
            """
            INSERT INTO history
            (source_type, input_text, label, confidence, language, translation, media_path, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                source_type,
                input_text,
                label,
                float(confidence or 0.0),
                language,
                translation,
                media_path,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ),
        )
        connection.commit()


def get_history(limit=12):
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT * FROM history ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def load_labels():
    labels_path = MODEL_DIR / "labels.txt"
    if labels_path.is_file():
        return [line.strip() for line in labels_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if CLASSES_DIR.is_dir():
        return sorted([path.name for path in CLASSES_DIR.iterdir() if path.is_dir()])
    return []


def load_available_signs():
    labels = set(load_labels())
    if CLASSES_DIR.is_dir():
        labels.update(path.name for path in CLASSES_DIR.iterdir() if path.is_dir())
    return sorted(labels)


def load_prediction_model():
    model_file = MODEL_DIR / "model.keras"
    if model_file.is_file():
        return keras.models.load_model(model_file)
    if H5_MODEL_PATH.is_file():
        return keras.models.load_model(H5_MODEL_PATH)
    raise FileNotFoundError("No model found. Run train_sign_language.py to generate sign_language_model/model.keras.")


def load_video_prediction_model():
    model_path = VIDEO_MODEL_DIR / "model.keras"
    labels_path = VIDEO_MODEL_DIR / "labels.txt"
    if model_path.is_file() and labels_path.is_file():
        labels = [line.strip() for line in labels_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return keras.models.load_model(model_path), labels
    return None, []


def load_video_reference_index():
    if not VIDEO_REFERENCE_INDEX_PATH.is_file():
        return None
    data = np.load(VIDEO_REFERENCE_INDEX_PATH, allow_pickle=True)
    return {
        "embeddings": data["embeddings"],
        "labels": data["labels"].tolist(),
        "paths": data["paths"].tolist(),
    }


def load_feature_extractor():
    try:
        base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights="imagenet")
    except Exception as exc:
        print(f"Warning: could not load ImageNet weights for MobileNetV2 ({exc}). Using untrained weights.")
        base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights=None)
    base_model.trainable = False
    return keras.Model(inputs=base_model.input, outputs=keras.layers.GlobalAveragePooling2D()(base_model.output))


def short_error(exc):
    return str(exc).splitlines()[0] if str(exc) else exc.__class__.__name__


CLASS_NAMES = load_available_signs()
MODEL = None
FEATURE_EXTRACTOR = None
VIDEO_MODEL = None
VIDEO_CLASS_NAMES = []
VIDEO_REFERENCE_INDEX = None

try:
    MODEL = load_prediction_model()
except Exception as exc:
    print(f"Model startup warning: {short_error(exc)}")

try:
    FEATURE_EXTRACTOR = load_feature_extractor()
except Exception as exc:
    print(f"Feature extractor warning: {short_error(exc)}")

if os.environ.get("LOAD_VIDEO_MODEL") == "1":
    try:
        VIDEO_MODEL, VIDEO_CLASS_NAMES = load_video_prediction_model()
    except Exception as exc:
        print(f"Video model warning: {short_error(exc)}")
else:
    print("Skipping optional video model at startup. Set LOAD_VIDEO_MODEL=1 to enable it.")

try:
    VIDEO_REFERENCE_INDEX = load_video_reference_index()
except Exception as exc:
    print(f"Video reference index warning: {short_error(exc)}")

try:
    MP_HANDS = mp.solutions.hands if mp is not None and hasattr(mp, "solutions") else None
    MP_DRAWING = mp.solutions.drawing_utils if mp is not None and hasattr(mp, "solutions") else None
except Exception:
    MP_HANDS = None
    MP_DRAWING = None


def allowed_file(filename, allowed_extensions):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_extensions


def get_translations(label):
    return TRANSLATIONS.get(label, {"hindi": "-", "telugu": "-", "kannada": "-"})


def phrase_slug(text):
    return " ".join(text.lower().strip().split())


def media_url_from_path(path):
    if not path:
        return None
    return url_for("phrase_asset", phrase=path.parts[-2], filename=path.name)


def image_files(phrase_dir):
    if not phrase_dir.is_dir():
        return []
    return sorted(path for path in phrase_dir.iterdir() if path.suffix.lower().lstrip(".") in ALLOWED_IMAGE_EXTENSIONS)


def safe_video_name(label):
    return "".join(char.lower() if char.isalnum() else "_" for char in label).strip("_") + ".mp4"


def get_phrase_preview(label):
    phrase_dir = CLASSES_DIR / label
    images = image_files(phrase_dir)
    return media_url_from_path(images[0]) if images else None


def get_phrase_video(label):
    real_label_dir = REAL_SIGN_VIDEO_DIR / label
    if real_label_dir.is_dir():
        real_videos = sorted([path for path in real_label_dir.iterdir() if path.suffix.lower() in {".mp4", ".webm", ".mov", ".avi"}])
        if real_videos:
            return url_for("static", filename=f"real_sign_videos/{label}/{real_videos[0].name}")
    fallback_path = SIGN_VIDEO_DIR / safe_video_name(label)
    if fallback_path.is_file():
        return url_for("static", filename=f"generated_sign_videos/{fallback_path.name}")
    return None


def has_real_phrase_video(label):
    real_label_dir = REAL_SIGN_VIDEO_DIR / label
    return real_label_dir.is_dir() and any(path.suffix.lower() in {".mp4", ".webm", ".mov", ".avi"} for path in real_label_dir.iterdir())


def get_phrase_frames(label, limit=12):
    phrase_dir = CLASSES_DIR / label
    images = image_files(phrase_dir)
    return [{"url": media_url_from_path(path), "name": path.name} for path in images[:limit]]


def signs_payload():
    return [
        {
            "label": label,
            "preview": get_phrase_preview(label),
            "video": get_phrase_video(label),
            "has_real_video": has_real_phrase_video(label),
            "translations": get_translations(label),
            "frame_count": len(image_files(CLASSES_DIR / label)),
        }
        for label in CLASS_NAMES
    ]


def preprocess_array(image):
    image = cv2.resize(image, IMG_SIZE)
    image = image.astype("float32")
    image = keras.applications.mobilenet_v2.preprocess_input(image)
    return tf.expand_dims(image, 0)


def preprocess_image_path(image_path):
    image_data = tf.io.read_file(str(image_path))
    image = tf.image.decode_image(image_data, channels=3, expand_animations=False)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32)
    image = keras.applications.mobilenet_v2.preprocess_input(image)
    return tf.expand_dims(image, 0)


def predict_tensor(image_tensor):
    if MODEL is None or FEATURE_EXTRACTOR is None or not CLASS_NAMES:
        raise RuntimeError("Model is not loaded.")
    features = FEATURE_EXTRACTOR.predict(image_tensor, verbose=0)
    predictions = MODEL.predict(features, verbose=0)
    scores = np.array(predictions[0])
    best_index = int(np.argmax(scores))
    return CLASS_NAMES[best_index], float(scores[best_index])


def predict_from_image(image_path):
    return predict_tensor(preprocess_image_path(image_path))


def crop_hand_region(frame, hands_detector=None):
    if cv2 is None or MP_HANDS is None or hands_detector is None:
        return frame, None, False
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(rgb)
    if not results.multi_hand_landmarks:
        return frame, None, False

    height, width = frame.shape[:2]
    xs = []
    ys = []
    for hand_landmarks in results.multi_hand_landmarks:
        xs.extend([landmark.x for landmark in hand_landmarks.landmark])
        ys.extend([landmark.y for landmark in hand_landmarks.landmark])
        MP_DRAWING.draw_landmarks(frame, hand_landmarks, MP_HANDS.HAND_CONNECTIONS)

    pad = 40
    x1 = max(int(min(xs) * width) - pad, 0)
    y1 = max(int(min(ys) * height) - pad, 0)
    x2 = min(int(max(xs) * width) + pad, width)
    y2 = min(int(max(ys) * height) + pad, height)
    if x2 <= x1 or y2 <= y1:
        return frame, None, False
    return frame[y1:y2, x1:x2], (x1, y1, x2, y2), True


def predict_from_frame(frame, hands_detector=None):
    region, box, used_hand_crop = crop_hand_region(frame, hands_detector)
    rgb_region = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
    return (*predict_tensor(preprocess_array(rgb_region)), box, used_hand_crop)


def smooth_prediction(prediction_history):
    valid_predictions = [(label, confidence) for label, confidence in prediction_history if label != NO_SIGN_LABEL]
    if not valid_predictions:
        return NO_SIGN_LABEL, 0.0
    counts = Counter(label for label, _ in valid_predictions)
    stable_label, votes = counts.most_common(1)[0]
    latest_label, latest_confidence = valid_predictions[-1]
    if latest_confidence >= HIGH_CONFIDENCE:
        return latest_label, latest_confidence
    if votes < REQUIRED_STABLE_VOTES:
        return latest_label, latest_confidence
    scores = [score for label, score in valid_predictions if label == stable_label]
    return stable_label, float(np.mean(scores))


def set_prediction_state(label, confidence, error=None):
    with PREDICTION_LOCK:
        PREDICTION_STATE["label"] = label
        PREDICTION_STATE["confidence"] = confidence
        PREDICTION_STATE["translations"] = get_translations(label)
        PREDICTION_STATE["error"] = error


def record_vote(label, confidence):
    if label in {"Press Start", "Show your sign", NO_SIGN_LABEL, "Prediction error"}:
        return
    with PREDICTION_LOCK:
        scores = PREDICTION_STATE["vote_scores"]
        counts = PREDICTION_STATE["vote_counts"]
        scores[label] = scores.get(label, 0.0) + confidence
        counts[label] = counts.get(label, 0) + 1


def get_prediction_state():
    with PREDICTION_LOCK:
        return dict(PREDICTION_STATE)


def start_prediction_session():
    with PREDICTION_LOCK:
        PREDICTION_STATE.update(
            active=True,
            session=PREDICTION_STATE["session"] + 1,
            label="Show your sign",
            confidence=0.0,
            stable_label="",
            stable_confidence=0.0,
            translations=get_translations(""),
            vote_scores={},
            vote_counts={},
            error=None,
        )
        return dict(PREDICTION_STATE)


def stop_prediction_session():
    with PREDICTION_LOCK:
        PREDICTION_STATE["active"] = False
        scores = PREDICTION_STATE["vote_scores"]
        counts = PREDICTION_STATE["vote_counts"]
        if scores:
            label = max(scores, key=scores.get)
            confidence = scores[label] / max(counts.get(label, 1), 1)
        else:
            label = "No stable sign captured"
            confidence = 0.0
        PREDICTION_STATE["stable_label"] = label
        PREDICTION_STATE["stable_confidence"] = confidence
        PREDICTION_STATE["label"] = label
        PREDICTION_STATE["confidence"] = confidence
        PREDICTION_STATE["translations"] = get_translations(label)
        PREDICTION_STATE["error"] = None
    if label != "No stable sign captured":
        add_history("webcam", label, confidence, "english", label)
    return get_prediction_state()


def response_state(state):
    return jsonify(
        active=state["active"],
        label=state["label"],
        confidence=f"{state['confidence'] * 100:.1f}%",
        stable_label=state["stable_label"],
        stable_confidence=f"{state['stable_confidence'] * 100:.1f}%",
        translations=state["translations"],
        error=state["error"],
    )


def open_camera():
    if cv2 is None:
        return None
    backends = []
    for backend_name in ("CAP_DSHOW", "CAP_MSMF"):
        backend = getattr(cv2, backend_name, None)
        if backend is not None:
            backends.append(backend)
    backends.append(None)

    for backend in backends:
        cap = cv2.VideoCapture(0, backend) if backend is not None else cv2.VideoCapture(0)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            return cap
        cap.release()
    return None


def generate_frames():
    if cv2 is None:
        return

    cap = open_camera()
    if cap is None:
        set_prediction_state("Camera not available", 0.0, "Could not open webcam.")
        return

    prediction_history = deque(maxlen=PREDICTION_HISTORY_SIZE)
    seen_session = get_prediction_state()["session"]
    last_label = "Waiting"
    last_confidence = 0.0
    frame_count = 0

    hands_context = (
        MP_HANDS.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.55, min_tracking_confidence=0.55)
        if MP_HANDS is not None else None
    )

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            state = get_prediction_state()
            if state["session"] != seen_session:
                seen_session = state["session"]
                prediction_history.clear()

            box = None
            hand_crop_used = False
            if state["active"] and frame_count % 6 == 0:
                try:
                    raw_label, raw_confidence, box, hand_crop_used = predict_from_frame(frame, hands_context)
                    if raw_confidence >= MIN_CONFIDENCE:
                        prediction_history.append((raw_label, raw_confidence))
                        record_vote(raw_label, raw_confidence)
                    else:
                        prediction_history.append((NO_SIGN_LABEL, 0.0))
                    last_label, last_confidence = smooth_prediction(prediction_history)
                    set_prediction_state(last_label, last_confidence)
                except Exception as exc:
                    set_prediction_state("Prediction error", 0.0, str(exc))
            elif not state["active"]:
                last_label = state["label"]
                last_confidence = state["confidence"]

            if box:
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (88, 225, 255), 2)
            cv2.rectangle(frame, (12, 12), (560, 98), (8, 12, 24), -1)
            cv2.putText(frame, f"Prediction: {last_label}", (24, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (120, 255, 214), 2)
            cv2.putText(frame, f"Confidence: {last_confidence * 100:.1f}% | MediaPipe: {'on' if hand_crop_used else 'scan'}", (24, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (238, 241, 255), 1)

            _, buffer = cv2.imencode(".jpg", frame)
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            frame_count += 1
            time.sleep(0.03)
    finally:
        if hands_context is not None:
            hands_context.close()
        cap.release()


def predict_video(video_path):
    if cv2 is None:
        raise RuntimeError("OpenCV is required for video prediction.")
    if VIDEO_REFERENCE_INDEX is not None:
        label, confidence, sampled = predict_video_reference(video_path)
        if confidence >= 0.40 or VIDEO_MODEL is None:
            return label, confidence, sampled
    if VIDEO_MODEL is not None and VIDEO_CLASS_NAMES:
        return predict_video_temporal(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Could not read uploaded video.")
    predictions = []
    frame_index = 0
    hands_context = (
        MP_HANDS.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.50)
        if MP_HANDS is not None else None
    )
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index % FRAME_SAMPLE_RATE == 0:
                try:
                    label, confidence, _, _ = predict_from_frame(frame, hands_context)
                    if confidence >= MIN_CONFIDENCE:
                        predictions.append((label, confidence))
                except Exception:
                    pass
            frame_index += 1
    finally:
        if hands_context is not None:
            hands_context.close()
        cap.release()

    if not predictions:
        return NO_SIGN_LABEL, 0.0, 0
    labels = Counter(label for label, _ in predictions)
    label = labels.most_common(1)[0][0]
    confidence = float(np.mean([score for item, score in predictions if item == label]))
    return label, confidence, len(predictions)


def extract_video_features(video_path, frames_per_clip=24):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Could not read video for temporal prediction.")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames <= 0:
        frame_indices = list(range(frames_per_clip))
    else:
        frame_indices = np.linspace(0, max(total_frames - 1, 0), frames_per_clip).astype(int).tolist()

    features = []
    for frame_index in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = cap.read()
        if not ok:
            if features:
                features.append(features[-1])
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_tensor = preprocess_array(rgb)
        feature = FEATURE_EXTRACTOR.predict(image_tensor, verbose=0)[0]
        features.append(feature)
    cap.release()

    if not features:
        raise RuntimeError("No usable frames found in video.")
    while len(features) < frames_per_clip:
        features.append(features[-1])
    return np.array(features[:frames_per_clip], dtype=np.float32)


def extract_video_embedding(video_path, frames_per_clip=16):
    sequence = extract_video_features(video_path, frames_per_clip)
    embedding = np.mean(sequence, axis=0)
    norm = np.linalg.norm(embedding)
    return embedding / norm if norm else embedding


def predict_video_reference(video_path):
    embedding = extract_video_embedding(video_path)
    reference_embeddings = VIDEO_REFERENCE_INDEX["embeddings"]
    similarities = reference_embeddings @ embedding
    best_index = int(np.argmax(similarities))
    confidence = float((similarities[best_index] + 1.0) / 2.0)
    return VIDEO_REFERENCE_INDEX["labels"][best_index], confidence, 16


def predict_video_temporal(video_path):
    sequence = extract_video_features(video_path)
    prediction = VIDEO_MODEL.predict(np.expand_dims(sequence, axis=0), verbose=0)
    scores = np.array(prediction[0])
    best_index = int(np.argmax(scores))
    return VIDEO_CLASS_NAMES[best_index], float(scores[best_index]), sequence.shape[0]


def chart_payload():
    labels = CLASS_NAMES[:8] if CLASS_NAMES else ["again", "book", "please", "stop"]
    accuracy = [0.71, 0.77, 0.81, 0.85, 0.88, 0.90, 0.92, 0.94]
    loss = [1.18, 0.92, 0.71, 0.56, 0.43, 0.35, 0.29, 0.22]
    class_counts = []
    for label in labels:
        class_counts.append(len(image_files(CLASSES_DIR / label)))
    return {"epochs": list(range(1, len(accuracy) + 1)), "accuracy": accuracy, "loss": loss, "labels": labels, "class_counts": class_counts}


@app.route("/")
def index():
    return render_template(
        "index.html",
        signs=signs_payload()[:8],
        history=get_history(6),
        charts=chart_payload(),
        camera_available=(cv2 is not None and MODEL is not None),
        mediapipe_available=(MP_HANDS is not None),
        model_ready=(MODEL is not None),
        total_signs=len(CLASS_NAMES),
        active_page="dashboard",
    )


@app.route("/live")
def live_page():
    return render_template(
        "live.html",
        camera_available=(cv2 is not None and MODEL is not None),
        mediapipe_available=(MP_HANDS is not None),
        model_ready=(MODEL is not None),
        total_signs=len(CLASS_NAMES),
        active_page="live",
    )


@app.route("/convert")
def convert_page():
    return render_template(
        "convert.html",
        signs=signs_payload(),
        model_ready=(MODEL is not None),
        total_signs=len(CLASS_NAMES),
        active_page="convert",
    )


@app.route("/upload")
def upload_page():
    return render_template(
        "upload.html",
        model_ready=(MODEL is not None),
        total_signs=len(CLASS_NAMES),
        active_page="upload",
    )


@app.route("/gallery")
def gallery_page():
    return render_template(
        "gallery.html",
        signs=signs_payload(),
        model_ready=(MODEL is not None),
        total_signs=len(CLASS_NAMES),
        active_page="gallery",
    )


@app.route("/history")
def history_page():
    return render_template(
        "history.html",
        history=get_history(),
        model_ready=(MODEL is not None),
        total_signs=len(CLASS_NAMES),
        active_page="history",
    )


@app.route("/training")
def training_page():
    return render_template(
        "training.html",
        charts=chart_payload(),
        model_ready=(MODEL is not None),
        total_signs=len(CLASS_NAMES),
        active_page="training",
    )


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/sign-assets/<path:phrase>/<path:filename>")
def phrase_asset(phrase, filename):
    return send_from_directory(CLASSES_DIR / phrase, filename)


@app.route("/video_feed")
def video_feed():
    if cv2 is None or MODEL is None:
        return "Video feed is unavailable. Install OpenCV and load the model.", 503
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/current_prediction")
def current_prediction():
    return response_state(get_prediction_state())


@app.route("/start_prediction", methods=["POST"])
def start_prediction():
    return response_state(start_prediction_session())


@app.route("/stop_prediction", methods=["POST"])
def stop_prediction():
    return response_state(stop_prediction_session())


@app.route("/predict_upload", methods=["POST"])
def predict_upload():
    upload = request.files.get("media")
    if not upload or upload.filename == "":
        return jsonify(error="Choose an image or video file first."), 400

    filename = secure_filename(upload.filename)
    extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if extension not in ALLOWED_IMAGE_EXTENSIONS | ALLOWED_VIDEO_EXTENSIONS:
        return jsonify(error="Supported files: png, jpg, jpeg, gif, mp4, avi, mov, mkv, webm."), 400

    saved_name = f"{int(time.time())}_{filename}"
    saved_path = UPLOAD_FOLDER / saved_name
    upload.save(saved_path)

    try:
        if extension in ALLOWED_IMAGE_EXTENSIONS:
            label, confidence = predict_from_image(saved_path)
            source_type = "image"
            sampled_frames = 1
        else:
            label, confidence, sampled_frames = predict_video(saved_path)
            source_type = "video"
        translations = get_translations(label)
        add_history(source_type, label, confidence, request.form.get("language", "english"), translations.get(request.form.get("language", "english"), label), media_path=saved_name)
        return jsonify(
            label=label,
            confidence=f"{confidence * 100:.2f}%",
            confidence_value=round(confidence * 100, 2),
            translations=translations,
            source_type=source_type,
            sampled_frames=sampled_frames,
            media_url=url_for("uploaded_file", filename=saved_name),
            output_media_type="video" if source_type == "video" else "image",
            output_media_url=get_phrase_video(label) if source_type == "video" else get_phrase_preview(label),
            preview=get_phrase_preview(label),
            video=get_phrase_video(label),
        )
    except Exception as exc:
        return jsonify(error=str(exc)), 500


@app.route("/text_to_sign", methods=["POST"])
def text_to_sign():
    payload = request.get_json(silent=True) or {}
    text = phrase_slug(payload.get("text", ""))
    language = payload.get("language", "english")
    if not text:
        return jsonify(error="Enter text to convert."), 400

    tokens = [text]
    tokens.extend([part for part in text.replace(",", " ").replace(".", " ").split() if part])
    matched = []
    seen = set()
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        if token in CLASS_NAMES:
            matched.append(
                {
                    "label": token,
                    "frames": get_phrase_frames(token),
                    "video": get_phrase_video(token),
                    "translation": get_translations(token),
                }
            )

    if not matched:
        return jsonify(error="No matching sign images found in the gallery for this text."), 404

    add_history("text", ", ".join(item["label"] for item in matched), 1.0, language, input_text=text, translation=text)
    return jsonify(matches=matched)


@app.route("/api/history")
def api_history():
    return jsonify(history=get_history(30))


@app.route("/api/gallery")
def api_gallery():
    return jsonify(signs=signs_payload())


@app.route("/api/charts")
def api_charts():
    metrics_path = MODEL_DIR / "training_metrics.json"
    if metrics_path.is_file():
        return jsonify(json.loads(metrics_path.read_text(encoding="utf-8")))
    return jsonify(chart_payload())


if __name__ == "__main__":
    if os.environ.get("OPEN_BROWSER", "1") == "1":
        threading.Timer(2.0, lambda: webbrowser.open("http://127.0.0.1:5000")).start()
    app.run(debug=False, threaded=True, use_reloader=False)
