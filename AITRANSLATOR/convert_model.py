import os
import sys

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
H5_MODEL_PATH = os.path.join(SCRIPT_DIR, "sign_language_model.h5")

def convert_model():
    print(f"Loading model from: {MODEL_DIR}")

    # Load the SavedModel format
    model = keras.models.load_model(MODEL_DIR)

    print(f"Saving model to H5 format: {H5_MODEL_PATH}")
    model.save(H5_MODEL_PATH)

    print("Model conversion complete!")
    print(f"New H5 model saved at: {H5_MODEL_PATH}")

if __name__ == "__main__":
    convert_model()