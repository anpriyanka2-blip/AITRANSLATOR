import tensorflow as tf
from tensorflow import keras
import os

MODEL_PATH = "sign_language_model"

model_file = os.path.join(MODEL_PATH, "model.keras")
labels_file = os.path.join(MODEL_PATH, "labels.txt")

model = keras.models.load_model(model_file)

with open(labels_file, "r", encoding="utf-8") as f:
    labels = [line.strip() for line in f.readlines() if line.strip()]

print("Model loaded successfully!")
print("Classes:", labels)