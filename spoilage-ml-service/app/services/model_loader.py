import json
import joblib
import tensorflow as tf

def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_keras_model(path: str):
    return tf.keras.models.load_model(path)

def load_joblib(path: str):
    return joblib.load(path)
