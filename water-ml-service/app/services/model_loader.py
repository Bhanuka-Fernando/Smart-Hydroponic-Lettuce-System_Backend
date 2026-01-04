import joblib
from functools import lru_cache
from app.core.config import MODEL_PATH

@lru_cache(maxsize=1)
def load_artifact():
    return joblib.load(MODEL_PATH)
