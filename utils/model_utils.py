from pathlib import Path

import joblib
from tensorflow.keras.models import load_model


MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_PATH = MODELS_DIR / "lstm_model.keras"
TOKENIZER_PATH = MODELS_DIR / "lstm_tokenizer.joblib"
LABELS_PATH = MODELS_DIR / "lstm_labels.joblib"
TFIDF_MODEL_PATH = MODELS_DIR / "tfidf_classifier.joblib"
TFIDF_VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"
HYBRID_WEIGHT_PATH = MODELS_DIR / "hybrid_lstm_weight.joblib"
CHAR_MODEL_PATH = MODELS_DIR / "char_classifier.joblib"
CHAR_VECTORIZER_PATH = MODELS_DIR / "char_vectorizer.joblib"
PRODUCTION_MODEL_PATH = MODELS_DIR / "production_model.joblib"
PRODUCTION_VECTORIZER_PATH = MODELS_DIR / "production_vectorizer.joblib"
PRODUCTION_LABELS_PATH = MODELS_DIR / "production_labels.joblib"
MAX_SEQUENCE_LENGTH = 100


def save_artifacts(
    model,
    tokenizer,
    labels,
    vectorizer=None,
    classifier=None,
    lstm_weight=0.9,
    char_vectorizer=None,
    char_classifier=None,
):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    joblib.dump(tokenizer, TOKENIZER_PATH)
    joblib.dump(labels, LABELS_PATH)
    if vectorizer is not None and classifier is not None:
        joblib.dump(vectorizer, TFIDF_VECTORIZER_PATH)
        joblib.dump(classifier, TFIDF_MODEL_PATH)
        joblib.dump(lstm_weight, HYBRID_WEIGHT_PATH)
    if char_vectorizer is not None and char_classifier is not None:
        joblib.dump(char_vectorizer, CHAR_VECTORIZER_PATH)
        joblib.dump(char_classifier, CHAR_MODEL_PATH)


def load_artifacts():
    if all(path.exists() for path in (
        PRODUCTION_MODEL_PATH,
        PRODUCTION_VECTORIZER_PATH,
        PRODUCTION_LABELS_PATH,
    )):
        return (
            None,
            None,
            joblib.load(PRODUCTION_LABELS_PATH),
            joblib.load(PRODUCTION_VECTORIZER_PATH),
            joblib.load(PRODUCTION_MODEL_PATH),
            0.0,
        )

    required_paths = [MODEL_PATH, TOKENIZER_PATH, LABELS_PATH]
    if not all(path.exists() for path in required_paths):
        raise FileNotFoundError("LSTM model artifacts are missing. Run train_model.py first.")
    model = load_model(MODEL_PATH)
    tokenizer = joblib.load(TOKENIZER_PATH)
    labels = joblib.load(LABELS_PATH)
    if CHAR_MODEL_PATH.exists() and CHAR_VECTORIZER_PATH.exists() and HYBRID_WEIGHT_PATH.exists():
        vectorizer = joblib.load(CHAR_VECTORIZER_PATH)
        classifier = joblib.load(CHAR_MODEL_PATH)
        lstm_weight = joblib.load(HYBRID_WEIGHT_PATH)
        return model, tokenizer, labels, vectorizer, classifier, lstm_weight
    if TFIDF_MODEL_PATH.exists() and TFIDF_VECTORIZER_PATH.exists() and HYBRID_WEIGHT_PATH.exists():
        vectorizer = joblib.load(TFIDF_VECTORIZER_PATH)
        classifier = joblib.load(TFIDF_MODEL_PATH)
        lstm_weight = joblib.load(HYBRID_WEIGHT_PATH)
        return model, tokenizer, labels, vectorizer, classifier, lstm_weight
    return model, tokenizer, labels, None, None, 1.0
