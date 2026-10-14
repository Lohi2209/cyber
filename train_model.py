from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import json
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

from utils.model_utils import MAX_SEQUENCE_LENGTH, MODELS_DIR, save_artifacts
from utils.preprocess import clean_text


DATA_PATHS = [
    Path(__file__).resolve().parent / "train.csv.zip",
    Path(__file__).resolve().parent / "data" / "harassment_corpus.csv",
    Path(__file__).resolve().parent / "data" / "cyberbullying_severity.csv",
    Path(__file__).resolve().parent / "data" / "cyberbullying_tweets.csv",
    Path(__file__).resolve().parent / "hate-speech-and-offensive-language-master" / "data" / "labeled_data.csv",
    Path(r"C:\Users\manda\Downloads\cyberbullying_tweets.csv"),
]

TEXT_COLUMNS = ("tweet_text", "text", "tweet", "comment_text", "message", "content")
LABEL_COLUMNS = ("severity", "label", "category", "cyberbullying_type", "class")
SEVERITY_LABELS = {
    "high": "high",
    "high severity": "high",
    "medium": "medium",
    "medium severity": "medium",
    "low": "low",
    "low severity": "low",
    "non-cyberbullying": "not_cyberbullying",
    "non cyberbullying": "not_cyberbullying",
    "not_cyberbullying": "not_cyberbullying",
    "not cyberbullying": "not_cyberbullying",
    "not harassing": "not_cyberbullying",
    "age": "age",
    "gender": "gender",
    "ethnicity": "ethnicity",
    "religion": "religion",
    "other_cyberbullying": "other_cyberbullying",
    "other cyberbullying": "other_cyberbullying",
}

DAVIDSON_LABELS = {
    "0": "high",
    "1": "medium",
    "2": "not_cyberbullying",
}

JIGSAW_LABELS = {
    "safe": "safe",
    "medium": "medium",
    "high": "high",
}


def load_dataset():
    data_path = next((path for path in DATA_PATHS if path.exists()), None)
    if data_path is None:
        locations = ", ".join(str(path) for path in DATA_PATHS)
        raise FileNotFoundError(f"Dataset not found. Checked: {locations}")
    dataframe = pd.read_csv(data_path)
    text_column = next((column for column in TEXT_COLUMNS if column in dataframe.columns), None)
    label_column = next((column for column in LABEL_COLUMNS if column in dataframe.columns), None)
    jigsaw_columns = {"toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"}
    if text_column is not None and jigsaw_columns.issubset(dataframe.columns):
        dataframe = dataframe.rename(columns={text_column: "tweet_text"})
        high_risk = dataframe[["severe_toxic", "threat", "identity_hate"]].max(axis=1) > 0
        medium_risk = dataframe[["toxic", "obscene", "insult"]].max(axis=1) > 0
        dataframe["cyberbullying_type"] = np.select(
            [high_risk, medium_risk],
            ["high", "medium"],
            default="safe",
        )
        dataframe = dataframe[["tweet_text", "cyberbullying_type"]]
        print(f"Using Jigsaw dataset: {data_path}")
        print("Using Jigsaw severity-only labels: safe, medium, high")
        print("Class distribution:")
        print(dataframe["cyberbullying_type"].value_counts())
        return dataframe
    if text_column is None or label_column is None:
        raise ValueError(
            f"Dataset must contain one text column from {TEXT_COLUMNS} and one label column from {LABEL_COLUMNS}."
        )
    dataframe = dataframe.rename(columns={text_column: "tweet_text", label_column: "cyberbullying_type"})
    normalized_labels = dataframe["cyberbullying_type"].astype(str).str.strip().str.lower()
    dataframe["cyberbullying_type"] = normalized_labels.map(SEVERITY_LABELS).fillna(
        normalized_labels.map(DAVIDSON_LABELS).fillna(normalized_labels.map(JIGSAW_LABELS))
    )
    if dataframe["cyberbullying_type"].isna().any():
        unknown_labels = sorted(dataframe.loc[dataframe["cyberbullying_type"].isna(), label_column].astype(str).unique())
        raise ValueError(
            f"Unsupported labels: {unknown_labels}. Expected severity labels or legacy categories: "
            f"{sorted(set(SEVERITY_LABELS.values()))}"
        )
    print(f"Using dataset: {data_path}")
    print("Class distribution:")
    print(dataframe["cyberbullying_type"].value_counts())
    return dataframe


def train_model():
    dataframe = load_dataset().copy()
    dataframe["tweet_text"] = dataframe["tweet_text"].fillna("").astype(str)
    dataframe["clean_text"] = dataframe["tweet_text"].apply(clean_text)
    dataframe = dataframe[dataframe["clean_text"].str.len() > 0]

    features = dataframe["clean_text"].tolist()
    target = dataframe["cyberbullying_type"].astype(str).tolist()

    label_encoder = LabelEncoder()
    encoded_target = label_encoder.fit_transform(target)

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        encoded_target,
        test_size=0.2,
        random_state=42,
        stratify=encoded_target if len(label_encoder.classes_) > 1 else None,
    )

    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=20000, oov_token="<OOV>")
    tokenizer.fit_on_texts(x_train)
    x_train_sequences = tf.keras.preprocessing.sequence.pad_sequences(
        tokenizer.texts_to_sequences(x_train), maxlen=MAX_SEQUENCE_LENGTH, padding="post", truncating="post"
    )
    x_test_sequences = tf.keras.preprocessing.sequence.pad_sequences(
        tokenizer.texts_to_sequences(x_test), maxlen=MAX_SEQUENCE_LENGTH, padding="post", truncating="post"
    )

    vocabulary_size = min(20000, len(tokenizer.word_index) + 1)
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(42)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(MAX_SEQUENCE_LENGTH,)),
        tf.keras.layers.Embedding(vocabulary_size, 128, mask_zero=True),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True)),
        tf.keras.layers.GlobalMaxPooling1D(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(len(label_encoder.classes_), activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    class_counts = np.bincount(y_train)
    class_weights = {
        index: len(y_train) / (len(class_counts) * count)
        for index, count in enumerate(class_counts)
    }
    model.fit(
        x_train_sequences,
        np.asarray(y_train),
        validation_split=0.1,
        epochs=50,
        batch_size=128,
        class_weight=class_weights,
        callbacks=[
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=1, min_lr=1e-5
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy", patience=3, restore_best_weights=True
            ),
        ],
        verbose=1,
    )

    lstm_probabilities = model.predict(x_test_sequences, verbose=0)
    tfidf_vectorizer = TfidfVectorizer(max_features=30000, ngram_range=(1, 2), sublinear_tf=True)
    x_train_tfidf = tfidf_vectorizer.fit_transform(x_train)
    x_test_tfidf = tfidf_vectorizer.transform(x_test)
    tfidf_model = LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced")
    tfidf_model.fit(x_train_tfidf, y_train)
    tfidf_probabilities = tfidf_model.predict_proba(x_test_tfidf)

    production_vectorizer = FeatureUnion([
        ("word", TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=2,
            max_features=100000,
            sublinear_tf=True,
            strip_accents="unicode",
        )),
        ("character", TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            min_df=2,
            max_features=150000,
            sublinear_tf=True,
            strip_accents="unicode",
        )),
    ])
    x_train_production = production_vectorizer.fit_transform(x_train)
    x_test_production = production_vectorizer.transform(x_test)
    production_model = CalibratedClassifierCV(
        LinearSVC(C=1.0, random_state=42),
        cv=3,
        n_jobs=-1,
    )
    production_model.fit(x_train_production, y_train)
    production_probabilities = production_model.predict_proba(x_test_production)

    lstm_weight = 0.0
    probabilities = production_probabilities
    predictions = probabilities.argmax(axis=1)
    accuracy = accuracy_score(y_test, predictions)

    print(f"Word + character calibrated LinearSVC accuracy: {accuracy:.4f}")
    print("Macro F1 and per-class precision/recall:")
    print(classification_report(y_test, predictions, target_names=label_encoder.classes_, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    save_artifacts(
        model,
        tokenizer,
        label_encoder.classes_.tolist(),
        tfidf_vectorizer,
        tfidf_model,
        lstm_weight,
        production_vectorizer,
        production_model,
    )
    joblib.dump(production_vectorizer, MODELS_DIR / "production_vectorizer.joblib")
    joblib.dump(production_model, MODELS_DIR / "production_model.joblib")
    joblib.dump(label_encoder.classes_.tolist(), MODELS_DIR / "production_labels.joblib")
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_test, predictions, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_test, predictions, average="weighted", zero_division=0)),
        "labels": label_encoder.classes_.tolist(),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
    }
    with (MODELS_DIR / "performance_metrics.json").open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)


if __name__ == "__main__":
    train_model()
