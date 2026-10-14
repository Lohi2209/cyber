import csv
import io
import json
import logging
import zipfile
from pathlib import Path

from flask import Flask, flash, redirect, render_template, request, send_file, url_for

from utils.database import (
    fetch_average_confidence,
    fetch_category_counts,
    fetch_recent_results,
    fetch_severity_counts,
    fetch_total_results,
    initialize_database,
    save_detection_result,
    save_detection_results,
    update_review_status,
)
from utils.model_utils import load_artifacts
from utils.model_utils import MAX_SEQUENCE_LENGTH
from utils.preprocess import clean_text
from utils.severity import (
    apply_rule_based_override,
    build_suggestion,
    calculate_fuzzy_severity,
    map_category_to_severity,
)


app = Flask(__name__)
app.secret_key = "cyberbullying-secret-key"
METRICS_PATH = Path(__file__).resolve().parent / "models" / "performance_metrics.json"
LOGGER = logging.getLogger(__name__)
_ARTIFACTS = None

try:
    _ARTIFACTS = load_artifacts()
except FileNotFoundError:
    LOGGER.warning("Model artifacts are not available yet; predictions will fail until training is complete.")


def load_performance_metrics():
    if not METRICS_PATH.exists():
        return None
    with METRICS_PATH.open(encoding="utf-8") as metrics_file:
        return json.load(metrics_file)


def predict_messages(messages):
    global _ARTIFACTS
    if _ARTIFACTS is None:
        _ARTIFACTS = load_artifacts()
    model, tokenizer, labels, vectorizer, classifier, lstm_weight = _ARTIFACTS
    cleaned_messages = [clean_text(message) for message in messages]
    if model is None:
        probabilities = classifier.predict_proba(vectorizer.transform(cleaned_messages))
    else:
        from tensorflow.keras.preprocessing.sequence import pad_sequences

        sequences = pad_sequences(
            tokenizer.texts_to_sequences(cleaned_messages),
            maxlen=MAX_SEQUENCE_LENGTH,
            padding="post",
            truncating="post",
        )
        probabilities = model.predict(sequences, verbose=0)
        if vectorizer is not None and classifier is not None:
            companion_probabilities = classifier.predict_proba(vectorizer.transform(cleaned_messages))
            probabilities = lstm_weight * probabilities + (1 - lstm_weight) * companion_probabilities

    results = []
    for message, message_probabilities in zip(messages, probabilities):
        prediction = labels[int(message_probabilities.argmax())]
        confidence = float(message_probabilities.max())
        severity, _ = calculate_fuzzy_severity(message, prediction, confidence)
        if severity == "Safe":
            severity = map_category_to_severity(prediction)
        prediction, severity, overridden = apply_rule_based_override(message, prediction, severity)
        category_description = {
            "high": "hate-speech-related language",
            "medium": "offensive language",
            "not_cyberbullying": "no cyberbullying detected",
        }.get(prediction, f"{prediction} content")
        results.append({
            "message": message,
            "predicted_category": prediction,
            "severity": severity,
            "confidence": confidence,
            "suggestion": build_suggestion(severity),
            "explanation": f"Detected {category_description}; assigned {severity} severity at {confidence:.0%} confidence.",
            "overridden": overridden,
        })
    return results


@app.before_request
def setup_database():
    initialize_database()


@app.route("/")
def index():
    severity_counts = fetch_severity_counts()
    total_messages = fetch_total_results()
    average_confidence = fetch_average_confidence()
    overview_cards = [
        {"label": "Total analyses", "value": total_messages},
        {"label": "Safe", "value": severity_counts.get("Safe", 0)},
        {"label": "High risk", "value": severity_counts.get("High", 0)},
        {"label": "Average confidence", "value": f"{average_confidence * 100:.1f}%"},
    ]
    return render_template(
        "index.html",
        overview_cards=overview_cards,
        total_messages=total_messages,
        severity_counts=severity_counts,
        average_confidence=average_confidence,
    )


@app.route("/predict", methods=["POST"])
def predict():
    message = request.form.get("message", "").strip()
    if not message:
        flash("Please enter a message to analyze.", "error")
        return redirect(url_for("index"))

    try:
        result = predict_messages([message])[0]
    except FileNotFoundError:
        flash("Train the model first so the app can make predictions.", "error")
        return redirect(url_for("index"))

    save_detection_result(
        result["message"], result["predicted_category"], result["severity"],
        result["confidence"], result["suggestion"],
    )

    return render_template(
        "index.html",
        result={
            "message": message,
            "predicted_category": result["predicted_category"],
            "severity": result["severity"],
            "confidence": round(result["confidence"] * 100, 2),
            "suggestion": result["suggestion"],
            "overridden": result["overridden"],
        },
    )


@app.route("/bulk", methods=["GET", "POST"])
def bulk_analysis():
    results = None
    summary = None
    if request.method == "POST":
        uploaded_file = request.files.get("file")
        if uploaded_file is None or not uploaded_file.filename:
            flash("Choose a CSV file to analyze.", "error")
        elif not uploaded_file.filename.lower().endswith((".csv", ".zip", ".txt")):
            flash("Bulk analysis accepts TXT, CSV, or ZIP files containing a CSV.", "error")
        else:
            try:
                uploaded_bytes = uploaded_file.read()
                filename = uploaded_file.filename.lower()
                if filename.endswith(".txt"):
                    messages = [
                        line.strip()
                        for line in uploaded_bytes.decode("utf-8-sig").splitlines()
                        if line.strip()
                    ]
                    rows = [{"message": message} for message in messages]
                    skipped_rows = 0
                else:
                    if filename.endswith(".zip"):
                        with zipfile.ZipFile(io.BytesIO(uploaded_bytes)) as archive:
                            csv_names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
                            if not csv_names:
                                raise ValueError("ZIP file does not contain a CSV file.")
                            content = archive.read(csv_names[0]).decode("utf-8-sig")
                    else:
                        content = uploaded_bytes.decode("utf-8-sig")
                    rows = list(csv.DictReader(io.StringIO(content)))
                    message_column = next(
                        (column for column in ("message", "text", "tweet", "tweet_text", "comment_text", "content") if rows and column in rows[0]),
                        None,
                    )
                    if message_column is None and rows:
                        raise ValueError("CSV must contain a message, text, tweet, tweet_text, comment_text, or content column.")
                    messages = [row.get(message_column, "").strip() for row in rows]
                    skipped_rows = sum(not message for message in messages)
                    messages = [message for message in messages if message]
                if not messages:
                    raise ValueError("The selected message column does not contain any non-empty messages.")
                if len(messages) > 200000:
                    raise ValueError("Please upload at most 200,000 messages per batch.")
                results = []
                for start in range(0, len(messages), 5000):
                    batch_results = predict_messages(messages[start:start + 5000])
                    results.extend(batch_results)
                    save_detection_results(batch_results)
                summary = {
                    "total": len(results),
                    "input_rows": len(rows),
                    "skipped_rows": skipped_rows,
                    "safe": sum(result["severity"] == "Safe" for result in results),
                    "low": sum(result["severity"] == "Low" for result in results),
                    "medium": sum(result["severity"] == "Medium" for result in results),
                    "high": sum(result["severity"] == "High" for result in results),
                }
            except (UnicodeDecodeError, ValueError, zipfile.BadZipFile) as error:
                flash(str(error), "error")
            except FileNotFoundError:
                flash("Train the model first so the app can make predictions.", "error")
            except Exception:
                LOGGER.exception("Bulk analysis failed")
                flash("Bulk analysis could not be completed. Check the file format and try a smaller batch.", "error")
    return render_template("bulk.html", results=results, summary=summary)


@app.route("/dashboard")
def dashboard():
    severity = request.args.get("severity", "All")
    search = request.args.get("search", "").strip()
    review_status = request.args.get("review_status", "All")
    results = fetch_recent_results(
        limit=20,
        severity=severity,
        search=search or None,
        review_status=review_status,
    )
    severity_counts = fetch_severity_counts()
    category_counts = fetch_category_counts()
    total_messages = fetch_total_results()
    average_confidence = fetch_average_confidence()
    severity_chart_labels = ["Safe", "Low", "Medium", "High"]
    severity_chart_values = [severity_counts.get(label, 0) for label in severity_chart_labels]
    category_chart_labels = list(category_counts.keys())
    category_chart_values = list(category_counts.values())
    total_messages = sum(severity_counts.values())
    return render_template(
        "dashboard.html",
        results=results,
        severity_counts=severity_counts,
        category_counts=category_counts,
        total_messages=total_messages,
        average_confidence=average_confidence,
        severity_chart_labels=severity_chart_labels,
        severity_chart_values=severity_chart_values,
        category_chart_labels=category_chart_labels,
        category_chart_values=category_chart_values,
        current_severity=severity,
        current_search=search,
        current_review_status=review_status,
        performance_metrics=load_performance_metrics(),
    )


@app.route("/review/<int:result_id>", methods=["POST"])
def review_result(result_id):
    update_review_status(result_id, request.form.get("review_status", "Pending Review"))
    return redirect(request.referrer or url_for("dashboard"))


@app.route("/export-csv")
def export_csv():
    severity = request.args.get("severity", "All")
    search = request.args.get("search", "").strip()
    results = fetch_recent_results(limit=1000, severity=severity, search=search or None)

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["id", "message", "predicted_category", "severity", "confidence", "suggestion", "review_status", "created_at"])

    for row in results:
        writer.writerow([
            row["id"],
            row["message"],
            row["predicted_category"],
            row["severity"],
            f"{row['confidence']:.4f}",
            row["suggestion"],
            row["review_status"],
            row["created_at"],
        ])

    buffer.seek(0)
    filename = "cyberbullying_results.csv"
    return send_file(
        io.BytesIO(buffer.getvalue().encode("utf-8")),
        mimetype="text/csv",
        as_attachment=True,
        download_name=filename,
    )


@app.route("/report")
def report():
    severity_counts = fetch_severity_counts()
    category_counts = fetch_category_counts()
    total_messages = fetch_total_results()
    average_confidence = fetch_average_confidence()
    recent_results = fetch_recent_results(limit=5)

    highest_severity = max(severity_counts.items(), key=lambda item: item[1])[0] if severity_counts else "No data"
    top_category = max(category_counts.items(), key=lambda item: item[1])[0] if category_counts else "No data"

    return render_template(
        "report.html",
        total_messages=total_messages,
        average_confidence=average_confidence,
        severity_counts=severity_counts,
        category_counts=category_counts,
        recent_results=recent_results,
        highest_severity=highest_severity,
        top_category=top_category,
    )


if __name__ == "__main__":
    app.run(debug=False, use_reloader=False)
