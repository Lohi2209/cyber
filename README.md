# Cyberbullying Detection and Severity Assessment

This project is a complete cyberbullying severity detection system that combines:

- NLP preprocessing
- Tokenization and padded text sequences
- Word and character TF-IDF with calibrated LinearSVC classification
- Severity mapping
- Suggestion generation
- SQLite result storage
- Flask web UI

## Project Structure

- `cyberbullying_training.ipynb`: step-by-step model development notebook
- `train_model.py`: reusable training script
- `app.py`: Flask application
- `utils/`: preprocessing, severity, suggestions, and database helpers
- `templates/`: HTML pages
- `static/`: CSS and JavaScript assets
- `train.csv.zip`: Jigsaw training data used by the current notebook and trainer
- `models/`: saved model artifacts

## Run the Saved Application

Double-click `run_app.bat`. It starts Flask with the configured Anaconda Python interpreter and opens:

`http://127.0.0.1:5000/`

The application loads the saved files in `models/`; it does not retrain when it starts.

Available pages:

- `/`: single-message analysis
- `/bulk`: TXT, CSV, or ZIP bulk analysis
- `/dashboard`: charts, metrics, review status, and export
- `/report`: summary report

## Retrain the Model

1. Install dependencies with `C:\ProgramData\anaconda3\python.exe -m pip install -r requirements.txt`.
2. Open `cyberbullying_training.ipynb` and run the cells in order.
3. The final production cell saves the calibrated three-class model and dashboard metrics.
4. Alternatively run `C:\ProgramData\anaconda3\python.exe train_model.py`.

The current model predicts three severity classes: `safe`, `medium`, and `high`. It uses only message text as an input feature; dataset IDs and annotation metadata are excluded. The verified held-out accuracy is approximately 94.65%, and the dashboard also reports weighted metrics and a confusion matrix.
