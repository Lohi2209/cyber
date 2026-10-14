import sqlite3
from datetime import datetime
from pathlib import Path


DATABASE_PATH = Path(__file__).resolve().parents[1] / "data" / "cyberbullying_results.db"


def get_connection():
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database():
    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS detection_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                message TEXT NOT NULL,
                predicted_category TEXT NOT NULL,
                severity TEXT NOT NULL,
                confidence REAL NOT NULL,
                suggestion TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(detection_results)").fetchall()
        }
        if "review_status" not in columns:
            connection.execute(
                "ALTER TABLE detection_results ADD COLUMN review_status TEXT NOT NULL DEFAULT 'Pending Review'"
            )
        connection.commit()


def save_detection_result(message: str, predicted_category: str, severity: str, confidence: float, suggestion: str):
    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO detection_results (
                message,
                predicted_category,
                severity,
                confidence,
                suggestion,
                created_at,
                review_status
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (message, predicted_category, severity, confidence, suggestion, datetime.utcnow().isoformat(), "Pending Review"),
        )
        connection.commit()


def save_detection_results(results):
    with get_connection() as connection:
        connection.executemany(
            """
            INSERT INTO detection_results (
                message, predicted_category, severity, confidence, suggestion, created_at, review_status
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    result["message"],
                    result["predicted_category"],
                    result["severity"],
                    result["confidence"],
                    result["suggestion"],
                    datetime.utcnow().isoformat(),
                    "Pending Review",
                )
                for result in results
            ],
        )
        connection.commit()


def update_review_status(result_id: int, review_status: str):
    if review_status not in {"Pending Review", "Reviewed", "Resolved"}:
        raise ValueError("Unsupported review status")
    with get_connection() as connection:
        connection.execute(
            "UPDATE detection_results SET review_status = ? WHERE id = ?",
            (review_status, result_id),
        )
        connection.commit()


def fetch_recent_results(
    limit: int = 50,
    severity: str | None = None,
    search: str | None = None,
    review_status: str | None = None,
):
    with get_connection() as connection:
        clauses = []
        parameters = []

        if severity and severity != "All":
            clauses.append("severity = ?")
            parameters.append(severity)

        if search:
            clauses.append("(message LIKE ? OR predicted_category LIKE ? OR suggestion LIKE ?)")
            search_term = f"%{search}%"
            parameters.extend([search_term, search_term, search_term])

        if review_status and review_status != "All":
            clauses.append("review_status = ?")
            parameters.append(review_status)

        where_clause = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        cursor = connection.execute(
            """
            SELECT id, message, predicted_category, severity, confidence, suggestion, created_at, review_status
            FROM detection_results
            {where_clause}
            ORDER BY id DESC
            LIMIT ?
            """.format(where_clause=where_clause),
            (*parameters, limit),
        )
        return cursor.fetchall()


def fetch_severity_counts():
    with get_connection() as connection:
        cursor = connection.execute(
            """
            SELECT severity, COUNT(*) AS count
            FROM detection_results
            GROUP BY severity
            """
        )
        return {row["severity"]: row["count"] for row in cursor.fetchall()}


def fetch_category_counts():
    with get_connection() as connection:
        cursor = connection.execute(
            """
            SELECT predicted_category, COUNT(*) AS count
            FROM detection_results
            GROUP BY predicted_category
            ORDER BY count DESC, predicted_category ASC
            """
        )
        return {row["predicted_category"]: row["count"] for row in cursor.fetchall()}


def fetch_total_results():
    with get_connection() as connection:
        cursor = connection.execute("SELECT COUNT(*) AS total FROM detection_results")
        row = cursor.fetchone()
        return row["total"] if row else 0


def fetch_average_confidence():
    with get_connection() as connection:
        cursor = connection.execute("SELECT AVG(confidence) AS average_confidence FROM detection_results")
        row = cursor.fetchone()
        return row["average_confidence"] if row and row["average_confidence"] is not None else 0.0
