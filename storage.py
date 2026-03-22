import json
import sqlite3
from pathlib import Path
from typing import Any


class PredictionStorage:
    def __init__(self, db_path: str = "data/predictions.db") -> None:
        db_file = Path(db_path)
        db_file.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = str(db_file)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    prediction TEXT NOT NULL,
                    is_attack INTEGER NOT NULL,
                    score REAL,
                    probabilities_json TEXT,
                    record_json TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def save(self, record: dict[str, Any], result: dict[str, Any]) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO predictions (prediction, is_attack, score, probabilities_json, record_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    str(result.get("prediction")),
                    1 if result.get("is_attack") else 0,
                    float(result.get("attack_score", 0.0)),
                    json.dumps(result.get("probabilities", {})),
                    json.dumps(record),
                ),
            )
            conn.commit()

    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT id, created_at, prediction, is_attack, score, probabilities_json, record_json
                FROM predictions
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def stats(self) -> dict[str, Any]:
        """Aggregate counts for dashboard metrics."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT
                    COUNT(*) AS total,
                    SUM(CASE WHEN is_attack = 1 THEN 1 ELSE 0 END) AS attacks,
                    SUM(CASE WHEN is_attack = 0 THEN 1 ELSE 0 END) AS benign
                FROM predictions
                """
            ).fetchone()
        total = int(row[0] or 0)
        attacks = int(row[1] or 0)
        benign = int(row[2] or 0)
        rate = (attacks / total * 100.0) if total else 0.0
        return {
            "total": total,
            "attacks": attacks,
            "benign": benign,
            "detection_rate_percent": round(rate, 1),
        }
