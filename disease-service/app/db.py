import sqlite3
from pathlib import Path

DB_PATH = Path("data/disease.db")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_conn() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS scan_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plant_id TEXT NOT NULL,
            captured_at TEXT NOT NULL,
            health_score INTEGER NOT NULL,
            status TEXT NOT NULL,
            main_issue TEXT NOT NULL,
            probs_json TEXT NOT NULL,
            tipburn_json TEXT NOT NULL,
            image_name TEXT
        );
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_logs_plant_time ON scan_logs(plant_id, captured_at);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_logs_time ON scan_logs(captured_at);")
        conn.commit()
