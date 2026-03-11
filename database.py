import sqlite3
from datetime import datetime

DB = "fakenews.db"


def init_db():
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS checks (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp    TEXT,
            article_text TEXT,
            bert_label   TEXT,
            trust_score  REAL,
            fact_verdict TEXT,
            supported    INTEGER,
            total_claims INTEGER
        )
    """)
    conn.commit()
    conn.close()
    print("✅ Database ready!")


def save_check(article, bert_label, trust_score, fact_verdict, supported, total):
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("""
        INSERT INTO checks
        (timestamp, article_text, bert_label, trust_score, fact_verdict, supported, total_claims)
        VALUES (?,?,?,?,?,?,?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        article[:500],
        bert_label,
        trust_score,
        fact_verdict,
        supported,
        total
    ))
    conn.commit()
    conn.close()


def get_recent(limit=10):
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("""
        SELECT id, timestamp, bert_label, trust_score, fact_verdict
        FROM checks ORDER BY id DESC LIMIT ?
    """, (limit,))
    rows = c.fetchall()
    conn.close()
    return rows


def get_stats():
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM checks")
    total = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM checks WHERE bert_label='FAKE'")
    fake = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM checks WHERE bert_label='REAL'")
    real = c.fetchone()[0]
    c.execute("SELECT AVG(trust_score) FROM checks")
    avg_trust = c.fetchone()[0] or 0
    conn.close()
    return {
        "total": total, "fake": fake,
        "real": real,
        "avg_trust": round(avg_trust, 1)
    }


if __name__ == "__main__":
    init_db()
    print("Stats:", get_stats())
