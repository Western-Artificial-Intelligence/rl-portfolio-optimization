import sqlite3
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "db", "news.db")


def get_news_by_id(record_id: int):
    """
    Fetch a single news article by its ID.
    Returns a dictionary or None if not found.
    """

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  
    cur = conn.cursor()

    cur.execute(
        """
        SELECT id, timestamp_utc, ticker, headline, body, source, category
        FROM news_articles
        WHERE id = ?
        """,
        (record_id,)
    )

    row = cur.fetchone()
    conn.close()

    if row is None:
        return None

    return {key: row[key] for key in row.keys()}


def main():
    record = get_news_by_id(1)

    if record:
        print("Record for ID 1:")
        print(record)
    else:
        print("No record found with ID = 1.")


if __name__ == "__main__":
    main()

