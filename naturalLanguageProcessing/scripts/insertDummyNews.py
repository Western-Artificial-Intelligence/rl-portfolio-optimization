import sqlite3
import random
import datetime

DB_PATH = "../db/news.db"   
TICKERS = ["AAPL", "TSLA", "SQQQ", "NVDA", "GOOG"]
CATEGORIES = ["news", "earnings", "macro", "other"]
SOURCES = ["Bloomberg", "Reuters", "Yahoo Finance", "MarketWatch"]

def random_timestamp():
    now = datetime.datetime.utcnow()
    delta = datetime.timedelta(days=random.randint(0, 30),
                               hours=random.randint(0, 23),
                               minutes=random.randint(0, 59))
    ts = now - delta
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")

def main():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    for _ in range(50): 
        ticker = random.choice(TICKERS)
        timestamp = random_timestamp()
        category = random.choice(CATEGORIES)
        source = random.choice(SOURCES)

        headline = f"Dummy headline for {ticker}, category {category}"
        body = f"This is placeholder article text for {ticker} published at {timestamp}."

        raw_json = {
            "debug": True,
            "ticker": ticker,
            "timestamp": timestamp,
            "body_length": len(body)
        }

        cur.execute(
            """
            INSERT INTO news_articles
            (timestamp_utc, ticker, headline, body, source, category, raw_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (timestamp, ticker, headline, body, source, category, str(raw_json))
        )

    conn.commit()
    conn.close()
    print("Inserted 50 dummy news articles.")

if __name__ == "__main__":
    main()

