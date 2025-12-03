CREATE TABLE IF NOT EXISTS news_articles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp_utc TEXT NOT NULL,     
    ticker TEXT NOT NULL,
    headline TEXT NOT NULL,
    body TEXT,
    source TEXT NOT NULL,
    category TEXT NOT NULL, 
    raw_json TEXT       
);

CREATE INDEX IF NOT EXISTS idx_news_ticker_time
    ON news_articles (ticker, timestamp_utc);

CREATE INDEX IF NOT EXISTS idx_news_timestamp
    ON news_articles (timestamp_utc);

