# Natural Language Processing – News Database Module

This module provides a simple and extensible system for storing, validating, and querying financial news articles for downstream NLP tasks such as embeddings, summarization, trading signals, or sentiment analysis.

The system uses **SQLite** as a lightweight embedded database and includes helper scripts for data ingestion and retrieval.

## 🧱 Database Schema

The schema is defined in:  
`db/schema_news.sql`

### **Indexes**

```sql
CREATE INDEX idx_news_ticker_time
    ON news_articles (ticker, timestamp_utc);

CREATE INDEX idx_news_timestamp
    ON news_articles (timestamp_utc);

These indexes ensure fast ticker and date-range queries.

## 🔍 Fetching a News Article by ID

The script:
```
scripts/getNewsById.py
```

Provides a function:

```python
get_news_by_id(record_id: int) -> dict
```

### Example usage (inside the script):

```python
record = get_news_by_id(1)
print(record)
```

### Run it:

```bash
python scripts/getNewsById.py
```

### Expected output:

```
Record for ID 1:
{'id': 1,
 'timestamp_utc': '2025-11-15T11:20:19Z',
 'ticker': 'SQQQ',
 'headline': 'Dummy headline for SQQQ, category other',
 'body': 'This is placeholder article text...',
 'source': 'Bloomberg',
 'category': 'other'}
```

---

## 🧭 How to Query the Database Manually

Start SQLite:

```bash
sqlite3 db/news.db
```

### Useful commands inside SQLite:

```sql
.tables;
SELECT COUNT(*) FROM news_articles;
SELECT * FROM news_articles LIMIT 5;
SELECT * FROM news_articles WHERE ticker='AAPL' ORDER BY timestamp_utc DESC;
```

Exit with:

```sql
.exit
```

