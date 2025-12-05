#!/usr/bin/env python3
"""
Fetch the most recent news articles for AAPL using yfinance,
scrape the full article text where possible, and save to:
/naturalLanguageProcessing/data

One note this also pulls from wsj, for these we can only parse summary and title, for yfinance we can pull full text
"""

import json
from pathlib import Path

import requests
from bs4 import BeautifulSoup
import yfinance as yf


def fetch_article_text(url: str, timeout: int = 10):
    if not url:
        return None

    try:
        resp = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": "Mozilla/5.0 (compatible; HamzaBot/1.0)"}
        )
        resp.raise_for_status()
    except Exception as e:
        print(f"[WARN] Failed to fetch article URL {url}: {e}")
        return None

    soup = BeautifulSoup(resp.text, "html.parser")

    # Heuristic 1: grab <p> tags inside <article>
    article_tag = soup.find("article")
    if article_tag:
        paragraphs = article_tag.find_all("p")
    else:
        # Heuristic 2: fallback to all <p> tags on page
        paragraphs = soup.find_all("p")

    texts = [p.get_text(strip=True) for p in paragraphs]
    texts = [t for t in texts if t]

    if not texts:
        return None

    return "\n\n".join(texts)


def main():
    ticker_symbol = "AAPL"
    max_articles = 1000  # number of STORY articles to keep

    nl_root = Path(__file__).resolve().parent.parent

    # naturalLanguageProcessing/data/
    output_dir = nl_root / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "aapl_news_latest.json"

    ticker = yf.Ticker(ticker_symbol)
    raw_news_items = ticker.news or []

    cleaned = []
    for raw in raw_news_items:
        content = raw.get("content", {})
        content_type = content.get("contentType")

        # We only want actual written stories, not videos, etc.
        if content_type != "STORY":
            continue

        provider = content.get("provider") or {}
        # prefer clickThroughUrl, fall back to canonicalUrl
        link_info = content.get("clickThroughUrl") or content.get("canonicalUrl") or {}
        link = link_info.get("url")

        title = content.get("title")
        publisher = provider.get("displayName")
        pub_date = content.get("pubDate")
        summary = content.get("summary")

        print(f"[INFO] STORY: {title!r} ({publisher})")
        print(f"       URL: {link}")

        article_text = fetch_article_text(link)
        if article_text:
            print("       -> extracted article_text")
        else:
            print("       -> NO article_text (likely video/short/premium/paywalled)")

        cleaned.append(
            {
                "ticker": ticker_symbol,
                "title": title,
                "publisher": publisher,
                "link": link,
                "pubDate": pub_date,
                "contentType": content_type,
                "summary": summary,
                "article_text": article_text,  # full body where possible
                "raw": raw,                    # keep full original payload
            }
        )

        if len(cleaned) >= max_articles:
            break

    # Write to JSON
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=2, ensure_ascii=False)

    num_with_text = sum(1 for x in cleaned if x.get("article_text"))
    print(
        f"\nSaved {len(cleaned)} AAPL STORY items "
        f"({num_with_text} with non-empty article_text) "
        f"to {output_file.resolve()}"
    )


if __name__ == "__main__":
    main()
