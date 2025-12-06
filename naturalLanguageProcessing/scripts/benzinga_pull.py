import os
import json
import argparse
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

import requests

###########################################################################
# Command for bezinga_pull.py
#
#does not read .env properly run this with API_KEY
#
#export BENZINGA_API=API_KEY
#python naturalLanguageProcessing/scripts/benzinga_pull.py --ticker AAPL --pages 1 --page-size 50
#
#--ticker STOCK_SYMBOL queries only selected stock
#--pages x how many pages for each article.
#--page-size x maximum number of pages. 25 is max for free
###########################################################################
BASE_URL = "https://api.benzinga.com/api/v2/news"


def get_api_key() -> str:
    key = os.getenv("BENZINGA_API")
    if not key:
        raise RuntimeError(
            "Missing BENZINGA_API environment variable. "
            "Export it first, e.g. export BENZINGA_API='...'"
        )
    return key


def fetch_news(
    ticker: str,
    page: int = 0,
    page_size: int = 15,
    display_output: str = "full",
    date: Optional[str] = None,
    updated_since: Optional[int] = None,
    channels: Optional[str] = None,
    timeout: int = 15,
):
    api_key = get_api_key()

    headers = {"accept": "application/json"}

    params = {
        "token": api_key,
        "tickers": ticker,
        "page": page,
        "pageSize": page_size,
        "displayOutput": display_output,  # full | abstract | headline
    }

    # Optional filters that help control result set size/perf
    if date:
        params["date"] = date  # YYYY-MM-DD
    if updated_since:
        params["updatedSince"] = updated_since  # unix ts
    if channels:
        params["channels"] = channels

    resp = requests.get(BASE_URL, params=params, headers=headers, timeout=timeout)

    # Attempt JSON decode even on errors so we can see API message
    try:
        payload = resp.json()
    except Exception:
        payload = {"raw_text": resp.text}

    return resp.status_code, payload, params


def save_payload(payload, out_dir: Path, label: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = out_dir / f"{label}_{ts}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    return path


def test_page_size_limits(ticker: str):
    sizes_to_test = [10, 50, 100, 101, 250, 1000]

    results = []
    for size in sizes_to_test:
        status, payload, params = fetch_news(
            ticker=ticker,
            page=0,
            page_size=size,
            display_output="full",
        )
        count = len(payload) if isinstance(payload, list) else None
        results.append(
            {
                "pageSize": size,
                "status": status,
                "items_returned": count,
                "error_preview": payload[:1] if isinstance(payload, list) else payload,
            }
        )

    return results


def fetch_recent_pages(ticker: str, pages: int, page_size: int):
    all_items = []
    for p in range(pages):
        status, payload, _ = fetch_news(
            ticker=ticker,
            page=p,
            page_size=page_size,
            display_output="full",
        )

        if status != 200:
            return status, payload, all_items

        if isinstance(payload, list):
            all_items.extend(payload)
        else:
            # Unexpected shape
            all_items.append(payload)

    return 200, all_items, all_items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--pages", type=int, default=1)
    parser.add_argument("--page-size", type=int, default=50)
    parser.add_argument("--out", default="naturalLanguageProcessing/data/benzinga")
    parser.add_argument("--test-limit", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out)

    if args.test_limit:
        results = test_page_size_limits(args.ticker)
        path = save_payload(results, out_dir, f"{args.ticker}_pagesize_limit_test")
        print(f"Saved limit test results -> {path}")
        print(json.dumps(results, indent=2))
        return

    # Normal fetch
    status, payload, _ = fetch_recent_pages(args.ticker, args.pages, args.page_size)

    label = f"{args.ticker}_recent_p{args.pages}_ps{args.page_size}"
    path = save_payload(payload, out_dir, label)

    if status != 200:
        print(f"Non-200 response ({status}). Saved error payload -> {path}")
        print(payload)
        return

    print(f"Saved {len(payload)} items -> {path}")

    for i, item in enumerate(payload[:10]):
        title = item.get("title")
        created = item.get("created")
        url = item.get("url")
        print(f"{i+1:02d}. {created} | {title}")
        if url:
            print(f"    {url}")


if __name__ == "__main__":
    main()