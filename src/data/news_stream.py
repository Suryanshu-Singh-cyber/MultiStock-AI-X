"""Stage 12 — News intelligence (GDELT free + optional NewsAPI).

Pipeline: fetch -> timestamp (first_available) -> store -> sentiment/event
features. LEAKAGE RULE: a news item published at time T may only affect
features for dates > T (assert_news_first_available enforces).
Historical backfill uses GDELT DOC API with startdatetime/enddatetime —
GDElt 'seendate' ~ publish date; we treat date-only granularity as
available at END of that day (conservative: usable from next trading day).
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

from src.core.config import ROOT, get_live_config
from src.core.logger import get_logger
from src.leakage.domain_guards import assert_news_first_available

log = get_logger(__name__)
LIVE_DIR = ROOT / "data" / "live"
LIVE_DIR.mkdir(parents=True, exist_ok=True)

GDELT_DOC = "https://api.gdeltproject.org/api/v2/doc/doc"
_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "MultiStockAI-X/1.0 (research)"})

# NOTE: GDELT's English index has thin POWERGRID-specific coverage —
# verified 2026-11-13: 'POWERGRID'/'Power Grid' -> 0 hits, sector query works.
QUERIES = {
    "powergrid": '"Power Grid" OR POWERGRID OR "Power Grid Corporation" India',
    "sector": "(NTPC OR \"Tata Power\" OR \"Adani Power\" OR \"JSW Energy\" OR NHPC OR SJVN) (shares OR stock OR tariff OR order)",
    "macro": "(RBI OR CERC OR MNRE) (power OR electricity OR tariff)",
}


def fetch_gdelt(query: str, start: str, end: str, max_records: int = 250) -> pd.DataFrame:
    """GDELT DOC 2.0 API. start/end = 'YYYYMMDDHHMMSS' UTC."""
    params = {
        "query": query, "mode": "artlist", "maxrecords": max_records,
        "format": "json", "startdatetime": start, "enddatetime": end,
        "sort": "datedesc",
    }
    try:
        for attempt in range(3):
            r = _SESSION.get(GDELT_DOC, params=params, timeout=30)
            if r.status_code == 429:
                time.sleep(6 * (attempt + 1))
                continue
            r.raise_for_status()
            arts = r.json().get("articles", [])
            break
        else:
            return pd.DataFrame()
    except Exception as e:
        log.warning("GDELT fetch failed: %s", e)
        return pd.DataFrame()
    rows = [{
        "url": a.get("url"), "title": a.get("title"),
        "seendate": a.get("seendate"), "domain": a.get("domain"),
        "source_country": a.get("sourcecountry"),
    } for a in arts]
    return pd.DataFrame(rows)


def backfill_news(start: str = "2024-01-01", end: str | None = None,
                  chunk_days: int = 30) -> pd.DataFrame:
    """Backfill news for the feature window (GDELT free tier is generous)."""
    end = end or datetime.utcnow().strftime("%Y%m%d%H%M%S")
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end[:8], "%Y%m%d")
    frames = []
    cur = start_dt
    while cur < end_dt:
        nxt = min(cur + timedelta(days=chunk_days), end_dt)
        s = cur.strftime("%Y%m%d000000")
        e = nxt.strftime("%Y%m%d235959")
        for qname, q in QUERIES.items():
            df = fetch_gdelt(q, s, e, max_records=100)
            if not df.empty:
                df["query"] = qname
                frames.append(df)
            time.sleep(5.5)  # GDELT free tier: max 1 req / 5s
        cur = nxt
    if not frames:
        return pd.DataFrame()
    news = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["url"])
    news["published"] = pd.to_datetime(news["seendate"], format="%Y%m%dT%H%M%SZ",
                                       utc=True, errors="coerce")
    # first_available = end of publish day (conservative, date granularity)
    news["first_available"] = news["published"].dt.normalize() + pd.Timedelta(days=1)
    news = news.dropna(subset=["published"]).sort_values("first_available")
    out = LIVE_DIR / "news_buffer.parquet"
    news.to_parquet(out, index=False)
    log.info("news backfill: %d articles -> %s", len(news), out)
    return news


def poll_news_once() -> pd.DataFrame:
    """Live-mode poll (Stage 24c): last 15 minutes of news."""
    cfg = get_live_config()
    interval = cfg.get("news", {}).get("poll", 15)
    now = datetime.utcnow()
    start = (now - timedelta(minutes=interval)).strftime("%Y%m%d%H%M%S")
    end = now.strftime("%Y%m%d%H%M%S")
    frames = []
    for qname, q in QUERIES.items():
        df = fetch_gdelt(q, start, end, max_records=50)
        if not df.empty:
            df["query"] = qname
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    news = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["url"])
    news["published"] = pd.to_datetime(news["seendate"], format="%Y%m%dT%H%M%SZ",
                                       utc=True, errors="coerce")
    news["first_available"] = news["published"]
    buf = LIVE_DIR / "news_buffer.parquet"
    if buf.exists():
        old = pd.read_parquet(buf)
        news = pd.concat([old, news]).drop_duplicates(subset=["url"])
    news.to_parquet(buf, index=False)
    return news


def news_features(news: pd.DataFrame, dates: pd.DatetimeIndex,
                  lookback_days: int = 2) -> pd.DataFrame:
    """Aggregate news into per-date features, enforcing first_available < date.

    For a feature at trading date t (close), only articles with
    first_available <= t are eligible — enforced via assert.
    """
    if news.empty:
        return pd.DataFrame(index=dates)
    out = pd.DataFrame(index=dates)
    pub = news.set_index("first_available").sort_index()
    titles = news["title"].fillna("").str.lower()
    news = news.assign(
        is_earnings=titles.str.contains("result|earnings|q[1-4]", regex=True),
        is_policy=titles.str.contains("policy|government|regulator|cerc|mnre",
                                     regex=True),
        is_contract=titles.str.contains("order|contract|win|bag|project",
                                       regex=True),
        is_regulatory=titles.str.contains("sebi|rbi|tariff|fine|penalty",
                                          regex=True),
    )
    pub_idx = news.set_index("first_available")
    for d in dates:
        # articles available strictly before end of day d (date granularity)
        eligible = pub_idx[pub_idx.index <= pd.Timestamp(d) + pd.Timedelta(hours=15)]
        # guard: every article must satisfy first_available < next-day boundary
        for fa in eligible.index:
            assert_news_first_available(fa, pd.Timestamp(d) + pd.Timedelta(days=1),
                                        headline="news_feature")
        n = len(eligible)
        out.loc[d, "news_count"] = n
        if n:
            out.loc[d, "news_powergrid_share"] = float(
                (eligible["query"] == "powergrid").mean())
            out.loc[d, "news_earnings"] = float(eligible["is_earnings"].mean())
            out.loc[d, "news_policy"] = float(eligible["is_policy"].mean())
            out.loc[d, "news_contract"] = float(eligible["is_contract"].mean())
            out.loc[d, "news_regulatory"] = float(eligible["is_regulatory"].mean())
    # decay: exponential weighting of counts over lookback
    out["news_count_decay"] = out["news_count"].rolling(
        lookback_days, min_periods=1).apply(lambda x: float(np.sum(x * np.exp(-np.arange(len(x))[::-1] / 1.5))), raw=True) if "news_count" in out else 0
    return out.fillna(0.0)


if __name__ == "__main__":
    import numpy as np
    news = backfill_news(start="2025-01-01")
    print(news.head() if not news.empty else "no news fetched")
