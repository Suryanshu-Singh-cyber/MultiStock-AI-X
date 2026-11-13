"""Render startup: best-effort data refresh, then serve the API.

Data parquets are gitignored, so on a fresh deploy we re-download from
yfinance once (~1-2 min). If the download fails we still serve whatever
data exists — the API degrades gracefully rather than crash-looping.
"""
import subprocess
import sys

import uvicorn


def main() -> None:
    print("[start] refreshing market data (best effort)...", flush=True)
    try:
        subprocess.run(
            [sys.executable, "-m", "src.data.historical"],
            timeout=600, check=False,
        )
    except Exception as e:  # noqa: BLE001
        print(f"[start] data refresh failed ({e}) — serving with existing data", flush=True)

    print("[start] launching uvicorn on $PORT", flush=True)

    # optional in-process daily scheduler (15:45 IST) — set RUN_SCHEDULER=1
    if __import__("os").environ.get("RUN_SCHEDULER") == "1":
        import threading
        from src.streaming.scheduler import start_scheduler
        threading.Thread(target=start_scheduler, daemon=True).start()
        print("[start] daily scheduler enabled (15:45 IST)", flush=True)

    uvicorn.run("src.api.main:app", host="0.0.0.0", port=int(__import__("os").environ.get("PORT", 8000)))


if __name__ == "__main__":
    main()
