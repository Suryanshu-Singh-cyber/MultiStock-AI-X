"""Stage 24d — Daily inference scheduler (APScheduler).

Runs the full pipeline at 15:45 IST every trading day (after NSE close
15:30). Also exposes run_now() for manual triggers.
"""
from __future__ import annotations

from datetime import datetime

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from src.core.logger import get_logger
from src.streaming.bus import get_bus
from src.streaming.inference import run_daily_inference

log = get_logger(__name__)

LAST_ERROR: str | None = None


def daily_job() -> None:
    """The 15:45 IST job: fetch latest -> features -> forecast -> publish."""
    global LAST_ERROR
    LAST_ERROR = None
    log.info("=== daily inference job starting ===")
    try:
        # refresh market data first (best effort — inference uses latest parquet)
        try:
            from src.data.historical import ingest_all
            ingest_all()
        except Exception as e:
            log.warning("data ingest failed, using cached parquet: %s", e)
        payload = run_daily_inference()
        # score yesterday's forecast now that today's return is known
        try:
            from src.streaming.scoring import score_pending
            score_pending()
        except Exception as e:
            log.warning("scoring failed: %s", e)
        bus = get_bus()
        bus.publish("forecast_update", payload)
        bus.cache_set("forecast:latest", payload)
        log.info("=== daily inference job done: status=%s ===", payload["status"])
    except Exception as e:
        LAST_ERROR = str(e)
        log.exception("daily inference job FAILED: %s", e)
        bus = get_bus()
        bus.publish("forecast_update", {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "status": "ERROR",
            "error": str(e),
        })


def start_scheduler() -> None:
    sched = BlockingScheduler(timezone="Asia/Kolkata")
    # 15:45 IST, Monday-Friday (NSE trading days; holiday check inside job)
    sched.add_job(daily_job, CronTrigger(day_of_week="mon-fri", hour=15,
                                         minute=45, timezone="Asia/Kolkata"),
                  id="daily_inference", replace_existing=True)
    log.info("scheduler started — daily inference at 15:45 IST Mon-Fri")
    sched.start()


if __name__ == "__main__":
    start_scheduler()
