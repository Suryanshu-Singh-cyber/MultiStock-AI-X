"""Stage 24c — Event bus: Redis Pub/Sub with in-memory fallback.

Topics: price_tick, news_event, forecast_update.
Falls back to in-process queue if Redis is unavailable (dev mode).
"""
from __future__ import annotations

import json
import queue
import threading
from typing import Callable

from src.core.config import get_live_config
from src.core.logger import get_logger

log = get_logger(__name__)

TOPICS = ("price_tick", "news_event", "forecast_update")


class EventBus:
    def __init__(self):
        self._redis = None
        self._local: dict[str, queue.Queue] = {t: queue.Queue() for t in TOPICS}
        self._subs: dict[str, list[Callable]] = {t: [] for t in TOPICS}
        self._connect()

    def _connect(self):
        try:
            import redis
            url = get_live_config().get("redis", {}).get(
                "url", "redis://localhost:6379/0")
            self._redis = redis.Redis.from_url(url, socket_timeout=2)
            self._redis.ping()
            log.info("event bus: Redis connected (%s)", url)
        except Exception as e:
            log.info("event bus: Redis unavailable (%s) — using in-memory", e)
            self._redis = None

    def publish(self, topic: str, payload: dict) -> None:
        if topic not in TOPICS:
            raise ValueError(f"unknown topic {topic}")
        data = json.dumps(payload, default=str)
        if self._redis is not None:
            try:
                self._redis.publish(topic, data)
                return
            except Exception as e:
                log.warning("redis publish failed: %s", e)
        self._local[topic].put(payload)
        for cb in self._subs.get(topic, []):
            try:
                cb(payload)
            except Exception as e:
                log.warning("subscriber error: %s", e)

    def subscribe(self, topic: str, callback: Callable) -> None:
        self._subs[topic].append(callback)
        if self._redis is not None:
            def _listen():
                ps = self._redis.pubsub()
                ps.subscribe(topic)
                for msg in ps.listen():
                    if msg["type"] == "message":
                        try:
                            callback(json.loads(msg["data"]))
                        except Exception as e:
                            log.warning("sub callback error: %s", e)
            threading.Thread(target=_listen, daemon=True).start()

    def cache_set(self, key: str, payload: dict, ttl: int = 86400) -> None:
        if self._redis is not None:
            try:
                self._redis.setex(key, ttl, json.dumps(payload, default=str))
                return
            except Exception:
                pass

    def cache_get(self, key: str) -> dict | None:
        if self._redis is not None:
            try:
                raw = self._redis.get(key)
                return json.loads(raw) if raw else None
            except Exception:
                return None
        return None


_bus: EventBus | None = None


def get_bus() -> EventBus:
    global _bus
    if _bus is None:
        _bus = EventBus()
    return _bus
