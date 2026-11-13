"""MultiStock-AI X — entry point.

Usage:
    python main.py --stage 0     # verify environment
    python main.py --stage 1     # download historical data
"""
import argparse

from src.core.config import get_config, get_all_symbols
from src.core.logger import get_logger

log = get_logger(__name__)


def stage0() -> None:
    cfg = get_config()
    symbols = get_all_symbols()
    log.info("project=%s target=%s", cfg["project"]["name"], cfg["project"]["target_symbol"])
    log.info("tracking %d symbols", len(symbols))
    for s in symbols:
        log.info("  %s", s)
    log.info("Stage 0 OK — config loaded, %d symbols registered", len(symbols))


def main() -> None:
    p = argparse.ArgumentParser(description="MultiStock-AI X")
    p.add_argument("--stage", type=int, default=0, help="build stage to run")
    args = p.parse_args()
    if args.stage == 0:
        stage0()
    else:
        log.info("Stage %d not implemented yet", args.stage)


if __name__ == "__main__":
    main()
