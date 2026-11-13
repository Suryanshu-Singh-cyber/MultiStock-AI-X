@echo off
REM ============================================================
REM  MultiStock-AI X - DAILY UPDATE
REM  1) Downloads fresh prices (today's close if market ran)
REM  2) Scores yesterday's forecast vs reality
REM  3) Trains/reuses models + produces tomorrow's forecast
REM  4) Saves it -> dashboard auto-refreshes (30s)
REM ============================================================
cd /d "%~dp0"

echo.
echo  [1/3] Fetching latest market data...
.venv\Scripts\python -m src.data.historical

echo.
echo  [2/3] Running daily inference + scoring...
.venv\Scripts\python -c "from src.streaming.scheduler import daily_job; daily_job()"

echo.
echo  [3/3] Done! Forecast saved.
echo  Open http://localhost:3000 to see it.
echo.
pause
