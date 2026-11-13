FROM python:3.13-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY requirements.txt requirements-ml.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-ml.txt

COPY . .

# Default: run the daily inference loop (Stage 24+).
# For dev: docker run -p 8000:8000 -p 8501:8501 <img> uvicorn src.api.main:app
CMD ["python", "main.py", "--stage", "0"]
