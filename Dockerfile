FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Cache du modele d'embedding fastembed (monte en volume par docker-compose)
    FASTEMBED_CACHE_PATH=/models

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY *.py ./

# data/ (manifest.csv) et storage/ (base Chroma) sont montes en volumes
RUN mkdir -p data storage /models

EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
