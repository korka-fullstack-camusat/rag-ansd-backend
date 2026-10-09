FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Cache du modele d'embedding fastembed (monte en volume par docker-compose)
    FASTEMBED_CACHE_PATH=/models \
    # Index de recherche rapide (monte en volume par docker-compose)
    FASTINDEX_DIR=/app/fastindex

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY *.py ./

# data/ (manifest.csv) et storage/ (base Chroma) sont montes en volumes
RUN mkdir -p data storage /models /app/fastindex

EXPOSE 8000

# Plusieurs processus par conteneur (un par coeur CPU environ) : WEB_CONCURRENCY.
# FORWARDED_ALLOW_IPS : adresses autorisees a transmettre l'IP reelle du client
# (X-Forwarded-For) — « * » uniquement derriere un repartiteur de charge.
ENV WEB_CONCURRENCY=4 \
    FORWARDED_ALLOW_IPS=127.0.0.1
CMD ["sh", "-c", "exec uvicorn api:app --host 0.0.0.0 --port 8000 --workers ${WEB_CONCURRENCY} --proxy-headers --forwarded-allow-ips=${FORWARDED_ALLOW_IPS} --timeout-keep-alive 30"]
