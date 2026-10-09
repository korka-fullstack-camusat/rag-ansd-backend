# Modeles locaux sur CPU : voix wolof (secours de Soynade) et traduction francais <-> pulaar — voir app.py.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Modeles Hugging Face (~3 Go), telecharges au premier demarrage, gardes dans un volume
    HF_HOME=/models

WORKDIR /app

# PyTorch CPU uniquement : bien plus leger que la version avec CUDA.
RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py speaker.npy ./

EXPOSE 8001
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]
