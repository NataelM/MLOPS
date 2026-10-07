# Dockerfile — Imagen del servicio API (FastAPI)
# Usado por docker-compose.preprod.yml (servicio: api)
# y por el Job 3 de GitHub Actions (build-docker)

FROM python:3.11-slim

WORKDIR /app

# Dependencias Python — copiar antes del código para aprovechar el cache
# IMPORTANTE: estas 4 primeras instrucciones son idénticas a las de
# Dockerfile.trainer. Así Docker construye la capa de pip install UNA sola vez
# y la comparte entre las dos imágenes (menos CPU, menos tiempo y menos disco).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

LABEL maintainer="MLOps Picos de Intensidad"
LABEL description="API de clasificación de picos de intensidad"
LABEL version="1.0.0"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ENV=preprod

# Código de la API y del pipeline
COPY api/ ./api/
COPY src/ ./src/

# Directorio de artefactos (el volumen Docker lo llenará en runtime)
RUN mkdir -p artifacts

EXPOSE 8000

# Health check interno
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]