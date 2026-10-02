# Dockerfile — API de Picos de Intensidad
# Usado por GitHub Actions (Job 3: build-docker) y por make docker

FROM python:3.11-slim

LABEL maintainer="MLOps Picos Intensidad"
LABEL description="Imagen CI/CD del modelo de clasificación de picos de intensidad"
LABEL version="1.0.0"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copiar e instalar dependencias Python
# IMPORTANTE: copiar requirements.txt antes del código para aprovechar el cache de Docker
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código fuente
COPY src/ ./src/

# Copiar artefactos del modelo (generados por train_pipeline.py)
# En GitHub Actions estos llegan via download-artifact
COPY artifacts/ ./artifacts/

# Verificación rápida al construir
RUN python -c "import pickle; \
    m = pickle.load(open('artifacts/modelo.pkl', 'rb')); \
    print('Modelo cargado OK:', type(m).__name__)" || echo "artifacts/ no disponible en build"

EXPOSE 8000

# Por defecto, verificar que el modelo carga correctamente
CMD ["python", "-c", \
     "import pickle; m=pickle.load(open('artifacts/modelo.pkl','rb')); \
      print('Modelo listo:', type(m).__name__)"]