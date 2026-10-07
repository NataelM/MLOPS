"""api/app.py — API REST del clasificador de picos de intensidad.

Carga el modelo desde artifacts/modelo.pkl al arrancar.
El entrenamiento previo es responsabilidad del servicio trainer.

Ejecutar directamente (sin Docker):
    uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload

Con Docker Compose (pre-prod):
    docker compose -f docker-compose.preprod.yml up --build
"""
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import ClienteInput, PrediccionOutput, HealthResponse
from api.predictor import predictor

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s | API | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga el modelo al arrancar la API (startup event)."""
    log.info("Iniciando API en entorno: %s", os.getenv("ENV", "dev"))
    predictor.cargar()
    log.info("Modelo listo: %s", type(predictor.modelo).__name__)
    yield
    log.info("API cerrando")


app = FastAPI(
    title="API Picos de Intensidad — Pre-producción",
    description=(
        "Clasificador de picos de intensidad para el sistema de molienda. "
        "Modelo entrenado con el pipeline MLOps del proyecto."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    debug=os.getenv("DEBUG", "False").lower() == "true",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Info"])
def root():
    """Información básica de la API."""
    return {
        "api":     "Picos de Intensidad",
        "env":     os.getenv("ENV", "dev"),
        "version": "1.0.0",
        "docs":    "/docs",
        "health":  "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["Salud"])
def health():
    """Estado de la API y metadatos del modelo cargado."""
    if predictor.modelo is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")
    return HealthResponse(
        status="ok",
        modelo=type(predictor.modelo).__name__,
        version="1.0.0",
        recall=float(predictor.metricas.get("recall", 0)),
        env=os.getenv("ENV", "dev"),
    )


@app.post("/predecir", response_model=PrediccionOutput, tags=["Prediccion"])
def predecir(cliente: ClienteInput):
    """
    Clasifica si los datos de operación corresponden a un pico de intensidad.

    - **score_riesgo**: probabilidad de pico [0.0 - 1.0]
    - **decision**: PICO DETECTADO | NORMAL
    """
    try:
        datos = cliente.model_dump(by_alias=True)
        return PrediccionOutput(**predictor.predecir(datos))
    except Exception as e:
        log.error("Error en predicción: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))