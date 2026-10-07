"""api/predictor.py — Carga el modelo.pkl y ejecuta predicciones."""
import json
import logging
import pickle
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

MODEL_PATH   = Path("artifacts/modelo.pkl")
METRICS_PATH = Path("artifacts/metrics.json")

FEATURES = ["Presion", "Tonelaje", "Velocidad", "%Solidos", "Potencia", "F80", "Brazo"]
UMBRAL   = 0.50  # score >= 0.50 → PICO DETECTADO


class Predictor:
    """Singleton que carga el modelo una vez y sirve predicciones."""

    def __init__(self):
        self.modelo   = None
        self.metricas = {}

    def cargar(self) -> None:
        """Carga modelo y métricas desde disco. Llamar UNA vez en startup."""
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Modelo no encontrado: {MODEL_PATH}. "
                "Asegúrate de que el servicio trainer completó exitosamente."
            )
        with open(MODEL_PATH, "rb") as f:
            self.modelo = pickle.load(f)

        if METRICS_PATH.exists():
            with open(METRICS_PATH) as f:
                self.metricas = json.load(f)

        log.info("Modelo cargado: %s", type(self.modelo).__name__)
        log.info("Recall en entrenamiento: %.4f", self.metricas.get("recall", 0))

    def predecir(self, datos: dict) -> dict:
        """Recibe dict con los campos del cliente y retorna la predicción."""
        if self.modelo is None:
            raise RuntimeError("Modelo no cargado. Llama a cargar() primero.")

        # Construir DataFrame con las features en el orden correcto
        X = pd.DataFrame([{f: datos.get(f, datos.get(f.replace("%", "Solidos"), 0.0))
                           for f in FEATURES}])

        # Manejar alias %Solidos → Solidos
        if "%Solidos" not in datos and "Solidos" in datos:
            X["%Solidos"] = datos["Solidos"]

        proba = float(self.modelo.predict_proba(X)[0][1])

        return {
            "score_riesgo":      round(proba, 4),
            "decision":          "PICO DETECTADO" if proba >= UMBRAL else "NORMAL",
            "probabilidad_pico": round(proba, 4),
            "umbral_usado":      UMBRAL,
            "modelo":            type(self.modelo).__name__,
        }


# Instancia global — singleton
predictor = Predictor()