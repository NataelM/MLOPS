"""tests/smoke/test_smoke.py — Smoke tests para verificar el entorno completo.

Se ejecutan DESPUÉS de docker compose up para verificar que todos
los servicios estén respondiendo correctamente.

Uso en Codespace (con el entorno levantado):
    pytest tests/smoke/ -v
    # o: make smoke

Variables de entorno configurables:
    API_URL    (default: http://localhost:8000)
    MLFLOW_URL (default: http://localhost:5000)
"""
import json
import os
import urllib.request

import pytest

API_URL    = os.getenv("API_URL",    "http://localhost:8000")
MLFLOW_URL = os.getenv("MLFLOW_URL", "http://localhost:5000")

PAYLOAD_NORMAL = {
    "Presion": 180.0, "Tonelaje": 350.0, "Velocidad": 1800.0,
    "%Solidos": 72.0, "Potencia": 800.0, "F80": 120.0, "Brazo": 55.0,
}
PAYLOAD_PICO = {
    "Presion": 225.0, "Tonelaje": 430.0, "Velocidad": 2250.0,
    "%Solidos": 79.0, "Potencia": 960.0, "F80": 148.0, "Brazo": 63.0,
}


def get_json(url: str) -> dict:
    """Hace GET a una URL y retorna el JSON parseado."""
    with urllib.request.urlopen(url, timeout=10) as resp:
        return json.loads(resp.read())


def post_json(url: str, payload: dict) -> dict:
    """Hace POST con JSON y retorna la respuesta."""
    data = json.dumps(payload).encode()
    req  = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


# ── Tests de la API ───────────────────────────────────────────────────────────

def test_api_root_responde():
    """El endpoint raíz de la API debe responder con info básica."""
    data = get_json(f"{API_URL}/")
    assert data.get("api") is not None
    assert data.get("version") == "1.0.0"


def test_api_health_ok():
    """El health check de la API debe retornar status ok."""
    data = get_json(f"{API_URL}/health")
    assert data["status"] == "ok"
    assert "modelo" in data
    assert 0 <= data.get("recall", 0) <= 1


def test_api_health_tiene_env():
    """El health check debe incluir el entorno (preprod)."""
    data = get_json(f"{API_URL}/health")
    assert "env" in data


def test_api_prediccion_payload_normal():
    """El endpoint /predecir debe retornar una predicción válida."""
    data = post_json(f"{API_URL}/predecir", PAYLOAD_NORMAL)
    assert "score_riesgo" in data
    assert data["decision"] in ["PICO DETECTADO", "NORMAL"]
    assert 0.0 <= data["score_riesgo"] <= 1.0
    assert "modelo" in data


def test_api_prediccion_payload_pico():
    """Un payload de alta intensidad debe retornar una predicción válida."""
    data = post_json(f"{API_URL}/predecir", PAYLOAD_PICO)
    assert "score_riesgo" in data
    assert data["decision"] in ["PICO DETECTADO", "NORMAL"]
    assert 0.0 <= data["score_riesgo"] <= 1.0


def test_api_prediccion_es_determinista():
    """El mismo payload debe producir siempre el mismo score."""
    r1 = post_json(f"{API_URL}/predecir", PAYLOAD_NORMAL)
    r2 = post_json(f"{API_URL}/predecir", PAYLOAD_NORMAL)
    assert r1["score_riesgo"] == r2["score_riesgo"]


def test_api_docs_disponible():
    """La documentación Swagger debe estar disponible en /docs."""
    with urllib.request.urlopen(f"{API_URL}/docs", timeout=10) as resp:
        assert resp.status == 200


# ── Tests de MLflow ───────────────────────────────────────────────────────────

def test_mlflow_ui_responde():
    """La UI de MLflow debe responder en el puerto configurado."""
    try:
        with urllib.request.urlopen(f"{MLFLOW_URL}/health", timeout=10) as resp:
            assert resp.status == 200
    except Exception:
        pytest.skip("MLflow no disponible — verifica que el stack está levantado")


def test_mlflow_experimentos_accesibles():
    """La API de experimentos de MLflow debe responder."""
    try:
        # MLflow 2.x/3.x: experiments/list ya no existe, se usa experiments/search
        data = get_json(f"{MLFLOW_URL}/api/2.0/mlflow/experiments/search?max_results=10")
        assert "experiments" in data
    except Exception:
        pytest.skip("MLflow API no disponible en este entorno")