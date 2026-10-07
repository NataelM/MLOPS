"""api/schemas.py — Modelos Pydantic para validación de entrada y salida."""
from pydantic import BaseModel, Field
from typing import Literal


class ClienteInput(BaseModel):
    """Datos de entrada del clasificador de picos de intensidad."""
    Presion:   float = Field(..., description="Presión del sistema (kPa)")
    Tonelaje:  float = Field(..., description="Tonelaje procesado (t/h)")
    Velocidad: float = Field(..., description="Velocidad del molino (rpm)")
    Solidos:   float = Field(..., alias="%Solidos", description="Porcentaje de sólidos")
    Potencia:  float = Field(..., description="Potencia consumida (kW)")
    F80:       float = Field(..., description="F80 del mineral (µm)")
    Brazo:     float = Field(..., description="Posición del brazo (m)")

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {"example": {
            "Presion": 210.0, "Tonelaje": 400.0, "Velocidad": 2100.0,
            "%Solidos": 75.0, "Potencia": 900.0, "F80": 135.0, "Brazo": 59.0,
        }},
    }


class PrediccionOutput(BaseModel):
    """Respuesta del clasificador de picos de intensidad."""
    score_riesgo:         float
    decision:             Literal["PICO DETECTADO", "NORMAL"]
    probabilidad_pico:    float
    umbral_usado:         float
    modelo:               str


class HealthResponse(BaseModel):
    """Respuesta del endpoint de salud."""
    status:  str
    modelo:  str
    version: str
    recall:  float
    env:     str