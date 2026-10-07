"""tests/test_model.py — Tests del modelo serializado."""
import os
import pickle
import sys

import numpy as np
import pytest

sys.path.insert(0, "src")
from generate_data import generate  # noqa: E402
from train_pipeline import train, FEATURES  # noqa: E402


@pytest.fixture(scope="module")
def modelo_entrenado(tmp_path_factory):
    """Fixture: entrena el modelo una vez para todos los tests del módulo."""
    tmp = tmp_path_factory.mktemp("workdir")
    cwd = os.getcwd()
    os.chdir(tmp)
    try:
        df = generate(n=400)
        train(df)
        with open(tmp / "artifacts" / "modelo.pkl", "rb") as f:
            modelo = pickle.load(f)
    finally:
        os.chdir(cwd)  # volver a la carpeta original para no afectar a otros tests
    return modelo


def test_model_has_predict(modelo_entrenado):
    """El modelo debe tener el método predict."""
    assert hasattr(modelo_entrenado, "predict")


def test_model_has_predict_proba(modelo_entrenado):
    """El modelo debe tener el método predict_proba."""
    assert hasattr(modelo_entrenado, "predict_proba")


def test_model_predicts_binary(modelo_entrenado):
    """Las predicciones deben ser binarias (0 o 1)."""
    X = generate(n=50)[FEATURES]
    y_pred = modelo_entrenado.predict(X)
    assert set(y_pred).issubset({0, 1})


def test_model_predict_proba_shape(modelo_entrenado):
    """predict_proba debe retornar shape (n, 2) con valores en [0, 1]."""
    X = generate(n=10)[FEATURES]
    proba = modelo_entrenado.predict_proba(X)
    assert proba.shape == (9, 2)
    assert (proba >= 0).all() and (proba <= 1).all()


def test_model_predict_proba_sums_to_one(modelo_entrenado):
    """Las probabilidades por fila deben sumar 1."""
    X = generate(n=20)[FEATURES]
    proba = modelo_entrenado.predict_proba(X)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_model_is_decision_tree(modelo_entrenado):
    """El modelo debe ser un DecisionTreeClassifier."""
    from sklearn.tree import DecisionTreeClassifier
    assert isinstance(modelo_entrenado, DecisionTreeClassifier)