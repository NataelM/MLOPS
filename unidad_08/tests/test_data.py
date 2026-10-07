"""tests/test_data.py — Tests del generador de datos."""
import sys
import pandas as pd

sys.path.insert(0, "src")
from generate_data import generate  # noqa: E402

FEATURES = ["Presion", "Tonelaje", "Velocidad", "%Solidos", "Potencia", "F80", "Brazo"]


def test_generate_returns_dataframe():
    """generate() debe retornar un DataFrame."""
    df = generate(n=200)
    assert isinstance(df, pd.DataFrame)


def test_generate_has_correct_columns():
    """El dataset debe tener todas las columnas esperadas."""
    df = generate(n=200)
    for col in FEATURES + ["picos_intens"]:
        assert col in df.columns, f"Columna faltante: {col}"


def test_generate_target_is_binary():
    """El target picos_intens debe ser binario (0 o 1)."""
    df = generate(n=300)
    assert set(df["picos_intens"].unique()).issubset({0, 1})


def test_generate_class_imbalance():
    """La tasa de clase 1 debe estar entre 5% y 20%."""
    df = generate(n=500)
    rate = df["picos_intens"].mean()
    assert 0.05 <= rate <= 0.20, f"Tasa de clase 1 fuera de rango: {rate:.2%}"


def test_generate_no_nulls():
    """El dataset no debe tener valores nulos."""
    df = generate(n=300)
    assert df.isnull().sum().sum() == 0


def test_generate_n_rows():
    """El dataset debe tener exactamente n filas."""
    df = generate(n=400)
    assert len(df) == 400


def test_generate_reproducible():
    """El mismo random_state debe producir el mismo dataset."""
    df1 = generate(n=200, random_state=99)
    df2 = generate(n=200, random_state=99)
    assert df1.reset_index(drop=True).equals(df2.reset_index(drop=True))