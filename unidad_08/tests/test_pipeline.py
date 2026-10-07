"""tests/test_pipeline.py — Tests del pipeline de entrenamiento."""
import json
import sys

sys.path.insert(0, "src")
from generate_data import generate  # noqa: E402
from train_pipeline import train  # noqa: E402


def test_train_returns_metrics(tmp_path, monkeypatch):
    """train() debe retornar un dict con las métricas esperadas."""
    monkeypatch.chdir(tmp_path)
    df = generate(n=300)
    metricas = train(df)
    assert "f1" in metricas
    assert "recall" in metricas
    assert "accuracy" in metricas


def test_train_f1_positive(tmp_path, monkeypatch):
    """El F1-score debe ser mayor que 0."""
    monkeypatch.chdir(tmp_path)
    metricas = train(generate(n=300))
    assert metricas["f1"] > 0.0


def test_train_metrics_in_range(tmp_path, monkeypatch):
    """Las métricas deben estar en el rango [0, 1]."""
    monkeypatch.chdir(tmp_path)
    metricas = train(generate(n=300))
    for key in ("f1", "recall", "accuracy"):
        assert 0.0 <= metricas[key] <= 1.0, f"{key} fuera de rango: {metricas[key]}"


def test_train_saves_model(tmp_path, monkeypatch):
    """train() debe guardar artifacts/modelo.pkl."""
    monkeypatch.chdir(tmp_path)
    train(generate(n=300))
    assert (tmp_path / "artifacts" / "modelo.pkl").exists()


def test_train_saves_metrics(tmp_path, monkeypatch):
    """train() debe guardar artifacts/metrics.json con las claves correctas."""
    monkeypatch.chdir(tmp_path)
    train(generate(n=300))
    metrics_file = tmp_path / "artifacts" / "metrics.json"
    assert metrics_file.exists()
    with open(metrics_file) as f:
        m = json.load(f)
    assert "f1" in m and "recall" in m and "accuracy" in m


def test_train_saves_best_params(tmp_path, monkeypatch):
    """metrics.json debe incluir los mejores parámetros del GridSearch."""
    monkeypatch.chdir(tmp_path)
    train(generate(n=300))
    with open(tmp_path / "artifacts" / "metrics.json") as f:
        m = json.load(f)
    assert "params" in m
    assert isinstance(m["params"], dict)