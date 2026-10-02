"""train_pipeline.py — Entrena el modelo y guarda artefactos."""
import json
import pickle
import logging
from pathlib import Path

import numpy as np  # noqa: F401
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, recall_score, accuracy_score
from imblearn.over_sampling import SMOTE

from generate_data import generate, DATA_PATH

ARTIFACTS = Path("artifacts")
MODEL_PATH = ARTIFACTS / "modelo.pkl"
METRICS_PATH = ARTIFACTS / "metrics.json"

FEATURES = ["Presion", "Tonelaje", "Velocidad", "%Solidos", "Potencia", "F80", "Brazo"]
TARGET = "picos_intens"
RANDOM_STATE = 123
TEST_SIZE = 0.30
RECALL_MIN = 0.70  # quality gate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | TRAIN | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def load_data() -> pd.DataFrame:
    """Carga o genera el dataset."""
    if not DATA_PATH.exists():
        log.info("Generando dataset sintético...")
        generate()
    return pd.read_csv(DATA_PATH)


def train(df: pd.DataFrame) -> dict:
    """Entrena con GridSearchCV + SMOTE y retorna métricas."""
    X, y = df[FEATURES], df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    smote = SMOTE(k_neighbors=5, random_state=RANDOM_STATE)
    X_tr_s, y_tr_s = smote.fit_resample(X_train, y_train)

    param_grid = {
        "max_depth": [5, 8, None],
        "min_samples_split": [2, 5, 10],
        "criterion": ["gini", "entropy"],
    }

    gs = GridSearchCV(
        DecisionTreeClassifier(random_state=RANDOM_STATE),
        param_grid,
        cv=StratifiedKFold(5),
        scoring="f1",
        n_jobs=-1,
    )
    gs.fit(X_tr_s, y_tr_s)

    y_pred = gs.best_estimator_.predict(X_test)
    metricas = {
        "f1": round(f1_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "params": gs.best_params_,
        "recall_minimo": RECALL_MIN,
    }

    log.info(
        "F1=%.4f | Recall=%.4f | Acc=%.4f",
        metricas["f1"], metricas["recall"], metricas["accuracy"],
    )

    ARTIFACTS.mkdir(exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(gs.best_estimator_, f)
    with open(METRICS_PATH, "w") as f:
        json.dump(metricas, f, indent=2)

    log.info("Artefactos guardados en %s", ARTIFACTS)
    return metricas


if __name__ == "__main__":
    df = load_data()
    log.info("Dataset: %d filas", len(df))
    metricas = train(df)
    print(json.dumps(metricas, indent=2))