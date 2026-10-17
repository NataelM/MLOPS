"""
Sección 8 — GridSearchCV + MLflow (Run 1).

Búsqueda exhaustiva de hiperparámetros del RandomForest sobre los datos
balanceados con UnderSampling (igual que el notebook).
4 x 4 x 3 = 48 combinaciones x 5 folds = 240 entrenamientos.
Registra el mejor en MLflow como Run 1.

Ejecutar: python pipeline/02_gridsearch_mlflow.py
Prerrequisito: src/base_data.pkl (generado por 01_base_pipeline.py)
"""
import pickle
import mlflow
import mlflow.sklearn
from pathlib import Path

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score,
    precision_score, roc_auc_score, classification_report,
)

import os
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

RANDOM_STATE = 42

# ── Cargar datos de la sección base ──────────────────────────────────────────
BASE_PKL = Path("src/base_data.pkl")
if not BASE_PKL.exists():
    raise FileNotFoundError(
        "No se encontró src/base_data.pkl. "
        "Ejecuta primero: python pipeline/01_base_pipeline.py"
    )

with open(BASE_PKL, "rb") as f:
    X_train_u, y_train_u, X_test, y_test = pickle.load(f)

print(f"Datos cargados — Train: {X_train_u.shape} | Test: {X_test.shape}")

# ── 8.1 Configurar MLflow ─────────────────────────────────────────────────────
# Usar ruta absoluta correcta — evita el bug de PermissionError en Codespaces
MLFLOW_DIR   = Path("mlruns").resolve()           # ruta absoluta real
TRACKING_URI = MLFLOW_DIR.as_uri()                # file:///workspaces/.../mlruns
EXPERIMENT   = "renovacion-prestamo-tuning"

MLFLOW_DIR.mkdir(parents=True, exist_ok=True)     # crear si no existe

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT)
print(f"MLflow tracking URI : {TRACKING_URI}")
print(f"MLflow experiment   : {EXPERIMENT}")

# ── 8.2 Definir espacio de búsqueda y ejecutar GridSearchCV ──────────────────
param_grid = {
    "n_estimators":     [10, 50, 100, 150],   # 4 opciones
    "max_depth":        [4, 10, 20, None],    # 4 opciones
    "min_samples_leaf": [1, 2, 4],            # 3 opciones
    "class_weight":     ["balanced"],         # 1 opción (desbalance residual)
}
# 4 x 4 x 3 x 1 = 48 combinaciones x 5 folds = 240 entrenamientos

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

gs = GridSearchCV(
    estimator          = RandomForestClassifier(random_state=RANDOM_STATE),
    param_grid         = param_grid,
    cv                 = skf,
    scoring            = "f1",
    n_jobs             = -1,
    verbose            = 1,
    return_train_score = True,
)

print("\nEjecutando GridSearchCV (48 combinaciones x 5 folds)...")
gs.fit(X_train_u, y_train_u)

print(f"\nMejores parámetros : {gs.best_params_}")
print(f"Mejor F1 CV        : {gs.best_score_:.4f}")

# ── 8.3 Registrar en MLflow ───────────────────────────────────────────────────
n_combinaciones = len(gs.cv_results_["params"])

with mlflow.start_run(run_name="GridSearch-RandomForest") as run:
    # Parámetros del mejor estimador
    mlflow.log_params(gs.best_params_)
    mlflow.log_param("metodo_busqueda",  "GridSearchCV")
    mlflow.log_param("n_combinaciones",  n_combinaciones)
    mlflow.log_param("tecnica_balanceo", "UnderSampling")
    mlflow.log_param("n_folds",          5)
    mlflow.log_param("random_state",     RANDOM_STATE)

    # Evaluar en test set (clase positiva = renueva el préstamo)
    y_pred_gs  = gs.best_estimator_.predict(X_test)
    y_proba_gs = gs.best_estimator_.predict_proba(X_test)[:, 1]
    mlflow.log_metrics({
        "accuracy":   round(accuracy_score(y_test,                    y_pred_gs), 4),
        "f1":         round(f1_score(y_test,                          y_pred_gs), 4),
        "recall":     round(recall_score(y_test,                      y_pred_gs), 4),
        "precision":  round(precision_score(y_test, y_pred_gs, zero_division=0), 4),
        "roc_auc":    round(roc_auc_score(y_test,                    y_proba_gs), 4),
        "cv_f1_mean": round(gs.best_score_, 4),
    })

    # Registrar modelo — MLflow 3.x: name= (antes artifact_path=) y skops_trusted_types
    # para que el formato skops acepte los árboles del bosque (sklearn.tree._tree.Tree)
    mlflow.sklearn.log_model(
        sk_model            = gs.best_estimator_,
        name                = "modelo_gridsearch",
        skops_trusted_types = ["sklearn.tree._tree.Tree"],
    )
    run_id_gs = run.info.run_id
    print(f"\nRun ID GridSearch: {run_id_gs}")

print("\n=== RESULTADO GRIDSEARCH ===")
print(classification_report(y_test, y_pred_gs, digits=3))

# Guardar run_id para scripts posteriores
Path("src").mkdir(exist_ok=True)
with open("src/run_ids.txt", "w") as f:
    f.write(f"gridsearch:{run_id_gs}\n")

print("✓ Run ID guardado en src/run_ids.txt")
print("  Siguiente paso: python pipeline/03_randomsearch_mlflow.py")
