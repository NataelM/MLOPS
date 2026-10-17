"""
Secciones 1-7 — Pipeline base sin tuning (Caso Renovación de Préstamo).

Reproduce el notebook "Caso Renovación de Préstamo.ipynb":
carga el dataset, renombra columnas, aplica capping + log1p, imputa nulos,
codifica las categóricas (One-Hot), agrega el cluster K-Means (K=3),
hace el split 70/30, compara Originales / Under / Over / SMOTE con un
DecisionTree y entrena un RandomForest base sobre Undersampling.
Exporta base_data.pkl para que los scripts de tuning lo reutilicen.

Ejecutar: python pipeline/01_base_pipeline.py
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (
    classification_report, accuracy_score, roc_auc_score,
    f1_score, recall_score, precision_score,
)
from imblearn.over_sampling import SMOTE

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ── Sección 1: Cargar dataset ─────────────────────────────────────────────────
DATA_PATH = Path("data/Renovacion_prestamo.csv")
if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"No se encontró {DATA_PATH}. Copia 'Dataset Renovacion_prestamo.csv' "
        "a data/Renovacion_prestamo.csv"
    )

df = pd.read_csv(DATA_PATH, sep=";")
print(f"Dataset cargado: {df.shape}")

# Renombrar columnas para un mejor entendimiento (igual que el notebook)
df.rename(columns={
    "LINEA_RENOVADO": "Linea_Renovado",
    "PLAZO_RENOVADO": "Plazo_Renovado",
    "USO_LINEA_TOTAL_TC_T2": "Uso_Linea",
    "USO_TRIM_LINEA_BBVA": "Uso_TrimLinea",
    "NR_ENTIDADES_TOTAL_T2": "Nro_Entidades",
    "DIFF_NRO_ENTIDA_TOTALES_T2_T12": "Dif_Entidades",
    "SDO_CONSUMO_T2": "Saldo_Consumo",
    "RESENCIA_OFERTA_PLD_RENOVADO": "Meses_oferta",
    "Ahorro_Sldo_Bco_T1": "Ahorro",
    "PConsumo_Sldo_Bco_T1": "Prestamo_vigente",
    "SDO_BCO_tot_sm_pasivo_Bco_6M": "Promed_6Mdeuda",
    "FLAG_LIMA_PROVINCIA": "Flag_LimProv",
    "CUBRIR_DEUDA_CONSUMO_SF_RENOVA_PLD": "Deuda_Cubierta%",
}, inplace=True)

# ── Sección 2: Exploración ────────────────────────────────────────────────────
TARGET = "FLAG_VENTA"
print(f"\nShape       : {df.shape}")
print(f"Columnas    : {list(df.columns)}")
print("\nDistribución del target:")
print(df[TARGET].value_counts())
print(df[TARGET].value_counts(normalize=True).round(4))
print(f"\nDuplicados  : {df.duplicated().sum()}")
print("\nNulos por columna (solo > 0):")
nulos = df.isna().sum()
print(nulos[nulos > 0].sort_values(ascending=False))

# ── Sección 3: Capping + transformación logarítmica ──────────────────────────
# Saldos negativos (deudas) -> 0 para poder aplicar log1p
for col in ["Ahorro", "Prestamo_vigente", "Promed_6Mdeuda"]:
    df[col] = np.maximum(0, df[col])

COLS_LOG = [
    "Uso_Linea", "Uso_TrimLinea", "Saldo_Consumo", "SUELDO_ESTIMADO", "ANTIGUEDAD_MES",
    "Linea_Renovado", "Ahorro", "Prestamo_vigente", "Promed_6Mdeuda", "Deuda_Cubierta%",
]
for col in COLS_LOG:
    df[f"{col}_LOG"] = np.log1p(df[col])

# ── Sección 4: Imputación de nulos ───────────────────────────────────────────
# > 15 % nulos: sampling aleatorio uniforme en [media - std, media + std]
for col in ["Uso_TrimLinea_LOG", "Uso_Linea_LOG", "Meses_oferta"]:
    media, std = df[col].mean(), df[col].std()
    nulos_col = df[col].isnull()
    df.loc[nulos_col, col] = np.random.uniform(
        max(0, media - std), media + std, nulos_col.sum()
    )

# < 15 % nulos: mediana
for col in ["Saldo_Consumo_LOG", "SUELDO_ESTIMADO_LOG", "ANTIGUEDAD_MES_LOG", "EDAD"]:
    df[col] = df[col].fillna(df[col].median())

# Categóricas: moda
CATEGORICAS = ["REGION", "SEXO", "EST_CIVIL"]
for col in CATEGORICAS:
    df[col] = df[col].fillna(df[col].mode()[0])

# ── Sección 5: One-Hot Encoding + Cluster K-Means ────────────────────────────
df_encoded = pd.get_dummies(df, columns=CATEGORICAS, drop_first=False, dtype=int)

FEATURES_CLUSTER = ["Uso_TrimLinea_LOG", "Prestamo_vigente_LOG", "Uso_Linea_LOG"]
kmeans = KMeans(n_clusters=3, init="k-means++", random_state=RANDOM_STATE, n_init=10)
df_encoded["Cluster"] = kmeans.fit_predict(df_encoded[FEATURES_CLUSTER])
print("\nDistribución de clientes por cluster:")
print(df_encoded["Cluster"].value_counts().sort_index().to_dict())

# Quitar las columnas originales que ya tienen versión _LOG
df_final = df_encoded.drop(columns=COLS_LOG)
print(f"\nDataset final: {df_final.shape}")

# ── Sección 6: Split 70/30 ────────────────────────────────────────────────────
# MES y CLIENTE no son variables predictivas
X = df_final.drop(columns=[TARGET, "MES", "CLIENTE"])
y = df_final[TARGET].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
)
print(f"\nTrain: {X_train.shape} | Test: {X_test.shape}")
print(f"Train target dist: {y_train.value_counts().to_dict()}")
print(f"Test  target dist: {y_test.value_counts().to_dict()}")

# ── Sección 6b: UnderSampling, OverSampling y SMOTE ──────────────────────────
n_minority   = int(y_train.sum())
n_majority   = int((y_train == 0).sum())
idx_majority = y_train[y_train == 0].index
idx_minority = y_train[y_train == 1].index

rng = np.random.RandomState(RANDOM_STATE)

# UnderSampling — reducir la clase mayoritaria al tamaño de la minoritaria
idx_under = rng.choice(idx_majority, size=n_minority, replace=False)
X_under = pd.concat([X_train.loc[idx_under], X_train.loc[idx_minority]])
y_under = pd.concat([y_train.loc[idx_under], y_train.loc[idx_minority]])

# OverSampling — duplicar la clase minoritaria al tamaño de la mayoritaria
idx_over = rng.choice(idx_minority, size=n_majority - n_minority, replace=True)
X_over = pd.concat([X_train, X_train.loc[idx_over]])
y_over = pd.concat([y_train, y_train.loc[idx_over]])

# SMOTE — generar ejemplos sintéticos de la clase minoritaria
smote = SMOTE(k_neighbors=5, sampling_strategy="minority", random_state=RANDOM_STATE)
X_smote, y_smote = smote.fit_resample(X_train, y_train)

print(f"\nUnderSampling: {X_under.shape} | dist: {y_under.value_counts().to_dict()}")
print(f"OverSampling : {X_over.shape} | dist: {y_over.value_counts().to_dict()}")
print(f"SMOTE        : {X_smote.shape} | dist: {pd.Series(y_smote).value_counts().to_dict()}")


def evaluar(modelo, nombre):
    """Métricas de la clase positiva (renueva = 1) sobre el test set."""
    y_pred  = modelo.predict(X_test)
    y_proba = modelo.predict_proba(X_test)[:, 1]
    res = {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "f1":        round(f1_score(y_test, y_pred), 4),
        "recall":    round(recall_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y_test, y_proba), 4),
    }
    print(f"\n[{nombre}]")
    for k, v in res.items():
        print(f"  {k:<9}: {v:.4f}")
    return res


# ── Sección 7: DecisionTree base (sin tuning) por estrategia de balanceo ─────
print("\n" + "=" * 50)
print("SECCIÓN 7 — DecisionTree base (sin tuning)")
print("=" * 50)

resultados_base = {}
for nombre, X_bal, y_bal in [
    ("Originales",    X_train, y_train),
    ("UnderSampling", X_under, y_under),
    ("OverSampling",  X_over,  y_over),
    ("SMOTE",         X_smote, y_smote),
]:
    dt = DecisionTreeClassifier(random_state=RANDOM_STATE)
    dt.fit(X_bal, y_bal)
    resultados_base[nombre] = evaluar(dt, f"DecisionTree — {nombre}")

# El notebook eligió Random Forest + UnderSampling: es la LÍNEA BASE del tuning
print("\n=== REPORTE COMPLETO — RandomForest + UnderSampling (línea base para tuning) ===")
rf_base = RandomForestClassifier(random_state=RANDOM_STATE)
rf_base.fit(X_under, y_under)
resultados_base["RandomForest-Under"] = evaluar(rf_base, "RandomForest — UnderSampling")
print(classification_report(y_test, rf_base.predict(X_test), digits=3))

print("\n=== RESUMEN LÍNEA BASE ===")
print(pd.DataFrame(resultados_base).T.to_string())

# ── Exportar datos para scripts de tuning ────────────────────────────────────
Path("src").mkdir(exist_ok=True)
with open("src/base_data.pkl", "wb") as f:
    pickle.dump((X_under, y_under, X_test, y_test), f)

print("\n✓ Datos exportados a src/base_data.pkl (train = UnderSampling)")
print("  Siguiente paso: python pipeline/02_gridsearch_mlflow.py")
