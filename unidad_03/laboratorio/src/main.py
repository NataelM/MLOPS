"""
main.py — Pipeline con MLflow tracking.
Version 2: agrega registro de experimentos a DVC + MLflow.
"""

import logging
import sys
import os
import pickle

import mlflow
import mlflow.sklearn

from data_loader import cargar_datos
from preprocessing import (
    limpiar_na_strings,
    winsorizar_columnas,
    imputar_nulos,
    dividir_y_balancear,
)
from features import (
    crear_score_retrasos,
    crear_categorias_edad,
    crear_categorias_dependientes,
    estandarizar_variables,
    seleccionar_features,
)
from train import comparar_modelos, analizar_errores


# ── Parámetros del pipeline (cambian entre versiones) ─────────────────────────
PARAMS = {
    'ruta_datos': 'data/df.csv',
    'target': 'Default',
    'k_features': 20, # Cambiar de 15 a 20
    'test_size': 0.2, 
    'tecnica_balanceo': 'undersampling', # Cambiar de smote a undersampling
    'random_state': 42,
    'version_pipeline': 'v3-optimizada', # Cambiar de v1-baseline a v2-optimizada
}


def configurar_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(name)-20s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('pipeline_crediticio.log'),
        ],
    )


def main():
    configurar_logging()
    log = logging.getLogger(__name__)

    # ── Configurar MLflow ──────────────────────────────────────────────────────
    mlflow.set_experiment('banco-wiesse-crediticio')

    with mlflow.start_run(run_name=PARAMS['version_pipeline']):
        mlflow.log_params(PARAMS)
        log.info('MLflow run iniciado: %s', mlflow.active_run().info.run_id)

        # Pipeline de datos
        df = cargar_datos(PARAMS['ruta_datos'])
        df = limpiar_na_strings(df)
        df = winsorizar_columnas(df)
        df = imputar_nulos(df)

        # Feature engineering
        df = crear_score_retrasos(df)
        df = crear_categorias_edad(df)
        df = crear_categorias_dependientes(df)
        df = estandarizar_variables(df)

        # Selección y split
        excluir = [PARAMS['target'], 'ID', 'Edad_cat', 'Deps_cat']
        X = df.drop(columns=[c for c in excluir if c in df.columns])
        X = X.select_dtypes(include=['number'])
        y = df[PARAMS['target']]

        top_feats = seleccionar_features(X, y, k=PARAMS['k_features'])
        X_sel = X[top_feats]

        X_train, X_test, y_train, y_test = dividir_y_balancear(
            X_sel,
            y,
            test_size=PARAMS['test_size'],
            tecnica=PARAMS['tecnica_balanceo'],
            random_state=PARAMS['random_state'],
        )

        # Entrenar y comparar modelos
        resultados = comparar_modelos(X_train, y_train, X_test, y_test)

        mejor_nombre = max(
            resultados,
            key=lambda k: resultados[k]['metricas']['f1'],
        )
        mejor_modelo = resultados[mejor_nombre]['modelo']
        mejor_m = resultados[mejor_nombre]['metricas']
        errores = analizar_errores(mejor_modelo, X_test, y_test)

        # Registrar métricas
        mlflow.log_metrics({
            'accuracy':        mejor_m['accuracy'],
            'precision':       mejor_m['precision'],
            'recall':          mejor_m['recall'],
            'f1':              mejor_m['f1'],
            'roc_auc':         mejor_m['roc_auc'] or 0.0,
            'false_negatives': errores['FN'],
            'false_positives': errores['FP'],
            'costo_estimado':  errores['FN'] * 10000 + errores['FP'] * 1000,
        })

        # Registrar modelo como artefacto
        mlflow.sklearn.log_model(
            mejor_modelo,
            name='modelo_crediticio',
            registered_model_name='DefaultCrediticioWiesse',
            skops_trusted_types=['sklearn.tree._tree.Tree'],
        )
        # Guardar modelo localmente también
        os.makedirs('models', exist_ok=True)
        model_path = f'models/modelo_{PARAMS["version_pipeline"]}.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(mejor_modelo, f)
        mlflow.log_artifact(model_path)

        log.info(
            '=== RESULTADO: %s | F1=%.4f | Recall=%.4f | FN=%d ===',
            mejor_nombre,
            mejor_m['f1'],
            mejor_m['recall'],
            errores['FN'],
        )


if __name__ == '__main__':
    main()