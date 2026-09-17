# comparar_runs.py — script de comparación programática
import mlflow, pandas as pd

experimento = mlflow.get_experiment_by_name('banco-wiesse-crediticio')
runs_df = mlflow.search_runs(
    experiment_ids=[experimento.experiment_id],
    order_by=['metrics.recall DESC'],
    )
cols = ['tags.mlflow.runName', 'metrics.recall', 'metrics.f1',
        'metrics.false_negatives', 'metrics.costo_estimado',
        'params.k_features', 'params.tecnica_balanceo']

print('COMPARACIÓN DE RUNS — ordenado por Recall:')
print(runs_df[cols].to_string(index=False))
mejor_run = runs_df.iloc[0]
print(f'\nMEJOR RUN: {mejor_run["tags.mlflow.runName"]}')
print(f' Recall : {mejor_run["metrics.recall"]:.4f}')
print(f' F1 : {mejor_run["metrics.f1"]:.4f}')
print(f' FN : {mejor_run["metrics.false_negatives"]:.0f}')
modelo_produccion = mlflow.sklearn.load_model( f'runs:/{mejor_run["run_id"]}/modelo_crediticio')
print(f'\nModelo cargado: {type(modelo_produccion).__name__}')