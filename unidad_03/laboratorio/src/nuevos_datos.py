import pandas as pd, numpy as np 
df = pd.read_csv('data/df.csv', sep=';') 

n = 200 
nuevo = pd.DataFrame({'ID': range(df['ID'].max()+1, df['ID'].max()+n+1),
                      'Default': np.random.binomial(1, 0.17, n),
                      'Prct_uso_tc': np.random.beta(2, 3, n).round(4),
                      'Prct_deuda_vs_ingresos': np.random.beta(2, 3, n).round(4),
                      'Mto_ingreso_mensual': np.random.uniform(500, 8000, n).round(0),
                      'Nro_dependiente': np.random.randint(0, 6, n),
                      'Edad': np.random.randint(21, 75, n),
                      'Nro_prestao_retrasados': np.random.randint(0, 6, n),
                      'Nro_prod_financieros_deuda': np.random.randint(0, 12, n),
                      'Nro_retraso_60dias': np.random.randint(0, 4, n),
                      'Nro_creditos_hipotecarios': np.random.randint(0, 4, n),
                      'Nro_retraso_ultm3anios': np.random.randint(0, 5, n), }) 
df2 = pd.concat([df, nuevo], ignore_index=True) 

df2.to_csv('data/df.csv', sep=';', index=False) 
print(f'Dataset actualizado: {len(df2)} filas')