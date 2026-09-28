import os
import re
import pandas as pd
import numpy as np

# ==============================================================================
# SCRIPT 1: GENERACIÓN DE DATASET BASE B1
# Proyecto: Predicción de Alertas de Demanda Operativa - DNA (UMSS 2026)
# ==============================================================================

# 1. Definir la ruta raíz de los archivos Excel originales
ruta_raiz = "./archivos/matrices dna"

# Mapeo de meses (11 meses operativos: Febrero a Diciembre; Enero no se registra por receso)
meses_map = {
    'enero': 1,
    'febrero': 2,
    'marzo': 3,
    'abril': 4,
    'mayo': 5,
    'junio': 6,
    'julio': 7,
    'agosto': 8,
    'septiembre': 9,
    'octubre': 10,
    'noviembre': 11,
    'diciembre': 12
}

# Grupos de complejidad normativa
alta = {9, 10, 12, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 93, 94, 95, 96}
baja = {1, 7, 8, 48, 49, 50, 51, 52, 53, 54, 55, 56, 86, 87, 88, 89, 90}

def asignar_complejidad(id_tip):
    if id_tip in alta:
        return 'Alta'
    elif id_tip in baja:
        return 'Baja'
    return 'Media'

registros_extraccion = []

# 2. Extracción recursiva desde las planillas Excel
for root, dirs, files in os.walk(ruta_raiz):
    for file in files:
        if file.endswith(('.xlsx', '.xls')) and not file.startswith('~$'):
            filepath = os.path.join(root, file)
            match_anio = re.search(r'202[3-6]', filepath)
            gestion = int(match_anio.group(0)) if match_anio else 2024
            file_lower = file.lower()
            mes_num = None
            for nombre_mes, num in meses_map.items():
                if nombre_mes in file_lower:
                    mes_num = num
                    break
            if mes_num is None:
                match_mes = re.search(r'\b(0?[1-9]|1[5, 6])\b', file)
                mes_num = int(match_mes.group(0)) if match_mes else 2
            try:
                df_pob = pd.read_excel(filepath, sheet_name='Tipología de Población')
                df_nin = pd.read_excel(filepath, sheet_name='TIPOLOGIA DE NIÑEZ Y ADOLESCENC')
                for idx in range(9, 108):
                    id_raw = df_pob.iloc[idx, 1]
                    nombre_tip = str(df_pob.iloc[idx, 2]).strip()
                    if pd.notna(id_raw) and nombre_tip not in ['nan', '']:
                        try:
                            id_tip = int(id_raw)
                        except ValueError:
                            continue
                        hombres = pd.to_numeric(df_pob.iloc[idx, 4], errors='coerce')
                        mujeres = pd.to_numeric(df_pob.iloc[idx, 5], errors='coerce')
                        nn = pd.to_numeric(df_nin.iloc[idx, 2], errors='coerce')
                        a = pd.to_numeric(df_nin.iloc[idx, 3], errors='coerce')
                        h = int(np.nan_to_num(hombres))
                        m = int(np.nan_to_num(mujeres))
                        ninos = int(np.nan_to_num(nn))
                        adol = int(np.nan_to_num(a))
                        total_atenciones = h + m
                        registros_extraccion.append({
                            'gestion': gestion,
                            'mes': mes_num,
                            'id_tipologia': id_tip,
                            'tipologia': nombre_tip,
                            'hombres': h,
                            'mujeres': m,
                            'ninos_ninas': ninos,
                            'adolescentes': adol,
                            'total_atenciones': total_atenciones,
                            'nivel_complejidad': asignar_complejidad(id_tip)
                        })
            except Exception as e:
                print(f"Error procesando {filepath}: {e}")

# 3. Crear DataFrame base
df_base = pd.DataFrame(registros_extraccion)

# 4. Consolidación por mes y tipología
df_agrupado = df_base.groupby(
    ['gestion', 'mes', 'id_tipologia', 'tipologia', 'nivel_complejidad']
).agg({
    'hombres': 'sum',
    'mujeres': 'sum',
    'ninos_ninas': 'sum',
    'adolescentes': 'sum',
    'total_atenciones': 'sum'
}).reset_index()

# 5. Crear Variable Objetivo (1 = Demanda Activa, 0 = Inactiva)
df_agrupado['target_demanda'] = (df_agrupado['total_atenciones'] > 0).astype(int)

# 6. Ordenar cronológicamente
df_agrupado = df_agrupado.sort_values(by=['id_tipologia', 'gestion', 'mes']).reset_index(drop=True)

# 7. Crear Lags e Indicadores de Tendencia agrupados por tipología
df_agrupado['atenciones_lag_1'] = df_agrupado.groupby('id_tipologia')['total_atenciones'].shift(1).fillna(0).astype(int)
df_agrupado['atenciones_lag_2'] = df_agrupado.groupby('id_tipologia')['total_atenciones'].shift(2).fillna(0).astype(int)
df_agrupado['atenciones_lag_3'] = df_agrupado.groupby('id_tipologia')['total_atenciones'].shift(3).fillna(0).astype(int)

df_agrupado['promedio_movil_3m'] = (
    df_agrupado
    .groupby('id_tipologia')['total_atenciones']
    .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    .fillna(0)
    .round(2)
)

# 8. Exportar Dataset B1
output_b1 = "./data/raw/dataset_dna_demanda_b1.csv"
os.makedirs(os.path.dirname(output_b1), exist_ok=True)
df_agrupado.to_csv(output_b1, index=False, encoding='utf-8-sig')

print("¡Proceso completado con éxito!")
print(f"Dataset B1 guardado en: {output_b1}")
print(f"Total de observaciones temporales: {len(df_agrupado)}")
print("\nDistribución del Target de Demanda (0 = Inactivo, 1 = Demanda Activa):")
print(df_agrupado['target_demanda'].value_counts(normalize=True) * 100)