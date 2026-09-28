"""
==============================================================================
DASHBOARD INTERACTIVO DE ALERTAS OPERATIVAS PREVENTIVAS (STREAMLIT CORREGIDO)
Proyecto: Predicción de Alertas de Demanda Operativa - DNA (UMSS 2026)
==============================================================================
"""

import os
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# 1. Configuración general de la página
st.set_page_config(
    page_title="Sistema de Alertas Preventivas DNA",
    page_icon="🛡️",
    layout="wide"
)

# Mapeo de meses operativos (Febrero a Diciembre)
MESES_MAP = {
    2: "Febrero", 3: "Marzo", 4: "Abril", 5: "Mayo", 6: "Junio",
    7: "Julio", 8: "Agosto", 9: "Septiembre", 10: "Octubre",
    11: "Noviembre", 12: "Diciembre"
}

@st.cache_resource
def cargar_modelo_local():
    """Carga el pipeline entrenado de Regresión Logística Optimizada."""
    rutas = [
        "./models_saved/logistic_regression_tuned_model.pkl",
        "./logistic_regression_tuned_model.pkl",
        "../models_saved/logistic_regression_tuned_model.pkl"
    ]
    for r in rutas:
        if os.path.exists(r):
            return joblib.load(r)
    return None

pipeline = cargar_modelo_local()

# ------------------------------------------------------------------------------
# ENCABEZADO PRINCIPAL
# ------------------------------------------------------------------------------

st.title("🛡️ Sistema de Alertas Preventivas de Demanda Operativa")
st.subheader("Defensoría de la Niñez y Adolescencia - GAM Cochabamba")
st.markdown("---")

# Barra lateral de planificación (Corregido el selector de gestión)
st.sidebar.header("⚙️ Panel de Planificación")
gestion_sel = st.sidebar.selectbox("Gestión a Proyectar (t+1)", options=[2026, 2027, 2028, 2029, 2030], index=0)
mes_num_sel = st.sidebar.selectbox(
    "Mes a Proyectar (t+1)",
    options=list(MESES_MAP.keys()),
    format_func=lambda x: f"{x} - {MESES_MAP[x]}"
)

# Cargar dataset base de preparación
data_path = "./data/processed/dataset_dna_preparado.csv"
if not os.path.exists(data_path):
    data_path = "./dataset_dna_demanda_b1.csv"

if os.path.exists(data_path):
    df_raw = pd.read_csv(data_path)
else:
    df_raw = None

if pipeline is None:
    st.error("⚠️ No se pudo encontrar el artefacto del modelo (`logistic_regression_tuned_model.pkl`). Guarde el modelo en la carpeta `models_saved/`.")
else:
    st.sidebar.success("✅ Modelo `LR_OPT_v1.0.0` cargado exitosamente.")

    if df_raw is not None:
        # ======================================================================
        # CORRECCIÓN DE DUPLICADOS: SELECCIÓN DE LAS 98 TIPOLOGÍAS ÚNICAS
        # ======================================================================
        if "gestion" in df_raw.columns and "mes" in df_raw.columns:
            df_periodo = df_raw[(df_raw["gestion"] == (gestion_sel - 1)) & (df_raw["mes"] == mes_num_sel)]
            if len(df_periodo) >= 90:
                test_df = df_periodo.copy()
            else:
                test_df = df_raw.sort_values(by=["gestion", "mes"]).groupby("id_tipologia").last().reset_index()
        else:
            test_df = df_raw.groupby("id_tipologia").last().reset_index() if "id_tipologia" in df_raw.columns else df_raw.head(98).copy()

        # Reemplazar la variable temporal cíclica con el mes seleccionado para proyectar (t+1)
        test_df["mes_sin"] = np.sin(2 * np.pi * mes_num_sel / 12)
        test_df["mes_cos"] = np.cos(2 * np.pi * mes_num_sel / 12)

        # Lista estricta de las 13 características requeridas por el modelo
        features = [
            "mes_sin", "mes_cos",
            "pct_hombres_lag1", "pct_mujeres_lag1",
            "pct_ninos_ninas_lag1", "pct_adolescentes_lag1",
            "atenciones_lag_1", "atenciones_lag_2", "atenciones_lag_3",
            "promedio_movil_3m",
            "complejidad_Alta", "complejidad_Baja", "complejidad_Media"
        ]

        # Asegurar la existencia de las columnas
        for col in features:
            if col not in test_df.columns:
                test_df[col] = 0.0

        X_in = test_df[features].copy()
        
        # Inferencia con el pipeline de Regresión Logística
        probas = pipeline.predict_proba(X_in)[:, 1]

        test_df["probabilidad_activacion"] = probas
        test_df["prediccion_binaria"] = (probas >= 0.50).astype(int)

        def definir_alerta(p):
            if p >= 0.50:
                return "🔴 ROJO (Alerta Crítica)"
            elif p >= 0.30:
                return "🟡 AMARILLO (Moderado)"
            return "🟢 VERDE (Bajo Riesgo)"

        test_df["nivel_alerta"] = test_df["probabilidad_activacion"].apply(definir_alerta)

        # ----------------------------------------------------------------------
        # MÉTRICAS EJECUTIVAS
        # ----------------------------------------------------------------------
        tot_tipologias = len(test_df)
        activas_num = (test_df["prediccion_binaria"] == 1).sum()
        inactivas_num = tot_tipologias - activas_num
        altas_complejidad = test_df[(test_df["prediccion_binaria"] == 1) & (test_df.get("complejidad_Alta", 0) == 1)]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Catálogo Único Evaluado", f"{tot_tipologias} tipologías")
        col2.metric("Demanda Activa (Rojo)", f"{activas_num}", delta=f"{(activas_num/tot_tipologias)*100:.1f}%")
        col3.metric("Demanda Inactiva (Verde)", f"{inactivas_num}")
        col4.metric("Alertas en Alta Complejidad", f"{len(altas_complejidad)}")

        st.markdown("---")

        # ----------------------------------------------------------------------
        # PESTAÑAS INTERACTIVAS
        # ----------------------------------------------------------------------
        tab1, tab2, tab3 = st.tabs(["📊 Matriz de Alertas (Únicas)", "📈 Análisis de Distribución", "📄 Reporte de Planificación"])

        with tab1:
            st.subheader(f"Matriz de Riesgo Operativo - {MESES_MAP[mes_num_sel]} {gestion_sel}")
            
            filtro_alerta = st.multiselect(
                "Filtrar por Nivel de Alerta:",
                options=["🔴 ROJO (Alerta Crítica)", "🟡 AMARILLO (Moderado)", "🟢 VERDE (Bajo Riesgo)"],
                default=["🔴 ROJO (Alerta Crítica)", "🟡 AMARILLO (Moderado)"]
            )

            df_show = test_df[test_df["nivel_alerta"].isin(filtro_alerta)] if filtro_alerta else test_df

            cols_display = ["id_tipologia", "tipologia", "nivel_complejidad", "probabilidad_activacion", "nivel_alerta"]
            cols_valid = [c for c in cols_display if c in df_show.columns]

            # Ordenar por probabilidad descendente
            df_tabla = df_show[cols_valid].sort_values(by="probabilidad_activacion", ascending=False).reset_index(drop=True)

            st.dataframe(
                df_tabla,
                use_container_width=True,
                height=400
            )

        with tab2:
            st.subheader("Distribución de Probabilidades Predichas por el Modelo")
            col_g1, col_g2 = st.columns(2)

            with col_g1:
                fig_hist = px.histogram(
                    test_df,
                    x="probabilidad_activacion",
                    nbins=20,
                    title="Histograma de Probabilidades de Activación",
                    color="nivel_alerta",
                    color_discrete_map={
                        "🔴 ROJO (Alerta Crítica)": "red",
                        "🟡 AMARILLO (Moderado)": "gold",
                        "🟢 VERDE (Bajo Riesgo)": "green"
                    }
                )
                st.plotly_chart(fig_hist, use_container_width=True)

            with col_g2:
                if "nivel_complejidad" in test_df.columns:
                    fig_pie = px.pie(
                        test_df[test_df["prediccion_binaria"] == 1],
                        names="nivel_complejidad",
                        title="Alertas Rojas por Nivel de Complejidad Normativa",
                        color_discrete_sequence=px.colors.sequential.RdBu
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)

        with tab3:
            st.subheader("Acciones de Reasignación del Personal Multidisciplinario")
            st.info("""
            **Recomendación Institucional para el Mes Seleccionado:**
            * **Alertas Rojas de Alta Complejidad:** Garantizar guardia continua de Abogados y asignación inmediata de Psicólogos para peritajes de emergencia.
            * **Alertas Rojas de Media/Baja Complejidad:** Programar citaciones de conciliación y seguimiento por Trabajadores Sociales.
            """)
            
            # Exportar reporte en CSV optimizado
            csv_export = test_df.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="📥 Descargar Planificación Mensual en CSV",
                data=csv_export,
                file_name=f"planificacion_dna_{gestion_sel}_{mes_num_sel}.csv",
                mime="text/csv"
            )