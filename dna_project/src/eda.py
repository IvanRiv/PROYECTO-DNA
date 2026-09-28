from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
# SCRIPT: ANÁLISIS EXPLORATORIO DE DATOS COMPLETO (EDA)
# Proyecto: Predicción de Alertas de Demanda Operativa - DNA (UMSS 2026)
# ==============================================================================

class DNADataEDA:
    def __init__(self, input_file, output_dir):
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir)
        self.df = None
        sns.set_theme(style="whitegrid")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self):
        self.df = pd.read_csv(self.input_file)
        print("=" * 60)
        print("DATASET CARGADO EN EDA")
        print("=" * 60)
        print(f"Archivo: {self.input_file}")
        print(f"Dimensiones: {self.df.shape}")
        return self

    def general_overview(self):
        print("\n" + "=" * 60)
        print("1. COMPRENSIÓN GENERAL DE LOS DATOS")
        print("=" * 60)
        print("\nPrimeras filas:")
        print(self.df.head())
        print("\nEstadísticas descriptivas:")
        print(self.df.describe().T)
        return self

    def convert_categorical_variables(self):
        columnas = ["nivel_complejidad", "tipologia"]
        for columna in columnas:
            self.df[columna] = self.df[columna].astype("category")
        return self

    def analyze_missing_values(self):
        missing = self.df.isnull().sum()
        print("\n" + "=" * 60)
        print("VALORES FALTANTES")
        print("=" * 60)
        print(missing)
        return missing

    def analyze_duplicates(self):
        duplicates = self.df.duplicated().sum()
        print(f"\nRegistros duplicados: {duplicates}")
        return duplicates

    def analyze_zeros(self):
        zeros = (self.df == 0).sum()
        print("\nCantidad de ceros por columna:")
        print(zeros)
        return zeros

    def analyze_target(self):
        print("\n" + "=" * 60)
        print("ANÁLISIS DEL TARGET Y JUSTIFICACIÓN CONCEPTUAL")
        print("=" * 60)
        total_obs = len(self.df)
        positivos = (self.df["target_demanda"] == 1).sum()
        pct_pos = (positivos / total_obs) * 100
        activos_df = self.df[self.df["target_demanda"] == 1]
        exactamente_1 = (activos_df["total_atenciones"] == 1).sum()
        pct_exactamente_1 = (exactamente_1 / positivos) * 100 if positivos > 0 else 0
        print(f"Total observaciones (Tipología-Mes): {total_obs}")
        print(f"Meses con Demanda Inactiva (0): {total_obs - positivos} ({(100 - pct_pos):.2f}%)")
        print(f"Meses con Demanda Activa (1): {positivos} ({pct_pos:.2f}%)")
        print(f"\nHALLAZGO CLAVE DEL EDA:")
        print(f" De los {positivos} meses activos, {exactamente_1} registros ({pct_exactamente_1:.2f}%) tienen exactamente 1 sola atención.")
        print(f"\nDECISIÓN METODOLÓGICA DERIVADA:")
        print(" Este comportamiento empírico demuestra que el target representa la Ocurrencia o")
        print(" Activación de Demanda Operativa (Presencia/Ausencia), y NO picos críticos o sobrecarga.")
        return self

    def plot_target_distribution(self):
        plt.figure(figsize=(6, 4))
        sns.countplot(
            data=self.df,
            x="target_demanda",
            hue="target_demanda",
            palette="viridis",
            legend=False
        )
        plt.title("Distribución de Target Demanda (0: Inactivo, 1: Activo)")
        plt.xlabel("Target Demanda")
        plt.ylabel("Cantidad de Meses-Tipología")
        plt.tight_layout()
        plt.savefig(self.output_dir / "distribucion_target.png", dpi=300)
        plt.close()
        return self

    def analyze_demographic_variables(self):
        columns = ["hombres", "mujeres", "ninos_ninas", "adolescentes", "total_atenciones"]
        print("\nEstadísticas demográficas:")
        print(self.df[columns].describe())
        return self

    def plot_monthly_seasonality(self):
        monthly = self.df.groupby("mes")["target_demanda"].sum()
        plt.figure(figsize=(10, 4))
        ax = monthly.plot(kind="bar", color="skyblue")
        nombres_meses = ['Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
        ax.set_xticklabels(nombres_meses[:len(monthly)], rotation=0)
        plt.title("Total de Activaciones de Demanda por Mes (Estacionalidad)")
        plt.xlabel("Mes Operativo")
        plt.ylabel("Cantidad de Activaciones")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig(self.output_dir / "estacionalidad_mensual.png", dpi=300)
        plt.close()
        return self

    def analyze_typologies(self):
        print("\nTop 10 tipologías más frecuentes:")
        print(self.df["tipologia"].value_counts().head(10))
        return self

    def plot_top_typologies(self):
        top = self.df.groupby("tipologia", observed=False)["target_demanda"].sum().nlargest(10)
        plt.figure(figsize=(10, 5))
        top.plot(kind="barh", color="teal")
        plt.gca().invert_yaxis()
        plt.title("Top 10 Tipologías con Más Meses de Demanda Activa")
        plt.xlabel("Frecuencia de Activación")
        plt.tight_layout()
        plt.savefig(self.output_dir / "top_10_tipologias.png", dpi=300)
        plt.close()
        return self

    def plot_attention_distribution(self):
        data = self.df[self.df["total_atenciones"] > 0]["total_atenciones"]
        plt.figure(figsize=(7, 4))
        sns.histplot(data, kde=True, bins=10, color="purple")
        plt.title("Distribución del Volumen de Atenciones (Solo Meses Activos)")
        plt.xlabel("Total de Atenciones")
        plt.ylabel("Frecuencia")
        plt.tight_layout()
        plt.savefig(self.output_dir / "distribucion_atenciones.png", dpi=300)
        plt.close()
        return self

    def plot_temporal_correlation(self):
        columns = ["total_atenciones", "target_demanda", "atenciones_lag_1", "atenciones_lag_2", "atenciones_lag_3", "promedio_movil_3m"]
        correlation = self.df[columns].corr()
        plt.figure(figsize=(8, 6))
        sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Matriz de Correlación - Variables de Serie Temporal")
        plt.tight_layout()
        plt.savefig(self.output_dir / "correlacion_variables_temporales.png", dpi=300)
        plt.close()
        return self

    def analyze_outliers(self):
        q1 = self.df["total_atenciones"].quantile(0.25)
        q3 = self.df["total_atenciones"].quantile(0.75)
        iqr = q3 - q1
        upper_limit = q3 + 1.5 * iqr
        outliers = self.df[self.df["total_atenciones"] > upper_limit]
        print(f"\nANÁLISIS DE OUTLIERS: Q1={q1}, Q3={q3}, IQR={iqr}, Límite={upper_limit}")
        return outliers

    def plot_attention_boxplot(self):
        plt.figure(figsize=(7, 3.5))
        sns.boxplot(x=self.df["total_atenciones"], color="lightcoral")
        plt.title("Detección de Valores Atípicos (Outliers)", fontweight="bold")
        plt.xlabel("Total de Atenciones")
        plt.tight_layout()
        plt.savefig(self.output_dir / "boxplot_outliers_atenciones.png", dpi=300)
        plt.close()
        return self

    def plot_complexity_boxplot(self):
        plt.figure(figsize=(9, 5))
        sns.boxplot(data=self.df, x="nivel_complejidad", y="total_atenciones", hue="nivel_complejidad", palette="Set2", legend=False)
        plt.title("Distribución de Atenciones Mensuales por Nivel de Complejidad")
        plt.xlabel("Nivel de Complejidad")
        plt.ylabel("Total de Atenciones Mensuales")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig(self.output_dir / "anexo_a_boxplot_atipicos.png", dpi=300)
        plt.close()
        return self

    def plot_age_group(self):
        data = self.df[["ninos_ninas", "adolescentes"]].sum().reset_index()
        data.columns = ["Grupo Etario", "Total Atenciones"]
        data["Grupo Etario"] = data["Grupo Etario"].replace({"ninos_ninas": "Niños y Niñas", "adolescentes": "Adolescentes"})
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.barplot(data=data, x="Grupo Etario", y="Total Atenciones", hue="Grupo Etario", palette="Blues_d", legend=False, ax=ax)
        ax.set_title("Volumen Total de Atenciones por Grupo Etario")
        ax.set_xlabel("Grupo Etario")
        ax.set_ylabel("Cantidad de Atenciones")
        for index, row in data.iterrows():
            ax.text(index, row["Total Atenciones"] * 0.05, f'{int(row["Total Atenciones"])}', ha="center", va="bottom", color="white", fontweight="bold")
        plt.tight_layout()
        plt.savefig(self.output_dir / "anexo_a_grupo_etario.png", dpi=300)
        plt.close()
        return self

    def plot_gender_distribution(self):
        data = self.df[["hombres", "mujeres"]].sum()
        plt.figure(figsize=(6, 6))
        plt.pie(data, labels=["Hombres", "Mujeres"], autopct="%1.1f%%", colors=["#4C72B0", "#DD8452"], startangle=140, explode=(0.05, 0))
        plt.title("Proporción de Atenciones por Sexo")
        plt.tight_layout()
        plt.savefig(self.output_dir / "anexo_a_distribucion_sexo.png", dpi=300)
        plt.close()
        return self

    def export_typology_catalog(self):
        table = self.df.groupby(["id_tipologia", "tipologia"], observed=True)["total_atenciones"].sum().reset_index().sort_values(by="total_atenciones", ascending=False)
        output_file = self.output_dir / "tabla_anexo_a_catalogo_completo.csv"
        table.to_csv(output_file, index=False)
        print(f"Catálogo exportado a: {output_file}")
        return table

    def run(self):
        self.load_data()
        self.general_overview()
        self.convert_categorical_variables()
        self.analyze_missing_values()
        self.analyze_duplicates()
        self.analyze_zeros()
        self.analyze_target()
        self.plot_target_distribution()
        self.analyze_demographic_variables()
        self.plot_monthly_seasonality()
        self.analyze_typologies()
        self.plot_top_typologies()
        self.plot_attention_distribution()
        self.plot_temporal_correlation()
        self.analyze_outliers()
        self.plot_attention_boxplot()
        self.plot_complexity_boxplot()
        self.plot_age_group()
        self.plot_gender_distribution()
        self.export_typology_catalog()
        print("\n" + "=" * 60)
        print("EDA COMPLETO FINALIZADO CORRECTAMENTE")
        print("=" * 60)
        return self