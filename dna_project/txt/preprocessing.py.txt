from pathlib import Path
import numpy as np
import pandas as pd

# ==============================================================================
# SCRIPT 3: PREPROCESAMIENTO Y DIVISIÓN TEMPORAL ANTI-LEAKAGE (3 BLOQUES)
# Proyecto: Predicción de Alertas de Demanda Operativa - DNA (UMSS 2026)
# ==============================================================================

class DNADataPreprocessor:
    def __init__(self, input_file, output_dir):
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir)
        self.df = None
        self.X = None
        self.y = None
        
        # Partición en 3 Bloques
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None
        self.features = []

    def load_data(self):
        self.df = pd.read_csv(self.input_file)

    def sort_data(self):
        self.df = self.df.sort_values(by=["id_tipologia", "gestion", "mes"]).reset_index(drop=True)

    def remove_duplicates(self):
        self.df = self.df.drop_duplicates().reset_index(drop=True)

    def handle_missing_values(self):
        cols_conteo = ["total_atenciones", "hombres", "mujeres", "ninos_ninas", "adolescentes"]
        self.df[cols_conteo] = self.df[cols_conteo].fillna(0)

    def create_demographic_ratios(self):
        total = self.df["total_atenciones"]
        self.df["pct_hombres_raw"] = np.where(total > 0, self.df["hombres"] / total, 0)
        self.df["pct_mujeres_raw"] = np.where(total > 0, self.df["mujeres"] / total, 0)
        self.df["pct_ninos_ninas_raw"] = np.where(total > 0, self.df["ninos_ninas"] / total, 0)
        self.df["pct_adolescentes_raw"] = np.where(total > 0, self.df["adolescentes"] / total, 0)

    def create_demographic_lags(self):
        columnas = {
            "pct_hombres_raw": "pct_hombres_lag1",
            "pct_mujeres_raw": "pct_mujeres_lag1",
            "pct_ninos_ninas_raw": "pct_ninos_ninas_lag1",
            "pct_adolescentes_raw": "pct_adolescentes_lag1"
        }
        for columna_original, columna_lag in columnas.items():
            self.df[columna_lag] = self.df.groupby("id_tipologia")[columna_original].shift(1).fillna(0)

    def create_cyclic_month(self):
        self.df["mes_sin"] = np.sin(2 * np.pi * self.df["mes"] / 12)
        self.df["mes_cos"] = np.cos(2 * np.pi * self.df["mes"] / 12)

    def create_attention_lags(self):
        grupo = self.df.groupby("id_tipologia")["total_atenciones"]
        self.df["atenciones_lag_1"] = grupo.shift(1).fillna(0)
        self.df["atenciones_lag_2"] = grupo.shift(2).fillna(0)
        self.df["atenciones_lag_3"] = grupo.shift(3).fillna(0)

    def create_rolling_average(self):
        self.df["promedio_movil_3m"] = (
            self.df.groupby("id_tipologia")["total_atenciones"]
            .transform(lambda x: x.shift(1).rolling(window=3, min_periods=1).mean())
            .fillna(0)
        )

    def encode_complexity(self):
        self.df = pd.get_dummies(self.df, columns=["nivel_complejidad"], prefix="complejidad", drop_first=False)
        columnas = [col for col in self.df.columns if col.startswith("complejidad_")]
        self.df[columnas] = self.df[columnas].astype(int)
        self.features.extend(columnas)

    def create_features(self):
        self.features = [
            "mes_sin",
            "mes_cos",
            "pct_hombres_lag1",
            "pct_mujeres_lag1",
            "pct_ninos_ninas_lag1",
            "pct_adolescentes_lag1",
            "atenciones_lag_1",
            "atenciones_lag_2",
            "atenciones_lag_3",
            "promedio_movil_3m"
        ] + self.features
        self.X = self.df[self.features].copy()
        self.y = self.df["target_demanda"].copy()

    def split_data(self):
        # DIVISIÓN STRICT EN 3 BLOQUES (Train 2023 / Val 2024 / Test 2025)
        train_mask = (self.df["gestion"] == 2023)
        val_mask = (self.df["gestion"] == 2024)
        test_mask = (self.df["gestion"] == 2025)
        
        self.X_train = self.X.loc[train_mask].copy()
        self.y_train = self.y.loc[train_mask].copy()
        self.X_val = self.X.loc[val_mask].copy()
        self.y_val = self.y.loc[val_mask].copy()
        self.X_test = self.X.loc[test_mask].copy()
        self.y_test = self.y.loc[test_mask].copy()

    def save_data(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(self.output_dir / "dataset_dna_preparado.csv", index=False)
        # Guardar los 3 bloques independientes
        self.X_train.to_csv(self.output_dir / "X_train.csv", index=False)
        self.y_train.to_csv(self.output_dir / "y_train.csv", index=False)
        self.X_val.to_csv(self.output_dir / "X_val.csv", index=False)
        self.y_val.to_csv(self.output_dir / "y_val.csv", index=False)
        self.X_test.to_csv(self.output_dir / "X_test.csv", index=False)
        self.y_test.to_csv(self.output_dir / "y_test.csv", index=False)

    def run(self):
        self.load_data()
        self.sort_data()
        self.remove_duplicates()
        self.handle_missing_values()
        self.create_demographic_ratios()
        self.create_demographic_lags()
        self.create_cyclic_month()
        self.create_attention_lags()
        self.create_rolling_average()
        self.encode_complexity()
        self.create_features()
        self.split_data()
        self.save_data()
        print("\n--- PREPROCESAMIENTO FINALIZADO ---")
        print(f"X_train (2023): {self.X_train.shape} | Positivos: {self.y_train.mean():.2%}")
        print(f"X_val (2024): {self.X_val.shape} | Positivos: {self.y_val.mean():.2%}")
        print(f"X_test (2025): {self.X_test.shape} | Positivos: {self.y_test.mean():.2%}")

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    input_file = project_root / "data" / "raw" / "dataset_dna_demanda_b1.csv"
    output_dir = project_root / "data" / "processed"
    preprocessor = DNADataPreprocessor(input_file, output_dir)
    preprocessor.run()