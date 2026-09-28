import pandas as pd


class ModelComparator:

    def __init__(self):
        self.results = []

    def add_result(self, result):

        self.results.append(result)

    def get_results(self):

        return (
            pd.DataFrame(self.results)
            .sort_values(
                by="F1-Score",
                ascending=False
            )
        )

    def save_results(self, output_path):

        df_results = self.get_results()

        df_results.to_csv(
            output_path,
            index=False
        )

    def print_results(self):

        df_results = self.get_results()

        print("\n" + "=" * 85)
        print(
            "TABLA ENTREGABLE DE MODELADO "
            "(FASE 7.4 CRISP-DM)"
        )
        print("=" * 85)

        columns = [
            "Modelo",
            "Accuracy",
            "Precision",
            "Recall",
            "F1-Score",
            "ROC-AUC",
            "Tiempo (s)"
        ]

        print(
            df_results[columns]
            .to_string(index=False)
        )

        print("\nHIPERPARÁMETROS OPTIMIZADOS:")

        for _, row in df_results.iterrows():

            print(
                f"* {row['Modelo']}: "
                f"{row['Hiperparametros Optimos']}"
            )
