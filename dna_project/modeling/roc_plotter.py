import matplotlib.pyplot as plt

from sklearn.metrics import RocCurveDisplay


class ROCCurvePlotter:

    def __init__(self):

        self.fig, self.ax = plt.subplots(
            figsize=(8, 6)
        )

    def add_model(
        self,
        trained_model,
        X_test,
        y_test
    ):

        pipeline = trained_model[
            "best_pipeline"
        ]

        name = trained_model[
            "model"
        ].get_name()

        RocCurveDisplay.from_estimator(
            pipeline,
            X_test,
            y_test,
            name=name,
            ax=self.ax
        )

    def save(self, output_path):

        self.ax.plot(
            [0, 1],
            [0, 1],
            "k--",
            label="Clasificador Aleatorio"
        )

        self.ax.set_title(
            "Comparativa de Curvas ROC - "
            "Evaluación en Validation (2024)"
        )

        self.ax.set_xlabel(
            "Tasa de Falsos Positivos "
            "(1 - Especificidad)"
        )

        self.ax.set_ylabel(
            "Tasa de Verdaderos Positivos "
            "(Sensibilidad / Recall)"
        )

        self.ax.grid(True)

        self.ax.legend()

        self.fig.tight_layout()

        self.fig.savefig(
            output_path,
            dpi=300
        )

        plt.close(
            self.fig
        )
