from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .base_model import BaseModel


class LogisticRegressionModel(BaseModel):

    def create_pipeline(self):

        return Pipeline(
            [
                (
                    "scaler",
                    StandardScaler()
                ),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000,
                        random_state=self.random_state
                    )
                )
            ]
        )

    def get_param_grid(self):

        return {
            "clf__C": [
                0.01,
                0.1,
                1.0,
                10.0
            ],
            "clf__solver": [
                "lbfgs",
                "liblinear"
            ]
        }

    def get_name(self):

        return "Regresion Logistica"
