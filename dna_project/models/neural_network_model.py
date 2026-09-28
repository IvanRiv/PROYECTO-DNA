from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .base_model import BaseModel


class NeuralNetworkModel(BaseModel):

    def create_pipeline(self):

        return Pipeline(
            [
                (
                    "scaler",
                    StandardScaler()
                ),
                (
                    "clf",
                    MLPClassifier(
                        max_iter=500,
                        random_state=self.random_state
                    )
                )
            ]
        )

    def get_param_grid(self):

        return {
            "clf__hidden_layer_sizes": [
                (32, 16),
                (64, 32)
            ],
            "clf__alpha": [
                0.0001,
                0.001
            ]
        }

    def get_name(self):

        return "Red Neuronal (MLP)"
