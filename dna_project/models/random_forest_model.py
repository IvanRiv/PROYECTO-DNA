from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from .base_model import BaseModel


class RandomForestModel(BaseModel):

    def create_pipeline(self):

        return Pipeline(
            [
                (
                    "clf",
                    RandomForestClassifier(
                        random_state=self.random_state,
                        class_weight="balanced"
                    )
                )
            ]
        )

    def get_param_grid(self):

        return {
            "clf__n_estimators": [
                100, 200
            ],
            "clf__max_depth": [
                5, 10, None
            ],
            "clf__min_samples_split": [
                2, 5
            ]
        }

    def get_name(self):

        return "Random Forest"
