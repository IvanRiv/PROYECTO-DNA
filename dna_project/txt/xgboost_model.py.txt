from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from .base_model import BaseModel


class XGBoostModel(BaseModel):

    def __init__(
        self,
        scale_pos_weight,
        random_state=42
    ):
        super().__init__(random_state)

        self.scale_pos_weight = scale_pos_weight

    def create_pipeline(self):

        return Pipeline(
            [
                (
                    "clf",
                    XGBClassifier(
                        random_state=self.random_state,
                        eval_metric="logloss",
                        scale_pos_weight=self.scale_pos_weight
                    )
                )
            ]
        )

    def get_param_grid(self):

        return {
            "clf__n_estimators": [
                100,
                200
            ],
            "clf__max_depth": [
                3,
                6
            ],
            "clf__learning_rate": [
                0.01,
                0.1
            ]
        }

    def get_name(self):

        return "XGBoost"
