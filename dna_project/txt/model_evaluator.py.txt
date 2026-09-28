from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)


class ModelEvaluator:

    def evaluate(
        self,
        trained_model,
        X_test,
        y_test
    ):

        best_pipeline = trained_model["best_pipeline"]

        y_pred = best_pipeline.predict(
            X_test
        )

        y_proba = best_pipeline.predict_proba(
            X_test
        )[:, 1]

        result = {
            "Modelo": trained_model["model"].get_name(),

            "Accuracy": round(
                accuracy_score(
                    y_test,
                    y_pred
                ),
                4
            ),

            "Precision": round(
                precision_score(
                    y_test,
                    y_pred,
                    zero_division=0
                ),
                4
            ),

            "Recall": round(
                recall_score(
                    y_test,
                    y_pred,
                    zero_division=0
                ),
                4
            ),

            "F1-Score": round(
                f1_score(
                    y_test,
                    y_pred,
                    zero_division=0
                ),
                4
            ),

            "ROC-AUC": round(
                roc_auc_score(
                    y_test,
                    y_proba
                ),
                4
            ),

            "Tiempo (s)": trained_model[
                "train_time"
            ],

            "Hiperparametros Optimos": str(
                trained_model["best_params"]
            )
        }

        return result
