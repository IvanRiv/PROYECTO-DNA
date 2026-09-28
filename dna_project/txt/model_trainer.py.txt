import time

from sklearn.model_selection import GridSearchCV


class ModelTrainer:

    def __init__(
        self,
        cv,
        scoring="f1",
        n_jobs=-1
    ):
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs

    def train(self, model):

        print(
            f"Procesando: {model.get_name()}..."
        )

        start_time = time.time()

        pipeline = model.create_pipeline()

        param_grid = model.get_param_grid()

        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=self.n_jobs
        )

        return grid, start_time

    def fit(self, model, X_train, y_train):

        grid, start_time = self.train(model)

        grid.fit(
            X_train,
            y_train
        )

        train_time = round(
            time.time() - start_time,
            2
        )

        return {
            "model": model,
            "grid": grid,
            "best_pipeline": grid.best_estimator_,
            "best_params": grid.best_params_,
            "train_time": train_time
        }
