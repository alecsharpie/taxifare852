from taxifare852.data import get_data, clean_df
from sklearn.model_selection import train_test_split
from taxifare852.model import get_model, get_grid_model
from taxifare852.pipeline import get_pipeline
from taxifare852.metrics import compute_rmse

from taxifare852.mlflowlogger import MLFlowBase

from sklearn.model_selection import GridSearchCV

import joblib

class ParamTrainer(MLFlowBase):

    def __init__(self, experiment_name):
        self.experiment_name = experiment_name
        self.MLFLOW_URI = "https://mlflow.lewagon.ai/"

    def train(self, model_params = {}):

        df = get_data()

        df = clean_df(df)

        y_train = df["fare_amount"]
        X_train = df.drop("fare_amount", axis=1)

        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1)

        fitted_pipelines = {}

        for model_name, params in model_params.items():

            hyper_params = params['hyper_params']

            self.mlflow_create_run()

            model = get_grid_model(model_name)

            pipeline = get_pipeline(model)

            grid_search = GridSearchCV(
            pipeline,
            param_grid=hyper_params,
            cv=3
            )

            grid_search.fit(X_train, y_train)

            fitted_pipelines[model_name] = grid_search

            best_model = grid_search.best_estimator_

            self.mlflow_log_param('model_name', model_name)

            for param_name, param_value in grid_search.best_params_.items():
                self.mlflow_log_param(param_name, param_value)

            joblib.dump(best_model, f'{model_name}.joblib')

            y_pred = best_model.predict(X_test)

            rmse = compute_rmse(y_test, y_pred)

            self.mlflow_log_metric('rmse', rmse)

            print('rmse: ', rmse)

        return fitted_pipelines
