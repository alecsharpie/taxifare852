from taxifare852.data import get_data, clean_df
from sklearn.model_selection import train_test_split
from taxifare852.model import get_model
from taxifare852.pipeline import get_pipeline
from taxifare852.metrics import compute_rmse

from taxifare852.mlflowlogger import MLFlowBase

import joblib

class Trainer(MLFlowBase):

    def __init__(self, experiment_name):
        self.experiment_name = experiment_name
        self.MLFLOW_URI = "https://mlflow.lewagon.ai/"

    def train(self, model_params = {}):

        self.mlflow_create_run()

        df = get_data()

        df = clean_df(df)

        y_train = df["fare_amount"]
        X_train = df.drop("fare_amount", axis=1)

        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1)

        model = get_model()

        model.set_params(**model_params)

        self.mlflow_log_param('model_name', 'random_forest')

        for param_name, param_value in model_params.items():
            self.mlflow_log_param(param_name, param_value)

        pipeline = get_pipeline(model)

        pipeline.fit(X_train, y_train)

        joblib.dump(pipeline, 'model.joblib')

        y_pred = pipeline.predict(X_test)

        rmse = compute_rmse(y_test, y_pred)

        self.mlflow_log_metric('rmse', rmse)

        print('rmse: ', rmse)

        return pipeline
