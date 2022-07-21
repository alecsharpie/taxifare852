from taxifare852.data import get_data, clean_df, get_gcp_data
from sklearn.model_selection import train_test_split
from taxifare852.model import get_model, save_model_to_gcp
from taxifare852.pipeline import get_pipeline
from taxifare852.metrics import compute_rmse

from taxifare852.mlflowlogger import MLFlowBase

import joblib

from taxifare852.params import BUCKET_NAME

class Trainer(MLFlowBase):

    def __init__(self, experiment_name):
        self.experiment_name = experiment_name
        self.MLFLOW_URI = "https://mlflow.lewagon.ai/"

    def train(self, model_params = {}):

        self.mlflow_create_run()

        print('getting data...')

        df = get_gcp_data(1000)

        df = clean_df(df)

        y_train = df["fare_amount"]
        X_train = df.drop("fare_amount", axis=1)

        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1)

        print('get model...')
        model = get_model()

        model.set_params(**model_params)

        self.mlflow_log_param('model_name', 'random_forest')

        for param_name, param_value in model_params.items():
            self.mlflow_log_param(param_name, param_value)

        pipeline = get_pipeline(model)

        print('train model...')

        pipeline.fit(X_train, y_train)

        print('save model...')

        joblib.dump(pipeline, 'model.joblib')

        save_model_to_gcp(BUCKET_NAME)

        y_pred = pipeline.predict(X_test)

        rmse = compute_rmse(y_test, y_pred)

        self.mlflow_log_metric('rmse', rmse)

        print('rmse: ', rmse)

        return pipeline

if __name__ == '__main__':
    trainer = Trainer('[MLB] 852 alecsharpie - v3')
    trainer.train()
