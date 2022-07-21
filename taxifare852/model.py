from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from google.cloud import storage
from datetime import datetime

def get_model():

    model = RandomForestRegressor()

    return model


def get_grid_model(model_name):

    if model_name == 'random_forest':
        model = RandomForestRegressor()

    if model_name == 'linear_regression':
        model = LinearRegression()

    return model


def save_model_to_gcp(BUCKET_NAME):

    date_time_now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    storage_location = f"models/random_forest_model_{date_time_now}.joblib"
    local_model_filename = "model.joblib"

    client = storage.Client()

    bucket = client.bucket(BUCKET_NAME)

    blob = bucket.blob(storage_location)

    blob.upload_from_filename(local_model_filename)
