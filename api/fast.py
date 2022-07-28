from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


@app.get("/")
def root():
    return {"greeting": "hello batch 852!"}



@app.get("/predict")
def predict(
        pickup_datetime,  # 2013-07-06 17:18:00
        lon1,  # -73.950655
        lat1,  # 40.783282
        lon2,  # -73.984365
        lat2,  # 40.769802
        passcount):  # 1


    X_pred = pd.DataFrame({
    "key": ["truc"],
    "pickup_datetime": [pickup_datetime + " UTC"],
    "pickup_longitude": [float(lon1)],
    "pickup_latitude": [float(lat1)],
    "dropoff_longitude": [float(lon2)],
    "dropoff_latitude": [float(lat2)],
    "passenger_count": [int(passcount)]})

    # load the trained model
    pipeline = joblib.load("model.joblib")

    # make a prediction for the parameters passed to the API
    y_pred = pipeline.predict(X_pred)

    # return the prediction
    prediction = y_pred[0] # basic types, int, float, str, list, dict

    return {"fare": prediction}
