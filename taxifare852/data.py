import pandas as pd
from google.cloud import storage

def get_data():
    url = "s3://wagon-public-datasets/taxi-fare-train.csv"
    df = pd.read_csv(url, nrows=100)
    return df

def get_gcp_data(line_count):
    df = pd.read_csv(
        "https://wagon-public-datasets.s3.amazonaws.com/taxi-fare-ny/train_10k.csv",
        nrows=line_count)

    return df


def get_data_using_blob(line_count):

    # get data from aws s3
    # url = "s3://wagon-public-datasets/taxi-fare-train.csv"

    # get data from my google storage bucket
    BUCKET_NAME = "le-wagon-data"
    BUCKET_TRAIN_DATA_PATH = "data/train_1k.csv"

    data_file = "train_1k.csv"

    client = storage.Client()  # verifies $GOOGLE_APPLICATION_CREDENTIALS

    bucket = client.bucket(BUCKET_NAME)

    blob = bucket.blob(BUCKET_TRAIN_DATA_PATH)

    blob.download_to_filename(data_file)

    # load downloaded data to dataframe
    df = pd.read_csv(data_file, nrows=line_count)

    return df

def clean_df(df):
    df = df.dropna(how='any', axis='rows')
    df = df[(df.dropoff_latitude != 0) | (df.dropoff_longitude != 0)]
    df = df[(df.pickup_latitude != 0) | (df.pickup_longitude != 0)]
    if "fare_amount" in list(df):
        df = df[df.fare_amount.between(0, 4000)]
    df = df[df.passenger_count < 8]
    df = df[df.passenger_count >= 1]
    df = df[df["pickup_latitude"].between(left=40, right=42)]
    df = df[df["pickup_longitude"].between(left=-74.3, right=-72.9)]
    df = df[df["dropoff_latitude"].between(left=40, right=42)]
    df = df[df["dropoff_longitude"].between(left=-74, right=-72.9)]
    return df
