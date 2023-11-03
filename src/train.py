from feature_engineering import get_model_training_data
from ranking_model import train_model

import pickle 
import boto3


def train(
        bucket: str = "whatnot_models_bucket",
        key: str = "model_latest.pkl") -> None:
    """
    Re-trains the ML model and saves its artifacts to an S3 bucket

    Argumnets:
        bucket: the name of the s3 bucket
        key: the pickled model name

    Returns:
        None
    """
    # train model
    X, y = get_model_training_data()
    model = train_model(X, y)

    # save pickled mdoel to S3 bucket
    s3_resource = boto3.resource("s3")
    pickled_model = pickle.dumps(model)
    s3_resource.Object(bucket, key).put(Body=pickled_model)



    