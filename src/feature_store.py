from time import gmtime, strftime, sleep
from sagemaker.feature_store.feature_group import FeatureGroup

import sagemaker
import sys

import boto3
import pandas as pd
import numpy as np
import io
import time
import datetime


role = sagemaker.get_execution_role()
sess = sagemaker.Session()
region = sess.boto_region_name
bucket = sess.default_bucket()
prefix = 'whatnot_livestream_recsys'

def create_feature_groups():
    """
    Make feature groups for users, auctions, interactions
    interaction is being used for training
    """
    feature_group = FeatureGroup(name="users", sagemaker_session=sess)
    feature_group.create(
        s3_uri=f's3://{bucket}/{prefix}',
        enable_online_store=False,
        record_identifier_name="user_id",
        event_time_feature_name="event_time_feature_name",
        description="users features",
        role_arn=role)

def check_feature_group_status(feature_group):
    status = feature_group.describe().get("FeatureGroupStatus")
    while status == "Creating":
        print("Waiting for Feature Group to be Created")
        time.sleep(5)
        status = feature_group.describe().get("FeatureGroupStatus")
    print(f"FeatureGroup {feature_group.name} successfully created.")

def ingest_feature(feature_group, df):
    feature_group.ingest(data_frame=df, max_workers=1, max_processes = 1, wait=True)

def get_feature_values(user_id, livestream_id):
    """
    Return feature values for online (prediction) query.
    Also you can use SDK for Python (Boto3)

    Arguments: 
        id (int): user_id or livestream_id
    
    Returns:
        feature values (numpy array[float])    
    """
    featurestore_runtime =  sess.boto_session.client(
        service_name='sagemaker-featurestore-runtime', 
        region_name=region)
    
    user_features = featurestore_runtime.get_record(
        FeatureGroupName="users", 
        RecordIdentifierValueAsString=user_id)
    
    livestream_features = featurestore_runtime.get_record(
        FeatureGroupName="livestreams", 
        RecordIdentifierValueAsString=livestream_id)    
    return 

check_feature_group_status(feature_group="users")





