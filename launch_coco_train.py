# Define IAM role
import boto3
import re
import json
import os
import numpy as np
import pandas as pd
import sagemaker
from sagemaker import get_execution_role
import argh
from argh import arg

role = get_execution_role()
sess = sagemaker.Session() # can use LocalSession() to run container locally
account = sess.boto_session.client('sts').get_caller_identity()['Account']

@arg('--bucket', help='s3 bucket for data retrieval and storage of results', default=None)
@arg('--image_name', help='Name of the Docker image to be used for training')
@arg('--region', help='', default='us-east-1')
@arg('--prefix_input', help='', default='detectron2-input')
@arg('--prefix_output', help='', default='detectron2-output')
@arg('--data_prefix', help='location in s3 for your data', default='train-coco')
@arg('--job_name', help='Name of the SageMaker training job to launch', default='d2-coco-train')
@arg('--instance_count', help='Number of instances to train on', default=2)
@arg('--instance_type', help='Type of EC2 instances to train on, for ', default='ml.p3.16xlarge')
@arg('--volume_size', help='Size of EBS volume attached to instance', default=100)
@arg('--use_spot', help='Whether to use spot instances for training', default=False)
@arg('--metric_path', help='Location for metric definition file', default=None)
@arg('--role', help='SageMaker execution role', default=None)
@arg('--max_run_time', help='', default=80000)
@arg('--max_wait_time', help='', default=None)
@arg('--hyperparam_path', help='Location for hyperparameters file', default=None)
def run_d2_sm(bucket=None, 
              image_name=None, 
              metric_path=None, 
              job_name='d2-coco-train', 
              region='us-east-1', 
              prefix_input='detectron2-input', 
              prefix_output='detectron2-output',
              instance_count=2, 
              data_prefix='train-coco',
              instance_type='ml.p3.16xlarge', 
              volume_size=100, 
              use_spot=False, 
              role=None, 
              max_run_time=80000, 
              max_wait_time=None,
              hyperparam_path=None):
    """
    Utility for launching detectron2 training jobs using the SageMaker Python SDK.
    Has options for launching jobs using spot instances, if launching spot,
    please set max_wait_time variable. Check different methods for additional documentation
    """
    assert bucket!=None, "Please specify a s3 bucket"
    assert hyperparam_path!=None, 'Please specify hyperparameters'
    
    s3_outpath = f's3://{bucket}/{prefix_output}/'
    if not image_name:
        image_name = f'{account}.dkr.ecr.{region}.amazonaws.com/d2-sm-coco:distributed'
    if not role:
        role = get_execution_role()
    metric_definitions = []
    with open(metric_path, 'r') as f:
        for line in f:
            metric_definitions.append(json.loads(line))

    with open(hyperparam_path, 'r') as f:
        hyperparameters = json.load(f)
        
    if use_spot:
        checkpoint_s3_uri = f"s3://{bucket}/checkpoints"
    else:
        checkpoint_s3_uri = None
        
    d2 = sagemaker.estimator.Estimator(image_name,
                                       role=role,
                                       train_instance_count=instance_count, 
                                       train_instance_type=instance_type,
    #                                   train_instance_type="local_gpu", # use local_gpu for quick troubleshooting
                                       train_volume_size=volume_size,
                                       train_use_spot_instances=use_spot,
                                       output_path=f"s3://{bucket}/{prefix_output}",
                                       metric_definitions = metric_definitions,
                                       hyperparameters = hyperparameters, 
                                       sagemaker_session=sess,
                                       train_max_run=max_run_time,
                                       train_max_wait=max_wait_time,
                                       checkpoint_s3_uri=checkpoint_s3_uri)

    data_path = f"s3://{bucket}/{data_prefix}"
    print(f'Grabbing data from {data_path}')

    d2.fit({'training':data_path},
           job_name = job_name,
           wait=False) 
    print('Job launched!')

def main():
    parser = argh.ArghParser()
    parser.add_commands([run_d2_sm])
    parser.dispatch()
    
if __name__ == "__main__":
    main()