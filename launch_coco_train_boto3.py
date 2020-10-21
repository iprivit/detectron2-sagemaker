# Define IAM role
import boto3
import re
import json
import os
import numpy as np
import pandas as pd
from sagemaker import get_execution_role
import argh
from argh import arg
import sagemaker
from time import gmtime, strftime

sess = sagemaker.Session() # can use LocalSession() to run container locally
account = sess.boto_session.client('sts').get_caller_identity()['Account']
sm_client = boto3.client('sagemaker')
 #f"s3://{bucket}/{prefix_output}"


@arg('--bucket', help='s3 bucket for data retrieval and storage of results', default=sess.default_bucket())
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
@arg('--d2_config', help='Detectron2 configuration file to use', default="COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x.yaml")
@arg('--metric_path', help='Location for metric definition file', default=None)
@arg('--role', help='SageMaker execution role', default=None)
@arg('--max_run_time', help='', default=80000)
@arg('--max_wait_time', help='', default=None)
@arg('--hyperparam_path', help='Location for hyperparameters file', default=None)
def run_d2_sm(bucket=sess.default_bucket(), 
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
             d2_config="COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x.yaml", 
              hyperparam_path=None):
    """
    Utility for launching detectron2 training jobs using boto3 create_training_job API.
    Has options for launching jobs using spot instances, if launching spot,
    please set max_wait_time variable. Check different methods for additional documentation
    """
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
        checkpoint_s3_uri = f"s3://{bucket}/"
        stop_cond = {
              'MaxRuntimeInSeconds': max_run_time,
              'MaxWaitTimeInSeconds': max_wait_time
          }
    else:
        checkpoint_s3_uri = None
        stop_cond = {
              'MaxRuntimeInSeconds': max_run_time,
          }
    
    sm_client.create_training_job(

          TrainingJobName=job_name,
          HyperParameters=hyperparameters,
          AlgorithmSpecification={
              'TrainingImage': image_name,
    #           'AlgorithmName': 'string',
              'TrainingInputMode': 'File',
              'MetricDefinitions': metric_definitions,
              'EnableSageMakerMetricsTimeSeries': True
          },
          RoleArn=role,
          InputDataConfig=[
              {
                  'ChannelName': 'string',
                  'DataSource': {
                      'S3DataSource': {
                          'S3DataType': 'S3Prefix',
                          'S3Uri': f's3://{bucket}/{data_prefix}',
                          'S3DataDistributionType': 'FullyReplicated', # |'ShardedByS3Key'

                      },
    #                   'FileSystemDataSource': {
    #                       'FileSystemId': 'string',
    #                       'FileSystemAccessMode': 'rw'|'ro',
    #                       'FileSystemType': 'EFS'|'FSxLustre',
    #                       'DirectoryPath': 'string'
    #                   }
                  },
                  'ContentType': 'string',
                  'CompressionType': 'None',
                  'RecordWrapperType': 'None',
                  'InputMode': 'File',
                  'ShuffleConfig': {
                      'Seed': 123
                  }
              },
          ],
          OutputDataConfig={
              'S3OutputPath': s3_outpath
          },
          ResourceConfig={
              'InstanceType': instance_type,
              'InstanceCount': instance_count,
              'VolumeSizeInGB': volume_size,
          },
    #       VpcConfig={
    #           'SecurityGroupIds': [
    #               'string',
    #           ],
    #           'Subnets': [
    #               'string',
    #           ]
    #       },
          StoppingCondition=stop_cond,

          EnableNetworkIsolation=False,
          EnableInterContainerTrafficEncryption=False,
          EnableManagedSpotTraining=False,
    )

    print('Job launched!')
    
@arg('--job_name', help='Name of the SageMaker training job to describe', default='d2-coco-train')
def check_d2_sm(job_name='d2-coco-train'):
    """
    Utility for checking training job status
    """
    contents = sm_client.describe_training_job(TrainingJobName=job_name)
    print('Job ARN: ', contents['TrainingJobArn'])
    print('Job Status: ',contents['TrainingJobStatus'])
    print('Hyperparameters: ',contents['HyperParameters'])
    
def main():
    parser = argh.ArghParser()
    parser.add_commands([run_d2_sm, check_d2_sm])
    parser.dispatch()
    
if __name__ == "__main__":
    main()
    
    
# create_training_job(

#       TrainingJobName=job_name,
#       HyperParameters=hyperparameters,
#       AlgorithmSpecification={
#           'TrainingImage': image_name,
#           'AlgorithmName': 'string',
#           'TrainingInputMode': 'Pipe'|'File',
#           'MetricDefinitions': metric_definitions,
#           'EnableSageMakerMetricsTimeSeries': True|False
#       },
#       RoleArn=role,
#       InputDataConfig=[
#           {
#               'ChannelName': 'string',
#               'DataSource': {
#                   'S3DataSource': {
#                       'S3DataType': 'ManifestFile'|'S3Prefix'|'AugmentedManifestFile',
#                       'S3Uri': 'string',
#                       'S3DataDistributionType': 'FullyReplicated', # |'ShardedByS3Key'
#                       'AttributeNames': [
#                           'string',
#                       ]
#                   },
#                   'FileSystemDataSource': {
#                       'FileSystemId': 'string',
#                       'FileSystemAccessMode': 'rw'|'ro',
#                       'FileSystemType': 'EFS'|'FSxLustre',
#                       'DirectoryPath': 'string'
#                   }
#               },
#               'ContentType': 'string',
#               'CompressionType': 'None'|'Gzip',
#               'RecordWrapperType': 'None'|'RecordIO',
#               'InputMode': 'Pipe'|'File',
#               'ShuffleConfig': {
#                   'Seed': 123
#               }
#           },
#       ],
#       OutputDataConfig={
#           'S3OutputPath': "s3://{}/{}".format(bucket, prefix_output)
#       },
#       ResourceConfig={
#           'InstanceType': instance_type,
#           'InstanceCount': instance_count,
#           'VolumeSizeInGB': volume_size,
#       },
#       VpcConfig={
#           'SecurityGroupIds': [
#               'string',
#           ],
#           'Subnets': [
#               'string',
#           ]
#       },
#       StoppingCondition={
#           'MaxRuntimeInSeconds': 123,
#           'MaxWaitTimeInSeconds': 123
#       },
#       Tags=[
#           {
#               'Key': 'string',
#               'Value': 'string'
#           },
#       ],
#       EnableNetworkIsolation=True|False,
#       EnableInterContainerTrafficEncryption=True|False,
#       EnableManagedSpotTraining=True|False,
#       CheckpointConfig={
#           'S3Uri': 'string',
#           'LocalPath': 'string'
#       },
#       DebugHookConfig={
#           'LocalPath': 'string',
#           'S3OutputPath': 'string',
#           'HookParameters': {
#               'string': 'string'
#           },
#           'CollectionConfigurations': [
#               {
#                   'CollectionName': 'string',
#                   'CollectionParameters': {
#                       'string': 'string'
#                   }
#               },
#           ]
#       },
#       DebugRuleConfigurations=[
#           {
#               'RuleConfigurationName': 'string',
#               'LocalPath': 'string',
#               'S3OutputPath': 'string',
#               'RuleEvaluatorImage': 'string',
#               'InstanceType': 'ml.t3.medium',
#               'VolumeSizeInGB': 123,
#               'RuleParameters': {
#                   'string': 'string'
#               }
#           },
#       ],
#       TensorBoardOutputConfig={
#           'LocalPath': 'string',
#           'S3OutputPath': 'string'
#       },
#       ExperimentConfig={
#           'ExperimentName': 'string',
#           'TrialName': 'string',
#           'TrialComponentDisplayName': 'string'
#       }
#   )


