# Define IAM role
import boto3
import re

import os
import numpy as np
import pandas as pd
from sagemaker import get_execution_role
import argh
from argh import arg

role = get_execution_role()

import sagemaker
from time import gmtime, strftime

sess = sagemaker.Session() # can use LocalSession() to run container locally
account = sess.boto_session.client('sts').get_caller_identity()['Account']

@arg('--bucket', help='s3 bucket for data retrieval and storage of results', default=sess.default_bucket())
@arg('--image_name', help='Name of the Docker image to be used for training')
@arg('--region', help='', default='us-east-1')
@arg('--prefix_input', help='', default='detectron2-input')
@arg('--prefix_output', help='', default='detectron2-output')
@arg('--job_name', help='Name of the SageMaker training job to launch', default='d2-coco-train')
@arg('--instance_count', help='Number of instances to train on', default=2)
@arg('--instance_type', help='Type of EC2 instances to train on, for ', default='ml.p3.16xlarge')
@arg('--volume_size', help='Size of EBS volume attached to instance', default=100)
@arg('--use_spot', help='Whether to use spot instances for training', default=False)
@arg('--d2_config', help='Detectron2 configuration file to use', default="COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x.yaml")
def run_d2_sm(bucket=sess.default_bucket(), image_name=None, job_name='d2-coco-train', region='us-east-1', 
              prefix_input='detectron2-input', prefix_output='detectron2-output', instance_count=2, 
              instance_type='ml.p3.16xlarge', volume_size=100, use_spot=False,
             d2_config="COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x.yaml"):
    if not image_name:
        image_name = f'{account}.dkr.ecr.{region}.amazonaws.com/d2-sm-coco:distributed'
    metric_definitions=[
        {
            "Name": "total_loss",
            "Regex": ".*total_loss:\s([0-9\\.]+)\s*"
        },
        {
            "Name": "loss_cls",
            "Regex": ".*loss_cls:\s([0-9\\.]+)\s*"
        },
        {
            "Name": "loss_box_reg",
            "Regex": ".*loss_box_reg:\s([0-9\\.]+)\s*"
        },
        {
            "Name": "loss_mask",
            "Regex": ".*loss_mask:\s([0-9\\.]+)\s*"
        },
        {
            "Name": "loss_rpn_cls",
            "Regex": ".*loss_rpn_cls:\s([0-9\\.]+)\s*"
        },
        {
            "Name": "loss_rpn_loc",
            "Regex": ".*loss_rpn_loc:\s([0-9\\.]+)\s*"
        }, 
        {
            "Name": "overall_training_speed",
            "Regex": ".*Overall training speed:\s([0-9\\.]+)\s*"
        },
        {
            "Name": "lr",  
            "Regex": ".*lr:\s([0-9\\.]+)\s*"
        },
        {
            "Name": "iter",  
            "Regex": ".*iter:\s([0-9\\.]+)\s*"
        }
    ]

    hyperparameters = {"config-file":d2_config, 
                       #"local-config-file" : "config.yaml", # if you'd like to supply custom config file, please add it in container_training folder, and provide file name here
                       "resume":"True", # whether to re-use weights from pre-trained model
                       "eval-only":"False", # whether to perform only D2 model evaluation
                      # opts are D2 model configuration as defined here: https://detectron2.readthedocs.io/modules/config.html#config-references
                      # this is a way to override individual parameters in D2 configuration from Sagemaker API
                       "opts": "SOLVER.MAX_ITER 20000"
                       }

    sessLocal = sagemaker.LocalSession() # can use LocalSession()

    d2 = sagemaker.estimator.Estimator(image_name,
                                       role=role,
                                       train_instance_count=instance_count, 
                                       train_instance_type= instance_type,
    #                                   train_instance_type="local_gpu", # use local_gpu for quick troubleshooting
                                       train_volume_size=volume_size,
                                       train_use_spot_instances=use_spot,
                                       output_path="s3://{}/{}".format(bucket, prefix_output),
                                       metric_definitions = metric_definitions,
                                       hyperparameters = hyperparameters, 
                                       sagemaker_session=sess)

    d2.fit({'training':f"s3://{bucket}/coco"},
           job_name = job_name,
           wait=False) 
    print('Job launched!')

def main():
    parser = argh.ArghParser()
    parser.add_commands([run_d2_sm])
    parser.dispatch()
    
if __name__ == "__main__":
    main()