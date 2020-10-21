#!/usr/bin/env bash

# mkdir efs
sudo mount -t nfs \
    -o nfsvers=4.1,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2 \
    172.31.2.152:/ \
    /home/ec2-user/SageMaker/efs

sudo chmod go+rw /home/ec2-user/SageMaker/efs