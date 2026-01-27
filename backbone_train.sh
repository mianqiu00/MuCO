#!/bin/bash
# sudo chmod +x ./run.sh
# nohup ./run.sh > nohup.log 2>&1 &


# Set which GPUs the launcher is allowed to use
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Launch multiprocesses, one for each visible GPU
nohup python backbone_train.py > backbone_train.log 2>&1 &
