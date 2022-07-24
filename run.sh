#!/bin/bash

## step 0: preprocess the data 
python m0_preprocess.py
export CUDA_VISIBLE_DEVICES="0"

## declare all models and all data sets
declare data_sets=("exp1_rew")
declare models=("mix_Explore")

## step 1: fit the model
for data_set in "${data_sets[@]}"; do 
    for model in "${models[@]}"; do 
        echo Data set=$data_set  Model=$model 
        python m1_fit.py -d=$data_set -n=$model -s=420 -f=60 -c=60 
    done 
done
