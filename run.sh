#!/bin/bash

## step 0: preprocess the data 
python m0_preprocess.py

## declare all models and all data sets
declare data_sets=("rew_data_exp1")
#declare models=("model1" "model2" "model11" "RRmodel")
declare models=("model1"  "model2")

## step 1: fit the model
for data_set in "${data_sets[@]}"; do 
    for model in "${models[@]}"; do 
        echo Data set=$data_set Model=$model 
            python m1_fits.py -d=$data_set -n=$model -s=21123 -f=50 -c=50 -m='mle' -g='avg'
    done 
done
