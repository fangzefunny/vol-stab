#!/bin/bash

## step 0: preprocess the data 
python m0_preprocess.py

## declare all models and all data sets
declare data_sets=("rew_data_exp1" "pain_data_exp1")
declare models=("model1" "model2" "model7"
                "model8" "model11" 
                "modelE1" "modelE2" "RRmodel")

## step 1: fit the model
for data_set in "${data_sets[@]}"; do 
    for model in "${models[@]}"; do 
        echo Data set=$data_set Model=$model 
            python m1_fits.py -d=$data_set -n=$model -s=215 -f=100 -c=100 -m='mle'
    done 
done
