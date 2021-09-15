#!/bin/bash

## step 0: preprocess the data 
python m0_preprocess.py

## declare all models and all data sets
declare models=("model1" "model2" "model7"
                "model8" "model11" "RRmodel")

## step 1: fit the model
for model in "${models[@]}"; do 
    echo Model=$model
        python m1_fits.py -n=$model -s=215 -f=100 -c=100
done 
