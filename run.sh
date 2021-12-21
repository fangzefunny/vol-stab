#!/bin/bash

## step 0: preprocess the data 
python m0_preprocess.py

## declare all models and all data sets
declare data_sets=("rew_data_exp1")
declare groups=("ind")
declare models=("model11_e" "model11_m")
#declare models=("model1" "model2" "model11" "RRmodel")

## step 1: fit the model
for data_set in "${data_sets[@]}"; do 
    for group in "${groups[@]}"; do 
        for model in "${models[@]}"; do 
            echo Data set=$data_set Group=$group Model=$model 
                python m1_fit.py -d=$data_set -n=$model -s=42 -f=100 -c=100 -l='mle' -g=$group
                #python m2_simulate.py -d=$data_set -n=$model
        done 
    done
done
