#!/bin/bash

## step 0: preprocess the data 
python m0_preprocess.py

## declare all models and all data sets
declare data_sets=("rew_data_exp1" "rew_data_exp1")
declare models=("model1" "model2" "model7"
                "model8" "model11" "RRmodel")

## step 1: fit the model
<<<<<<< HEAD
for model in "${models[@]}"; do 
    echo Model=$model
        python m1_fits.py -n=$model -s=215 -f=100 -c=100
done 
=======
for data_set in "${data_sets[@]}"; do 
    for model in "${models[@]}"; do 
        echo Data set=$data_set Model=$model 
            python m1_fits.py -d=$data_set -n=$model -s=215 -f=100 -c=100
    done 
done
>>>>>>> d2b367c9f28c2244f712463f4cf71043a1c8e228
