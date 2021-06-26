#!/bin/bash
echo "Dataset:" $1  "Backbone:" $2  "GPU index:" $3 "Tag:" $4 

python train_classifier_sideout_classifier_ybranch.py --gpu $3 --config ./configs/$1/$2-ybranch-$1.yaml --tag=$4