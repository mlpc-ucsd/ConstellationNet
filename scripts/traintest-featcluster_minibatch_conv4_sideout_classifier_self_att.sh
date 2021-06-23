#!/bin/bash
echo "GPU index:" $1 "Experiment:" $2 "Trial:" $3
python train_classifier_sideout_classifier.py --gpu $1 --config ./configs/current_configs/featcluster_minibatch_conv4_sideout_classifier_self_att/$2.yaml --tag=$3
python test_few_shot.py --shot 1 --gpu $1 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4featcluster-minibatch-$2_$3/max-va.pth
python test_few_shot.py --shot 5 --gpu $1 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4featcluster-minibatch-$2_$3/max-va.pth


