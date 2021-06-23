#!/bin/bash
echo "Train GPU index:" $1 "Experiment:" $2 "Trial:" $3 "Test GPU index:" $4
python train_classifier_sideout_classifier.py --gpu $1 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_fc100_self_att/$2.yaml --tag=$3
python test_few_shot.py --shot 1 --gpu $4 --config ./configs/current_configs/test_few_shot_fc100_general.yaml --load_encoder=./save/res12featcluster-minibatch-onlyconcat-classifier-fc100-$2_$3/max-va.pth
python test_few_shot.py --shot 5 --gpu $4 --config ./configs/current_configs/test_few_shot_fc100_general.yaml --load_encoder=./save/res12featcluster-minibatch-onlyconcat-classifier-fc100-$2_$3/max-va.pth
