#!/bin/bash
echo "Train GPU index:" $1 "Test GPU index:" $2 "Trial:" $3 

python train_classifier_sideout_classifier_ybranch.py --gpu $1 --config ./configs/fc100/res12-ybranch-fc100.yaml --tag=$3

python test_few_shot_ybranch.py --shot 1 --gpu $2 --config ./configs/fc100/test_few_shot_fc100_general.yaml --load=./save/res12-ybranch-fc100_$3/max-f-va.pth --method='cos,cos,cos,cos' --logits_coeff_list='20,20,20,20' --sideout --feat_source_list='final,before_avgpool,final,before_avgpool' --branch_list='1,1,2,2'

python test_few_shot_ybranch.py --shot 5 --gpu $2 --config ./configs/fc100/test_few_shot_fc100_general.yaml --load=./save/res12-ybranch-fc100_$3/max-f-va.pth --method='cos,cos,cos,cos' --logits_coeff_list='20,20,20,20' --sideout --feat_source_list='final,before_avgpool,final,before_avgpool' --branch_list='1,1,2,2'