#!/bin/bash
echo "GPU index:" $1 "Test GPU index:" $2 "Trial:" $3

python train_classifier_sideout_classifier_ybranch.py --gpu $1 --config ./configs/cifar-fs/res12-ybranch-cifar-fs.yaml --tag=$3


python test_few_shot_ybranch.py --shot 1 --gpu $2 --config ./configs/cifar-fs/test_few_shot_cifar-fs_general.yaml --load=./save/res12-ybranch-cifar-fs_$3/max-f-va.pth --method='cos,cos,cos' --logits_coeff_list='20,20,20' --sideout --feat_source_list='final,before_avgpool,final' --branch_list='1,1,2'


python test_few_shot_ybranch.py --shot 5 --gpu $2 --config ./configs/cifar-fs/test_few_shot_cifar-fs_general.yaml --load=./save/res12-ybranch-cifar-fs_$3/max-f-va.pth --method='cos,cos,cos' --logits_coeff_list='20,20,20' --sideout --feat_source_list='final,before_avgpool,final' --branch_list='1,1,2'
