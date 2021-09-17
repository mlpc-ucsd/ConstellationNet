#!/usr/bin/env bash
echo "Dataset:" $1  "Backbone:" $2  "GPU index:" $3 "Tag:" $4 "logits_coeffs" $5

python test_few_shot_ybranch.py --shot 1 --gpu $3 --config ./configs/$1/test_few_shot_$1_general.yaml --load=./save/$2-ybranch-$1_$4/max-f-va.pth --method='cos,cos,cos,cos' --logits_coeff_list=${5:-'4,3,2,1'} --sideout --feat_source_list='final,before_avgpool,final,before_avgpool' --branch_list='1,1,2,2'

python test_few_shot_ybranch.py --shot 5 --gpu $3 --config ./configs/$1/test_few_shot_$1_general.yaml --load=./save/$2-ybranch-$1_$4/max-f-va.pth --method='cos,cos,cos,cos' --logits_coeff_list=${5:-'4,3,2,1'} --sideout --feat_source_list='final,before_avgpool,final,before_avgpool' --branch_list='1,1,2,2'




