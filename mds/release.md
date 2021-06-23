python train_classifier_sideout_classifier.py --gpu 1 --config ./configs/mini-imagenet/conv4-classifier-mini.yaml --tag trial2

python train_classifier_sideout_classifier_ybranch.py --gpu 0 --config ./configs/mini-imagenet/conv4-ybranch-mini.yaml --tag trial1

python train_classifier_sideout_classifier_ybranch.py --gpu 0 --config ./configs/mini-imagenet/conv4-ybranch-mini-new.yaml --tag trial1

python test_few_shot.py --shot 1 --gpu 0 --config ./configs/mini-imagenet/test_few_shot_mini_general.yaml --load_encoder=./save/res12-classifier-mini-new_trial1/max-va.pth


python test_few_shot_ybranch.py --shot 1 --gpu 2 --config ./configs/mini-imagenet/test_few_shot_mini_general.yaml --load=./save/conv4-ybranch-mini-new_trial1/max-f-va.pth --method='cos,cos,cos,cos' --dist_func_list='none,none,none,none' --logits_coeff_list='20,20,20,20' --disable_out_feat_flatten --sideout --feat_source_list='final,before_avgpool,final,before_avgpool' --branch_list='1,1,2,2'


python test_few_shot_ybranch.py --shot 5 --gpu 2 --config ./configs/mini-imagenet/test_few_shot_mini_general.yaml --load=./save/conv4featcluster-minibatch-sideout-classifier-ybranch-TTTT-self-uv_dist-FFTT-att-out_conv-stage2-firstTconcat-train3shot-60epoch-sine_pe-pre_normF-init_rp-8_head-param_reduced_trial1/max-f-va.pth --method='cos,cos,cos,cos' --dist_func_list='none,none,none,none' --logits_coeff_list='20,20,20,20' --disable_out_feat_flatten --sideout --feat_source_list='final,before_avgpool,final,before_avgpool' --branch_list='1,1,2,2'


python train_classifier_sideout_classifier_ybranch.py --gpu 3 --config ./configs/mini-imagenet/conv4-ybranch-mini-new.yaml --tag trial1

python train_classifier_sideout_classifier_ybranch.py --gpu 0,1 --config ./configs/mini-imagenet/res12-ybranch-param_reduced-mini-new.yaml --tag trial1


