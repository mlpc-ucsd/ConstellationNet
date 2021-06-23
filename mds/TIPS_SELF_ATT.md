python test_few_shot_ybranch.py --shot 1 --gpu 0 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load=./save/conv4featcluster-minibatch-sideout-classifier-ybranch-TTTT-selfatt-FFTT-out_conv-stage2-firstTconcat-train3shot-60epoch_trial5/max-f-va.pth --method='cos,cos,cos,cos' --dist_func_list='none,none,none,none' --logits_coeff_list='20,20,20,20' --disable_out_feat_flatten --sideout --feat_source_list='final,before_avgpool,final,before_avgpool' --branch_list='1,1,2,2'

python train_classifier_sideout_classifier_ybranch.py --gpu 4,5 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_ybranch_self_att/TFF-FFF-TTT-FFF-self-uv_dist-att-FFF-FFF-TTT-FFF-out_conv-stage2-firstTconcat-otherTconcat_regular-train3shot-60epoch-sine_pe-pre_normF-init_rp.yaml --tag trial1



python train_classifier_sideout_classifier_ybranch_cifar-fs.py --gpu 0,6,2,4 --config ./configs/current_configs/featcluster_minibatch_sideout_classifier_ybranch_cifar-fs_self_att/TTTT-self-uv_dist-FFTT-att-out_conv-stage2-firstTconcat-train3shot-60epoch-sine_pe-pre_normF-init_rp.yaml --tag trial1 --resume ./save/conv4featcluster-minibatch-sideout-classifier-ybranch-cifar-fs-TTTT-self-uv_dist-FFTT-att-out_conv-stage2-firstTconcat-train3shot-60epoch-sine_pe-pre_normF-init_rp_trial1/epoch-last.pth

python train_classifier_sideout_classifier_temp.py --gpu 0 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_self_att/TFF-FFF-TTT-FFF-out_conv-firstTconcat-sine_pe.yaml --tag trial1

python train_classifier_sideout_classifier_temp.py --gpu 0,2,4,6 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_self_att/TFF-FFF-TTT-FFF-out_conv-firstTconcat-uv_dist_att-no_pe-pre_normT-init_rp.yaml --tag trial_no_sideout

python train_classifier_sideout_classifier_temp.py --gpu 0,1,2,3 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_self_att/TFF-FFF-TTT-FFF-out_conv-firstTconcat-no_pe-pre_normF.yaml --tag trial1

python train_classifier_sideout_classifier_temp.py --gpu 0,2,4,6 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_self_att/TFF-FFF-TTT-FFF-out_conv-firstTconcat-no_pe-pre_normT-init_rp.yaml --tag trial1


python train_classifier_sideout_classifier_temp.py --gpu 0,1,2,3 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_self_att/TFF-FFF-TTT-FFF-out_conv-firstTconcat-no_pe-pre_normT.yaml --tag trial1

python train_classifier_sideout_classifier_temp.py --gpu 0,1,2,3 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_self_att/TFF-FFF-TTT-FFF-out_conv-firstTconcat-no_pe-pre_normT-init_rdp.yaml --tag trial1

python train_classifier_sideout_classifier_temp.py --gpu 0,1,2,3 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_self_att/TFF-FFF-TTT-FFF-out_conv-firstTconcat-no_pe-pre_normT-init_rdp.yaml --tag trial2


python train_classifier_sideout_classifier_temp.py --gpu 0,1,2,3 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_self_att/T336FF-FFF-TTT84-FFF-out_conv-firstTconcat-no_pe-pre_normT-init_rdp.yaml --tag trial1

python train_classifier_sideout_classifier_temp.py --gpu 0,1,2,3 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_self_att/T128FF-FFF-T64T64T32-FFF-out_conv-firstTconcat-no_pe-pre_normT-init_rdp.yaml --tag trial1

python train_classifier_sideout_classifier_temp.py --gpu 0,1,2,3 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_self_att/T128FF-FFF-T64T64T64-FFF-out_conv-firstTconcat-no_pe-pre_normT-init_rdp.yaml --tag trial1

python train_classifier_sideout_classifier_temp.py --gpu 0,1,2,3 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_self_att/TFF-FFF-TTT-FFF-out_conv-firstTconcat-no_pe-pre_normT-init_meangaussian.yaml --tag trial1

python train_classifier_sideout_classifier_temp.py --gpu 0,1,2,3 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_self_att/TFF-FFF-TTT-FFF-out_conv-firstTconcat-sine_pe-pre_normT.yaml --tag trial1

python train_classifier_sideout_classifier_temp.py --gpu 0,1,2,3 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_self_att/TFF-FFF-TTT-FFF-out_conv-firstTconcat-no_pe-pre_normT.yaml --tag trial1

python train_classifier_sideout_classifier_temp.py --gpu 0 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_self_att/TFF-FFF-TTT-FFF-out_conv-firstTconcat.yaml --tag trial1

python train_classifier_sideout_classifier_temp.py --gpu 0,1 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_self_att/TFF-FFF-TTT-FFF-out_conv-firstTconcat.yaml --tag trial5

python train_classifier_sideout_classifier_temp.py --gpu 0,1,2,3 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_self_att/TFF-FFF-TTT-FFF-out_conv-firstTconcat.yaml --tag trial3 --resume ./save/res12featcluster-minibatch-sideout-classifier-TFF-FFF-TTT-FFF-self-att-FFF-FFF-TTT-FFF-out_conv-firstTconcat_trial3/epoch-last.pth

python train_classifier_sideout_classifier_temp.py --gpu 0,1 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_self_att/TFF-FFF-TTT-FFF-out_conv-firstTconcat.yaml --tag trial4 --resume ./save/res12featcluster-minibatch-sideout-classifier-TFF-FFF-TTT-FFF-self-att-FFF-FFF-TTT-FFF-out_conv-firstTconcat_trial4/epoch-last.pth

python train_classifier_sideout_classifier_temp.py --gpu 1 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier/TFF-FFF-TTT-FFF-out_conv-firstTconcat.yaml --tag trial1

python test_few_shot.py --shot 5 --gpu 0,6,2,4 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/res12featcluster-minibatch-sideout-classifier-TFF-FFF-TTT-FFF-self-uv_dist-att-FFF-FFF-TTT-FFF-out_conv-firstTconcat-sine_pe-pre_normT-init_rp_trial2/epoch-95.pth

python test_few_shot.py --shot 1 --gpu 0,1,4,5 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/res12featcluster-minibatch-sideout-classifier-TFF-FFF-TTT-FFF-self-uv_dist-att-FFF-FFF-TTT-FFF-out_conv-firstTconcat-sine_pe-pre_normT-init_rp_trial2/epoch-95.pth

python test_few_shot.py --shot 5 --gpu 0,1,4,5 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/res12featcluster-minibatch-sideout-classifier-TFF-FFF-TTT-FFF-self-uv_dist-att-FFF-FFF-TTT-FFF-out_conv-firstTconcat-sine_pe-pre_normT-init_rp-trial2/epoch-95.pth

python test_few_shot.py --shot 1 --gpu 0,6,2,4 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/res12featcluster-minibatch-sideout-classifier-TFF-FFF-TTT-FFF-out_conv-firstTconcat-no_att-init_rp_trial1/max-va.pth

python test_few_shot.py --shot 5 --gpu 0,6,2,4 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/res12featcluster-minibatch-sideout-classifier-TFF-FFF-TTT-FFF-out_conv-firstTconcat-no_att-init_rp_trial1/max-va.pth

python test_few_shot.py --shot 1 --gpu 0,6,2,4 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/res12featcluster-minibatch-sideout-classifier-TFF-FFF-TTT-FFF-self-uv_dist-att-FFF-FFF-TTT-FFF-out_conv-firstTconcat-no_pe-pre_normT-init_rp_trial_no_sideout/max-va.pth

python test_few_shot.py --shot 5 --gpu 0,6,2,4 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/res12featcluster-minibatch-sideout-classifier-TFF-FFF-TTT-FFF-self-uv_dist-att-FFF-FFF-TTT-FFF-out_conv-firstTconcat-no_pe-pre_normT-init_rp_trial_no_sideout/max-va.pth

python test_few_shot.py --shot 1 --gpu 0,6,2,4 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/res12featcluster-minibatch-sideout-classifier-TFF-FFF-TTT-FFF-self-att-FFF-FFF-TTT-FFF-out_conv-firstTconcat-no_pe-pre_normT-init_rp_trial1/max-va.pth

python test_few_shot.py --shot 5 --gpu 0,6,2,4 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/res12featcluster-minibatch-sideout-classifier-TFF-FFF-TTT-FFF-self-att-FFF-FFF-TTT-FFF-out_conv-firstTconcat-no_pe-pre_normT-init_rp_trial1/max-va.pth

python test_few_shot.py --shot 1 --gpu 0,6,2,4 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/res12featcluster-minibatch-sideout-classifier-TFF-FFF-TTT-FFF-self-att-FFF-FFF-TTT-FFF-out_conv-firstTconcat-no_pe-pre_normT-init_rdp_trial2_new/max-va.pth

python test_few_shot.py --shot 5 --gpu 0,6,2,4 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/res12featcluster-minibatch-sideout-classifier-TFF-FFF-TTT-FFF-self-att-FFF-FFF-TTT-FFF-out_conv-firstTconcat-no_pe-pre_normT-init_rdp_trial2_new/max-va.pth

python test_few_shot.py --shot 1 --gpu 0,6,2,4 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/res12featcluster-minibatch-sideout-classifier-TFF-FFF-TTT-FFF-self-att-FFF-FFF-TTT-FFF-out_conv-firstTconcat-no_pe-pre_normT-init_rdp_trial1_new/max-va.pth

python test_few_shot.py --shot 5 --gpu 0,6,2,4 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/res12featcluster-minibatch-sideout-classifier-TFF-FFF-TTT-FFF-self-att-FFF-FFF-TTT-FFF-out_conv-firstTconcat-no_pe-pre_normT-init_rdp_trial1_new/max-va.pth


python test_few_shot.py --shot 1 --gpu 0,1,2,3 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/res12featcluster-minibatch-sideout-classifier-TFF-FFF-TTT-FFF-self-att-FFF-FFF-TTT-FFF-out_conv-firstTconcat-no_pe-pre_normT-init_rdp_trial1/max-va.pth

python test_few_shot.py --shot 1 --gpu 0,1,2,3 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/res12featcluster-minibatch-sideout-classifier-T128FF-FFF-T64T64T64-FFF-self-att-FFF-FFF-TTT-FFF-out_conv-firstTconcat-no_pe-pre_normT-init_rdp_trial1/max-va.pth

python test_few_shot.py --shot 1 --gpu 0,1,2,3 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/res12featcluster-minibatch-sideout-classifier-TFF-FFF-TTT-FFF-self-att-FFF-FFF-TTT-FFF-out_conv-firstTconcat-no_pe-pre_normF_trial1/max-va.pth

python test_few_shot.py --shot 5 --gpu 0,1,2,3 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/res12featcluster-minibatch-sideout-classifier-TFF-FFF-TTT-FFF-self-att-FFF-FFF-TTT-FFF-out_conv-firstTconcat-no_pe-pre_normF_trial1/max-va.pth

python test_few_shot.py --shot 1 --gpu 0,1,2,3 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/res12featcluster-minibatch-sideout-classifier-TFF-FFF-TTT-FFF-self-att-FFF-FFF-TTT-FFF-out_conv-firstTconcat-sine_pe_trial1/max-va.pth

python test_few_shot.py --shot 1 --gpu 0,1,2,3 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/res12featcluster-minibatch-sideout-classifier-TFF-FFF-TTT-FFF-self-att-FFF-FFF-TTT-FFF-out_conv-firstTconcat-sine_pe-pre_normT_trial3/max-va.pth

python test_few_shot.py --shot 1 --gpu 0,1,2,3 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/res12featcluster-minibatch-sideout-classifier-TFF-FFF-TTT-FFF-self-att-FFF-FFF-TTT-FFF-out_conv-firstTconcat-sine_pe-pre_normF_trial3/max-va.pth

python test_few_shot.py --shot 1 --gpu 0,1,2,3 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/res12featcluster-minibatch-sideout-classifier-TFF-FFF-TTT-FFF-self-att-FFF-FFF-TTT-FFF-out_conv-firstTconcat_trial3/max-va.pth

python test_few_shot.py --shot 1 --gpu 0 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/res12featcluster-minibatch-sideout-classifier-TFF-FFF-TTT-FFF-self-att-FFF-FFF-TTT-FFF-out_conv-firstTconcat_trial1/max-va.pth

python test_few_shot.py --shot 1 --gpu 1 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/res12featcluster-minibatch-sideout-classifier-TFF-FFF-TTT-FFF-out_conv-firstTconcat_trial1/max-va.pth

python test_few_shot_ybranch.py --shot 1 --gpu 0,2,4,6 --config ./configs/current_configs/test_few_shot_cifar-fs_general.yaml --load=./save/conv4featcluster-minibatch-sideout-classifier-ybranch-cifar-fs-TTTT-self-uv_dist-FFTT-att-out_conv-stage2-firstTconcat-train3shot-60epoch-sine_pe-pre_normF-init_rp_trial1/max-f-va.pth --method='cos,cos' --dist_func_list='none,none' --logits_coeff_list='20,20' --disable_out_feat_flatten --sideout --feat_source_list='final,final' --branch_list='1,2'

python test_few_shot_ybranch.py --shot 1 --gpu 0,2,4,6 --config ./configs/current_configs/test_few_shot_cifar-fs_general.yaml --load=./save/conv4featcluster-minibatch-sideout-classifier-ybranch-cifar-fs-TTTT-self-uv_dist-FFTT-att-out_conv-stage2-firstTconcat-train3shot-60epoch-sine_pe-pre_normF-init_rp_trial1/max-f-va.pth --method='cos,cos,cos,cos' --dist_func_list='none,none,none,none' --logits_coeff_list='20,20,20,20' --disable_out_feat_flatten --sideout --feat_source_list='final,before_avgpool,final,before_avgpool' --branch_list='1,1,2,2'

python test_few_shot_ybranch.py --shot 5 --gpu 0,2,4,6 --config ./configs/current_configs/test_few_shot_cifar-fs_general.yaml --load=./save/conv4featcluster-minibatch-sideout-classifier-ybranch-cifar-fs-TTTT-self-uv_dist-FFTT-att-out_conv-stage2-firstTconcat-train3shot-60epoch-sine_pe-pre_normF-init_rp_trial2/max-f-va.pth --method='cos,cos,cos,cos' --dist_func_list='none,none,none,none' --logits_coeff_list='20,20,20,20' --disable_out_feat_flatten --sideout --feat_source_list='final,before_avgpool,final,before_avgpool' --branch_list='1,1,2,2'

python test_few_shot_ybranch.py --shot 1 --gpu 0,2,4,6 --config ./configs/current_configs/test_few_shot_cifar-fs_general.yaml --load=./save/conv4featcluster-minibatch-sideout-classifier-ybranch-cifar-fs-TTTT-self-uv_dist-FFTT-att-out_conv-stage2-firstTconcat-train3shot-60epoch-sine_pe-pre_normF-init_rp_trial2/max-f-va.pth --method='cos,cos,cos,cos' --dist_func_list='none,none,none,none' --logits_coeff_list='20,20,20,20' --disable_out_feat_flatten --sideout --feat_source_list='final,before_avgpool,final,before_avgpool' --branch_list='1,1,2,2'

python test_few_shot_ybranch.py --shot 1 --gpu 2 --config ./configs/current_configs/test_few_shot_cifar-fs_general.yaml --load=./save/conv4featcluster-minibatch-sideout-classifier-ybranch-cifar-fs-TTTT-self-uv_dist-FFTT-att-out_conv-stage2-firstTconcat-train3shot-60epoch-no_pe-pre_normF-init_rp-1_head_trial2/max-f-va.pth --method='cos,cos,cos,cos' --dist_func_list='none,none,none,none' --logits_coeff_list='20,20,20,20' --disable_out_feat_flatten --sideout --feat_source_list='final,before_avgpool,final,before_avgpool' --branch_list='1,1,2,2'


python test_few_shot_ybranch.py --shot 1 --gpu 2 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load=./save/conv4featcluster-minibatch-sideout-classifier-ybranch-TTTT-self-uv_dist-FFTT-att-out_conv-stage2-firstTconcat-train3shot-60epoch-no_pe-pre_normF-init_rp-1_head_trial1/max-f-va.pth --method='cos,cos,cos,cos' --dist_func_list='none,none,none,none' --logits_coeff_list='20,20,20,20' --disable_out_feat_flatten --sideout --feat_source_list='final,before_avgpool,final,before_avgpool' --branch_list='1,1,2,2'

python test_few_shot_ybranch.py --shot 1 --gpu 4 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load=./save/conv4featcluster-minibatch-sideout-classifier-ybranch-TTTT-out_conv-stage2-firstTconcat-train3shot-60epoch-no_att-init_rp_trial2/max-f-va.pth --method='cos,cos,cos,cos' --dist_func_list='none,none,none,none' --logits_coeff_list='20,20,20,20' --disable_out_feat_flatten --sideout --feat_source_list='final,before_avgpool,final,before_avgpool' --branch_list='1,1,2,2'

python test_few_shot_ybranch.py --shot 5 --gpu 2 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load=./save/conv4featcluster-minibatch-sideout-classifier-ybranch-TTTT-self-uv_dist-FFTT-att-out_conv-stage2-firstTconcat-train3shot-60epoch-no_pe-pre_normF-init_rp-1_head_trial1/max-f-va.pth --method='cos,cos,cos,cos' --dist_func_list='none,none,none,none' --logits_coeff_list='20,20,20,20' --disable_out_feat_flatten --sideout --feat_source_list='final,before_avgpool,final,before_avgpool' --branch_list='1,1,2,2'

python test_few_shot_ybranch.py --shot 5 --gpu 4 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load=./save/conv4featcluster-minibatch-sideout-classifier-ybranch-TTTT-out_conv-stage2-firstTconcat-train3shot-60epoch-no_att-init_rp_trial2/max-f-va.pth --method='cos,cos,cos,cos' --dist_func_list='none,none,none,none' --logits_coeff_list='20,20,20,20' --disable_out_feat_flatten --sideout --feat_source_list='final,before_avgpool,final,before_avgpool' --branch_list='1,1,2,2'

python test_few_shot_ybranch.py --shot 1 --gpu 4 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load=./save/res12featcluster-minibatch-sideout-classifier-ybranch-TFF-FFF-TTT-FFF-self-uv_dist-att-FFF-FFF-TTT-FFF-out_conv-stage2-firstTconcat-otherTconcat_regular-train3shot-60epoch-sine_pe-pre_normF-init_rp_trial1/max-f-va.pth --method='cos,cos,cos,cos' --dist_func_list='none,none,none,none' --logits_coeff_list='20,20,20,20' --disable_out_feat_flatten --sideout --feat_source_list='final,before_avgpool,final,before_avgpool' --branch_list='1,1,2,2'

python test_few_shot_ybranch.py --shot 5 --gpu 4 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load=./save/res12featcluster-minibatch-sideout-classifier-ybranch-TFF-FFF-TTT-FFF-self-uv_dist-att-FFF-FFF-TTT-FFF-out_conv-stage2-firstTconcat-otherTconcat_regular-train3shot-60epoch-sine_pe-pre_normF-init_rp_trial1/max-f-va.pth --method='cos,cos,cos,cos' --dist_func_list='none,none,none,none' --logits_coeff_list='20,20,20,20' --disable_out_feat_flatten --sideout --feat_source_list='final,before_avgpool,final,before_avgpool' --branch_list='1,1,2,2'
