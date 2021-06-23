### Key experiments to reproduce.

1. Ablation study: Role of modules.
    - Conv-4 on mini-ImageNet.
        + Baseline
            ```bash
            ./scripts/traintest-featcluster_minibatch_trial_delay.sh 0 FFFF-out_conv trial3
            ```
        + (+) Feature Clustering
            ```bash
            ./scripts/traintest-featcluster_minibatch_trial_delay.sh 1 TTTT-out_conv trial2
            ```
        + (+) Multi-Branch
            ```bash
            ./scripts/traintest-featcluster_minibatch_sideout_classifier_ybranch.sh 0 TTTT-out_conv-stage2-firstTconcat-train3shot-60epoch trial2 0
            ```
        + (+) Feature Augmentation
            ```bash
            python test_few_shot_ybranch.py --shot 1 --gpu 1 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load=./save/conv4featcluster-minibatch-sideout-classifier-ybranch-TTTT-out_conv-stage2-firstTconcat-train3shot-60epoch_trial2_2gpu/max-f-va.pth --method='cos,cos,cos,cos' --dist_func_list='none,none,none,none' --logits_coeff_list='20,20,20,20' --disable_out_feat_flatten --sideout --feat_source_list='final,before_avgpool,final,before_avgpool' --branch_list='1,1,2,2'
            ```
            ```bash
            python test_few_shot_ybranch.py --shot 5 --gpu 3 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load=./save/conv4featcluster-minibatch-sideout-classifier-ybranch-TTTT-out_conv-stage2-firstTconcat-train3shot-60epoch_trial2_2gpu/max-f-va.pth --method='cos,cos,cos,cos' --dist_func_list='none,none,none,none' --logits_coeff_list='20,20,20,20' --disable_out_feat_flatten --sideout --feat_source_list='final,before_avgpool,final,before_avgpool' --branch_list='1,1,2,2'
            ```

    - Res-12 on mini-ImageNet.
        + Baseline
            ```bash
            ./scripts/traintest-featcluster_minibatch_res12.sh 2 FFF-FFF-FFF-FFF
            ```
        + (+) Feature Clustering
            ```bash
            ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_trial.sh 0 TFF-FFF-TTT-FFF-out_conv-firstTconcat trial2_fix
            ```
        + (+) Multi-Branch
            ```bash
            ./scripts/test-featcluster_minibatch_res12_sideout_classifier_ybranch.sh 0 TFF-FFF-TTT-FFF-out_conv-stage2-firstTconcat-train3shot-60epoch_trial2_fix
            ```
        + (+) Feature Augmentation
            ```bash
            python test_few_shot_ybranch.py --shot 1 --gpu 2 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load=./save/res12featcluster-minibatch-sideout-classifier-ybranch-TFF-FFF-TTT-FFF-out_conv-stage2-firstTconcat-train3shot-60epoch_trial2_fix/max-f-va.pth --method='cos,cos,cos,cos' --dist_func_list='none,none,none,none' --logits_coeff_list='20,20,20,20' --disable_out_feat_flatten --sideout --feat_source_list='final,before_avgpool,final,before_avgpool' --branch_list='1,1,2,2'
            ```
            ```bash
            python test_few_shot_ybranch.py --shot 5 --gpu 2 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load=./save/res12featcluster-minibatch-sideout-classifier-ybranch-TFF-FFF-TTT-FFF-out_conv-stage2-firstTconcat-train3shot-60epoch_trial2_fix/max-f-va.pth --method='cos,cos,cos,cos' --dist_func_list='none,none,none,none' --logits_coeff_list='20,20,20,20' --disable_out_feat_flatten --sideout --feat_source_list='final,before_avgpool,final,before_avgpool' --branch_list='1,1,2,2'
            ```

        + (+) trainval
            TBA.
    
    - Conv-4 on CIFAR-FS.
        + Baseline.
            ```bash
            ./scripts/traintest-featcluster_minibatch_trial_cifar-fs.sh 0 FFFF trial2_fix 0
            ```
        + (+) Feature Clustering + Multi-Branch + Feature Augmentation.
            ```bash
            ./scripts/traintest-featcluster_minibatch_sideout_classifier_ybranch_cifar-fs.sh 2 TTTT-out_conv-stage2-firstTconcat-train3shot-60epoch trial2 2

            python test_few_shot_ybranch.py --shot 1 --gpu 1 --config ./configs/current_configs/test_few_shot_cifar-fs_general.yaml --load=./save/conv4featcluster-minibatch-sideout-classifier-ybranch-cifar-fs-TTTT-out_conv-stage2-firstTconcat-train3shot-60epoch_trial2/max-f-va.pth --method='cos,cos,cos,cos' --dist_func_list='none,none,none,none' --logits_coeff_list='20,20,20,20' --disable_out_feat_flatten --sideout --feat_source_list='final,before_avgpool,final,before_avgpool' --branch_list='1,1,2,2'

            python test_few_shot_ybranch.py --shot 5 --gpu 4 --config ./configs/current_configs/test_few_shot_cifar-fs_general.yaml --load=./save/conv4featcluster-minibatch-sideout-classifier-ybranch-cifar-fs-TTTT-out_conv-stage2-firstTconcat-train3shot-60epoch_trial2/max-f-va.pth --method='cos,cos,cos,cos' --dist_func_list='none,none,none,none' --logits_coeff_list='20,20,20,20' --disable_out_feat_flatten --sideout --feat_source_list='final,before_avgpool,final,before_avgpool' --branch_list='1,1,2,2'
            ```
    
    - Res-12 on CIFAR-FS.
        + Baseline.
            ```bash
            ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_cifar-fs.sh 1 FFF-FFF-FFF-FFF-out_conv-test_stop trial2 1;
            ```
        + (+) Feature Clustering + Multi-Branch + Feature Augmentation.
            ```bash
            ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_ybranch_cifar-fs.sh 0 TFF-FFF-TTT-FFF-out_conv-stage2-firstTconcat-train7shot-test_stop-cos trial2_fix 0

            python test_few_shot_ybranch.py --shot 1 --gpu 1 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load=./save/conv4featcluster-minibatch-sideout-classifier-ybranch-TTTT-out_conv-stage2-firstTconcat-train3shot-60epoch_trial2_2gpu/max-f-va.pth --method='cos,cos,cos,cos' --dist_func_list='none,none,none,none' --logits_coeff_list='20,20,20,20' --disable_out_feat_flatten --sideout --feat_source_list='final,before_avgpool,final,before_avgpool' --branch_list='1,1,2,2'

            python test_few_shot_ybranch.py --shot 5 --gpu 3 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load=./save/conv4featcluster-minibatch-sideout-classifier-ybranch-TTTT-out_conv-stage2-firstTconcat-train3shot-60epoch_trial2_2gpu/max-f-va.pth --method='cos,cos,cos,cos' --dist_func_list='none,none,none,none' --logits_coeff_list='20,20,20,20' --disable_out_feat_flatten --sideout --feat_source_list='final,before_avgpool,final,before_avgpool' --branch_list='1,1,2,2'
            ```
        + (+) trainval
            TBA.

2. Ablation study: #clusters.
    - Conv-4 on mini-ImageNet.
        + 32 clusters.
            ```bash
            ./scripts/traintest-featcluster_minibatch_trial_delay.sh 0 TTTT-out_conv-32clusters trial2;
            ```
        + 64 clusters.
            ```bash
            ./scripts/traintest-featcluster_minibatch_trial_delay.sh 0 TTTT-out_conv-64clusters trial2;
            ```
        + 128 clusters.
            ```bash
            ./scripts/traintest-featcluster_minibatch_trial_delay.sh 0 TTTT-out_conv-128clusters trial2;
            ```

3. Ablation study: Position of clustering module.
    - Conv-4 on mini-ImageNet: 
        + Layer 1.
            ```bash
            ./scripts/traintest-featcluster_minibatch_trial_delay.sh 7 TFFF-out_conv trial2;
            ```
        + Layer 1,2.
            ```bash
            ./scripts/traintest-featcluster_minibatch_trial_delay.sh 7 TTFF-out_conv trial2;
            ```
        + Layer 1,2,3.
            ```bash
            ./scripts/traintest-featcluster_minibatch_trial_delay.sh 7 TTTF-out_conv trial2;
            ```
        + Layer 1,2,3,4.
            ```bash
            ./scripts/traintest-featcluster_minibatch_trial_delay.sh 7 TTTT-out_conv trial2;
            ```
