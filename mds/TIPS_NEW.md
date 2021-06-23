1. Evaluation tricks.
    - Flatten + Avgpool
        ```bash
        python test_few_shot_ybranch.py --shot 1 --gpu 6 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load=./save/res12featcluster-minibatch-sideout-classifier-ybranch-TFF-FFF-TTT-FFF-out_conv-stage2-firstTconcat-train3shot-60epoch_trial2/max-f-va.pth --method='cos,cos' --dist_func_list='none,none' --logits_coeff_list='20,20' --disable_out_feat_flatten --sideout --feat_source_list='final,before_avgpool' --branch_list='1,1'

        python test_few_shot_ybranch.py --shot 1 --gpu 7 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load=./save/res12featcluster-minibatch-sideout-classifier-ybranch-TFF-FFF-TTT-FFF-out_conv-stage2-firstTconcat-train3shot-60epoch_trial2/max-f-va.pth --method='cos,cos' --dist_func_list='none,none' --logits_coeff_list='20,20' --disable_out_feat_flatten --sideout --feat_source_list='final,before_avgpool' --branch_list='2,2'

        python test_few_shot_ybranch.py --shot 1 --gpu 7 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load=./save/res12featcluster-minibatch-sideout-classifier-ybranch-TFF-FFF-TTT-FFF-out_conv-stage2-firstTconcat-train3shot-60epoch_trial2/max-f-va.pth --method='cos,cos,cos,cos' --dist_func_list='none,none,none,none' --logits_coeff_list='20,20,20,20' --disable_out_feat_flatten --sideout --feat_source_list='final,before_avgpool,final,before_avgpool' --branch_list='1,1,2,2'




        python test_few_shot_ybranch.py --shot 1 --gpu 4 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load=./save/res12featcluster-minibatch-sideout-classifier-ybranch-TFF-FFF-TTT-FFF-out_conv-stage2-firstTconcat-train3shot-60epoch_trial3/max-f-va.pth --method='cos,cos' --dist_func_list='none,none' --logits_coeff_list='20,20' --disable_out_feat_flatten --sideout --feat_source_list='final,before_avgpool' --branch_list='1,1';

        python test_few_shot_ybranch.py --shot 1 --gpu 4 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load=./save/res12featcluster-minibatch-sideout-classifier-ybranch-TFF-FFF-TTT-FFF-out_conv-stage2-firstTconcat-train3shot-60epoch_trial3/max-f-va.pth --method='cos,cos' --dist_func_list='none,none' --logits_coeff_list='20,20' --disable_out_feat_flatten --sideout --feat_source_list='final,before_avgpool' --branch_list='2,2'

        python test_few_shot_ybranch.py --shot 1 --gpu 0 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load=./save/res12featcluster-minibatch-sideout-classifier-ybranch-TFF-FFF-TTT-FFF-out_conv-stage2-firstTconcat-train3shot-60epoch_trial3/max-f-va.pth --method='cos,cos,cos,cos' --dist_func_list='none,none,none,none' --logits_coeff_list='20,20,20,20' --disable_out_feat_flatten --sideout --feat_source_list='final,before_avgpool,final,before_avgpool' --branch_list='1,1,2,2'


        # moss105
        python test_few_shot_ybranch.py --shot 5 --gpu 0 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load=./save/res12featcluster-minibatch-sideout-classifier-ybranch-TFF-FFF-TTT-FFF-out_conv-stage2-firstTconcat-train3shot-60epoch_trial2/max-f-va.pth --method='cos,cos' --dist_func_list='none,none' --logits_coeff_list='20,20' --disable_out_feat_flatten --sideout --feat_source_list='final,before_avgpool' --branch_list='1,1'

        python test_few_shot_ybranch.py --shot 5 --gpu 1 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load=./save/res12featcluster-minibatch-sideout-classifier-ybranch-TFF-FFF-TTT-FFF-out_conv-stage2-firstTconcat-train3shot-60epoch_trial3/max-f-va.pth --method='cos,cos' --dist_func_list='none,none' --logits_coeff_list='20,20' --disable_out_feat_flatten --sideout --feat_source_list='final,before_avgpool' --branch_list='1,1'
        ```

    - Prof's suggestion - Add side branch representation to the last.
        ```bash
        python test_few_shot_ybranch.py --shot 1 --gpu 6 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load=./save/res12featcluster-minibatch-sideout-classifier-ybranch-TFF-FFF-TTT-FFF-out_conv-stage2-firstTconcat-train3shot-60epoch_trial2/max-f-va.pth --method='cos,cos' --dist_func_list='none,none' --logits_coeff_list='20,20' --disable_out_feat_flatten --sideout --feat_source_list='final,branch1.0.mergeblock1.featcluster.avgpool' --branch_list='1,1'  # Make the performance worse.
        # ~59%  it is worse.
        ```
2. Learning rate schedule adjustment.
    - Train and evaluate.
        ```bash
        # Baseline: moss103 0-6/2-4/5-7/8-9
        
        # 2-4
        python train_classifier_sideout_classifier_ybranch.py --gpu 2,4 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_ybranch/FFF-FFF-FFF-FFF-out_conv-stage0-train3shot-LR-lee-modified.yaml --tag=trial2;

        python train_classifier_sideout_classifier_ybranch.py --gpu 2,4 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_ybranch/FFF-FFF-FFF-FFF-out_conv-stage0-train3shot-LR-lee-modified.yaml --tag=trial3


        # 0-6
        python train_classifier_sideout_classifier_ybranch.py --gpu 0,6 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_ybranch/FFF-FFF-FFF-FFF-out_conv-stage0-train3shot-LR-long-modified.yaml --tag=trial2;

        python train_classifier_sideout_classifier_ybranch.py --gpu 0,6 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_ybranch/FFF-FFF-FFF-FFF-out_conv-stage0-train3shot-LR-long-modified.yaml --tag=trial3

        # 5-7
        python train_classifier_sideout_classifier_ybranch.py --gpu 5,7 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_ybranch/FFF-FFF-FFF-FFF-out_conv-stage0-train3shot-LR-long.yaml --tag=trial2;
        
        python train_classifier_sideout_classifier_ybranch.py --gpu 5,7 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_ybranch/FFF-FFF-FFF-FFF-out_conv-stage0-train3shot-LR-long.yaml --tag=trial3


        # 8-9
        python train_classifier_sideout_classifier_ybranch.py --gpu 8,9 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_ybranch/FFF-FFF-FFF-FFF-out_conv-stage0-train3shot-LR-lee-modified.yaml --tag=trial4;

        python train_classifier_sideout_classifier_ybranch.py --gpu 8,9 --config ./configs/current_configs/featcluster_
        minibatch_res12_sideout_classifier_ybranch/FFF-FFF-FFF-FFF-out_conv-stage0-train3shot-LR-long-modified.yaml --tag=trial4

        # 1-3
        python train_classifier_sideout_classifier_ybranch.py --gpu 1,3 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_ybranch/FFF-FFF-FFF-FFF-out_conv-stage0-train3shot-LR-long.yaml --tag=trial4


        # Best model.
        # moss105 0-1/4-5/6-7
        python train_classifier_sideout_classifier_ybranch.py --gpu 0,1 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_ybranch/TFF-FFF-TTT-FFF-out_conv-stage2-firstTconcat-train3shot-LR-long-modified.yaml --tag=trial2;
        python train_classifier_sideout_classifier_ybranch.py --gpu 0,1 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_ybranch/TFF-FFF-TTT-FFF-out_conv-stage2-firstTconcat-train3shot-LR-long-modified.yaml --tag=trial3;
        python train_classifier_sideout_classifier_ybranch.py --gpu 0,1 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_ybranch/TFF-FFF-TTT-FFF-out_conv-stage2-firstTconcat-train3shot-LR-long-modified.yaml --tag=trial4

        python train_classifier_sideout_classifier_ybranch.py --gpu 4,5 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_ybranch/TFF-FFF-TTT-FFF-out_conv-stage2-firstTconcat-train3shot-LR-long.yaml --tag=trial2;
        python train_classifier_sideout_classifier_ybranch.py --gpu 4,5 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_ybranch/TFF-FFF-TTT-FFF-out_conv-stage2-firstTconcat-train3shot-LR-long.yaml --tag=trial3;
        python train_classifier_sideout_classifier_ybranch.py --gpu 4,5 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_ybranch/TFF-FFF-TTT-FFF-out_conv-stage2-firstTconcat-train3shot-LR-long.yaml --tag=trial4

        python train_classifier_sideout_classifier_ybranch.py --gpu 6,7 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_ybranch/TFF-FFF-TTT-FFF-out_conv-stage2-firstTconcat-train3shot-LR-lee-modified.yaml --tag=trial2;
        python train_classifier_sideout_classifier_ybranch.py --gpu 6,7 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_ybranch/TFF-FFF-TTT-FFF-out_conv-stage2-firstTconcat-train3shot-LR-lee-modified.yaml --tag=trial3;
        python train_classifier_sideout_classifier_ybranch.py --gpu 6,7 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_ybranch/TFF-FFF-TTT-FFF-out_conv-stage2-firstTconcat-train3shot-LR-lee-modified.yaml --tag=trial4

        # Evaluation.
        ./scripts/test-featcluster_minibatch_res12_sideout_classifier_ybranch.sh 2 FFF-FFF-FFF-FFF-out_conv-stage0-train3shot-LR-lee-modified_trial2
        ./scripts/test-featcluster_minibatch_res12_sideout_classifier_ybranch.sh 3 FFF-FFF-FFF-FFF-out_conv-stage0-train3shot-LR-lee-modified_trial3
        ./scripts/test-featcluster_minibatch_res12_sideout_classifier_ybranch.sh 4 FFF-FFF-FFF-FFF-out_conv-stage0-train3shot-LR-lee-modified_trial4
        ./scripts/test-featcluster_minibatch_res12_sideout_classifier_ybranch.sh 5 FFF-FFF-FFF-FFF-out_conv-stage0-train3shot-LR-long-modified_trial2
        ./scripts/test-featcluster_minibatch_res12_sideout_classifier_ybranch.sh 6 FFF-FFF-FFF-FFF-out_conv-stage0-train3shot-LR-long-modified_trial3
        ./scripts/test-featcluster_minibatch_res12_sideout_classifier_ybranch.sh 7 FFF-FFF-FFF-FFF-out_conv-stage0-train3shot-LR-long-modified_trial4
        ./scripts/test-featcluster_minibatch_res12_sideout_classifier_ybranch.sh 8 FFF-FFF-FFF-FFF-out_conv-stage0-train3shot-LR-long_trial2
        ./scripts/test-featcluster_minibatch_res12_sideout_classifier_ybranch.sh 9 FFF-FFF-FFF-FFF-out_conv-stage0-train3shot-LR-long_trial3;
        ./scripts/test-featcluster_minibatch_res12_sideout_classifier_ybranch.sh 9 FFF-FFF-FFF-FFF-out_conv-stage0-train3shot-LR-long_trial4

        ./scripts/test-featcluster_minibatch_res12_sideout_classifier_ybranch.sh 2 TFF-FFF-TTT-FFF-out_conv-stage2-firstTconcat-train3shot-LR-lee-modified_trial2
        ./scripts/test-featcluster_minibatch_res12_sideout_classifier_ybranch.sh 3 TFF-FFF-TTT-FFF-out_conv-stage2-firstTconcat-train3shot-LR-lee-modified_trial3;
        ./scripts/test-featcluster_minibatch_res12_sideout_classifier_ybranch.sh 3 TFF-FFF-TTT-FFF-out_conv-stage2-firstTconcat-train3shot-LR-lee-modified_trial4
        ./scripts/test-featcluster_minibatch_res12_sideout_classifier_ybranch.sh 6 TFF-FFF-TTT-FFF-out_conv-stage2-firstTconcat-train3shot-LR-long-modified_trial2;
        ./scripts/test-featcluster_minibatch_res12_sideout_classifier_ybranch.sh 6 TFF-FFF-TTT-FFF-out_conv-stage2-firstTconcat-train3shot-LR-long-modified_trial3;
        ./scripts/test-featcluster_minibatch_res12_sideout_classifier_ybranch.sh 6 TFF-FFF-TTT-FFF-out_conv-stage2-firstTconcat-train3shot-LR-long-modified_trial4
        ./scripts/test-featcluster_minibatch_res12_sideout_classifier_ybranch.sh 7 TFF-FFF-TTT-FFF-out_conv-stage2-firstTconcat-train3shot-LR-long_trial2;
        ./scripts/test-featcluster_minibatch_res12_sideout_classifier_ybranch.sh 7 TFF-FFF-TTT-FFF-out_conv-stage2-firstTconcat-train3shot-LR-long_trial3;
        ./scripts/test-featcluster_minibatch_res12_sideout_classifier_ybranch.sh 7 TFF-FFF-TTT-FFF-out_conv-stage2-firstTconcat-train3shot-LR-long_trial4
        ```

3. Tiered-imagenet.
    Debug:
    ```bash
    # moss103 0-6/2-4/5-7/8-9
    python train_classifier_sideout_classifier_temp_tiered.py --gpu 0,6,2,4 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_tiered/FFF-FFF-FFF-FFF-out_conv-test_stop.yaml --tag=trial2;
    python train_classifier_sideout_classifier_temp_tiered.py --gpu 0,6,2,4 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_tiered/FFF-FFF-FFF-FFF-out_conv-test_stop.yaml --tag=trial3

    python train_classifier_sideout_classifier_temp_tiered.py --gpu 5,7,8,9 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_tiered/TFF-FFF-TTT-FFF-out_conv-firstTconcat-test_stop.yaml --tag=trial2;
    python train_classifier_sideout_classifier_temp_tiered.py --gpu 5,7,8,9 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_tiered/TFF-FFF-TTT-FFF-out_conv-firstTconcat-test_stop.yaml --tag=trial3


    # Evaluation:
    ./scripts/test-featcluster_minibatch_res12_sideout_classifier_tiered.sh 0 FFF-FFF-FFF-FFF-out_conv-test_stop trial2 45
    ./scripts/test-featcluster_minibatch_res12_sideout_classifier_tiered.sh 1 FFF-FFF-FFF-FFF-out_conv-test_stop trial3 45
    ./scripts/test-featcluster_minibatch_res12_sideout_classifier_tiered.sh 2 TFF-FFF-TTT-FFF-out_conv-firstTconcat-test_stop trial2 45
    ./scripts/test-featcluster_minibatch_res12_sideout_classifier_tiered.sh 3 TFF-FFF-TTT-FFF-out_conv-firstTconcat-test_stop trial3 45


    # moss101 1-3/2-4/5-7/8-9
    python train_classifier_sideout_classifier_ybranch_tiered.py --gpu 1,3,2,4 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_ybranch_tiered/FFF-FFF-FFF-FFF-out_conv-stage1-train3shot-test_stop.yaml --tag=trial2;
    python train_classifier_sideout_classifier_ybranch_tiered.py --gpu 1,3,2,4 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_ybranch_tiered/FFF-FFF-FFF-FFF-out_conv-stage1-train3shot-test_stop.yaml --tag=trial3

    # moss105 0-1/4-5/6-7
    python train_classifier_sideout_classifier_ybranch_tiered.py --gpu 0,1,2,3 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_ybranch_tiered/FFF-FFF-FFF-FFF-out_conv-stage2-train3shot-test_stop.yaml --tag=trial2;
    python train_classifier_sideout_classifier_ybranch_tiered.py --gpu 0,1,2,3 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_ybranch_tiered/FFF-FFF-FFF-FFF-out_conv-stage2-train3shot-test_stop.yaml --tag=trial3;

    # moss101 1-3/2-4/5-7/8-9
    python train_classifier_sideout_classifier_ybranch_tiered.py --gpu 5,7,8,9 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_ybranch_tiered/FFF-FFF-FFF-FFF-out_conv-stage3-train3shot-test_stop.yaml --tag=trial2;
    python train_classifier_sideout_classifier_ybranch_tiered.py --gpu 5,7,8,9 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_ybranch_tiered/FFF-FFF-FFF-FFF-out_conv-stage3-train3shot-test_stop.yaml --tag=trial3
    

    # moss102 2-4/5-7/8-9
    python train_classifier_sideout_classifier_ybranch_tiered.py --gpu 5,7,8,9 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_ybranch_tiered/FFF-FFF-FFF-FFF-out_conv-stage2-train3shot-test_stop-meta_loss_coeff0.5.yaml --tag=trial2;
    python train_classifier_sideout_classifier_ybranch_tiered.py --gpu 5,7,8,9 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_ybranch_tiered/FFF-FFF-FFF-FFF-out_conv-stage2-train3shot-test_stop-meta_loss_coeff0.5.yaml --tag=trial3

    # moss105 0-1/4-5/6-7
    python train_classifier_sideout_classifier_ybranch_tiered.py --gpu 4,5,6,7 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_ybranch_tiered/TFF-FFF-TTT-FFF-out_conv-stage2-firstTconcat-train3shot-test_stop.yaml --tag=trial2;
    python train_classifier_sideout_classifier_ybranch_tiered.py --gpu 4,5,6,7 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_ybranch_tiered/TFF-FFF-TTT-FFF-out_conv-stage2-firstTconcat-train3shot-test_stop.yaml --tag=trial3

    # moss105 0-1/4-5/6-7
    python train_classifier_sideout_classifier_ybranch_tiered.py --gpu 0,1,2,3 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_ybranch_tiered/FFF-FFF-FFF-FFF-out_conv-stage0-train3shot-test_stop.yaml --tag=trial2;
    python train_classifier_sideout_classifier_ybranch_tiered.py --gpu 0,1,2,3 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_ybranch_tiered/FFF-FFF-FFF-FFF-out_conv-stage0-train3shot-test_stop.yaml --tag=trial3


    # Evaluation.
    # moss105 4567
    ./scripts/test-featcluster_minibatch_res12_sideout_classifier_ybranch_tiered.sh 4 FFF-FFF-FFF-FFF-out_conv-stage1-train3shot-test_stop_trial2
    ./scripts/test-featcluster_minibatch_res12_sideout_classifier_ybranch_tiered.sh 5 FFF-FFF-FFF-FFF-out_conv-stage2-train3shot-test_stop_trial2
    ./scripts/test-featcluster_minibatch_res12_sideout_classifier_ybranch_tiered.sh 6 FFF-FFF-FFF-FFF-out_conv-stage3-train3shot-test_stop_trial2
    ./scripts/test-featcluster_minibatch_res12_sideout_classifier_ybranch_tiered.sh 7 FFF-FFF-FFF-FFF-out_conv-stage2-train3shot-test_stop-meta_loss_coeff0.5._trial2
    
    # moss106
    ./scripts/test-featcluster_minibatch_res12_sideout_classifier_ybranch_tiered.sh 0 TFF-FFF-TTT-FFF-out_conv-stage2-firstTconcat-train3shot-test_stop_trial2

    #Debug:
    python test_few_shot_ybranch.py --shot 1 --gpu 1 --config ./configs/current_configs/test_few_shot_tiered_general.yaml --load=./save/res12featcluster-minibatch-sideout-classifier-ybranch-tiered-FFF-FFF-FFF-FFF-out_conv-stage1-train3shot-test_stop_trial2/epoch-45.pth --method='cos' --dist_func_list='none' --logits_coeff_list='20' --disable_out_feat_flatten --sideout --feat_source_list='final' --branch_list='2'

    python test_few_shot_ybranch.py --shot 1 --gpu 1 --config ./configs/current_configs/test_few_shot_tiered_general.yaml --load=./save/res12featcluster-minibatch-sideout-classifier-ybranch-tiered-FFF-FFF-FFF-FFF-out_conv-stage1-train3shot-test_stop_trial2/epoch-45.pth --method='cos' --dist_func_list='none' --logits_coeff_list='20' --disable_out_feat_flatten --sideout --feat_source_list='final' --branch_list='2'

    python test_few_shot_ybranch.py --shot 1 --gpu 1 --config ./configs/current_configs/test_few_shot_tiered_general.yaml --load=./save/res12featcluster-minibatch-sideout-classifier-ybranch-tiered-FFF-FFF-FFF-FFF-out_conv-stage0-train3shot-test_stop_trial2/epoch-25.pth --method='cos' --dist_func_list='none' --logits_coeff_list='20' --disable_out_feat_flatten --sideout --feat_source_list='final' --branch_list='2'
    ```

4. Feature clustering ablation.
    ```bash
    # moss102 012346
    ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier.sh 0 FFF-FFF-TTT-FFF-out_conv-cluster32
    ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier.sh 1 FFF-FFF-TTT-FFF-out_conv-cluster100
    ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier.sh 2 FFF-FFF-TTT-FFF-out_conv-cluster200
    ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier.sh 3 TFF-FFF-TTT-FFF-out_conv-firstTconcat-cluster32
    ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier.sh 4 TFF-FFF-TTT-FFF-out_conv-firstTconcat-cluster100
    # moss105
    ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier.sh 1 TFF-FFF-TTT-FFF-out_conv-firstTconcat-cluster200

    # Debug:
    python train_classifier_sideout_classifier_temp.py --gpu 0 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier/TFF-FFF-TTT-FFF-out_conv-firstTconcat-cluster200.yaml

    python test_few_shot_ybranch.py --shot 1 --gpu 3 --config ./configs/current_configs/test_few_shot_tiered_general.yaml --load=./save/res12featcluster-minibatch-sideout-classifier-ybranch-tiered-FFF-FFF-FFF-FFF-out_conv-stage3-train3shot-test_stop_trial3/epoch-25.pth --method='cos,cos' --dist_func_list='none,none' --logits_coeff_list='20,20' --disable_out_feat_flatten --sideout --feat_source_list='final,final' --branch_list='1,2'
    ```

5. Tiered-imagenet.
    ```bash
    # moss101 1-3/2-4/5-7/8-9
    # moss102 2-4/5-7/8-9
    # moss103 0-6/2-4/5-7/8-9
    # moss105 0-1/4-5/6-7
    # moss106 no nvlink
    python train_classifier_sideout_classifier_ybranch_tiered.py --gpu 4,5,6,7 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_ybranch_tiered/FFF-FFF-FFF-FFF-out_conv-stage3-train3shot-test_stop-cos.yaml;


    # moss101
    ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_ybranch_tiered.sh 1,3,2,4 FFF-FFF-FFF-FFF-out_conv-stage0-train3shot-test_stop-cos trial2 1;
    ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_ybranch_tiered.sh 1,3,2,4 FFF-FFF-FFF-FFF-out_conv-stage0-train3shot-test_stop-cos trial3 1

    # moss101
    ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_ybranch_tiered.sh 5,7,8,9 FFF-FFF-FFF-FFF-out_conv-stage1-train3shot-test_stop-cos trial2 5;
    ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_ybranch_tiered.sh 5,7,8,9 FFF-FFF-FFF-FFF-out_conv-stage1-train3shot-test_stop-cos trial3 5

    # moss102
    ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_ybranch_tiered.sh 5,7,8,9 FFF-FFF-FFF-FFF-out_conv-stage2-train3shot-test_stop-cos trial2 5;
    ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_ybranch_tiered.sh 5,7,8,9 FFF-FFF-FFF-FFF-out_conv-stage2-train3shot-test_stop-cos trial3 5

    # moss103
    ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_ybranch_tiered.sh 0,6,2,4 FFF-FFF-FFF-FFF-out_conv-stage3-train3shot-test_stop-cos trial2 0;
    ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_ybranch_tiered.sh 0,6,2,4 FFF-FFF-FFF-FFF-out_conv-stage3-train3shot-test_stop-cos trial3 0

    # moss103
    ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_ybranch_tiered.sh 5,7,8,9 FFF-FFF-FFF-FFF-out_conv-stage4-train3shot-test_stop-cos trial2 5;
    ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_ybranch_tiered.sh 5,7,8,9 FFF-FFF-FFF-FFF-out_conv-stage4-train3shot-test_stop-cos trial3 5
    
    # moss105
    ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_ybranch_tiered.sh 0,1,2,3 TFF-FFF-TTT-FFF-out_conv-stage2-firstTconcat-train3shot-test_stop-cos trial2 0;
    ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_ybranch_tiered.sh 0,1,2,3 TFF-FFF-TTT-FFF-out_conv-stage2-firstTconcat-train3shot-test_stop-cos trial3 0

    # moss105
    ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_ybranch_tiered.sh 4,5,6,7 TFF-FFF-FFF-FFF-out_conv-stage2-firstTconcat-train3shot-test_stop-cos trial2 4;
    ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_ybranch_tiered.sh 4,5,6,7 TFF-FFF-FFF-FFF-out_conv-stage2-firstTconcat-train3shot-test_stop-cos trial3 4

    ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_ybranch_tiered.sh 4,5,6,7 FFF-FFF-FFF-FFF-out_conv-stage0-train7shot-test_stop-cos trial2 5;
    ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_ybranch_tiered.sh 4,5,6,7 FFF-FFF-FFF-FFF-out_conv-stage0-train7shot-test_stop-cos trial3 5


    # moss106
    ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_ybranch_tiered.sh 0,1,2,3 FFF-FFF-TTT-FFF-out_conv-stage2-train3shot-test_stop-cos trial2 0;
    ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_ybranch_tiered.sh 0,1,2,3 FFF-FFF-TTT-FFF-out_conv-stage2-train3shot-test_stop-cos trial3 0

    # moss106
    ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_ybranch_tiered.sh 4,5,6,7 FFF-FFF-FFF-FFF-out_conv-stage2-train7shot-test_stop-cos trial2 4;
    ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_ybranch_tiered.sh 4,5,6,7 FFF-FFF-FFF-FFF-out_conv-stage2-train7shot-test_stop-cos trial3 4
    ```

    python test_few_shot_ybranch.py --shot 1 --gpu 0,1,2,3 --config ./configs/current_configs/test_few_shot_tiered_general.yaml --load=./save/res12featcluster-minibatch-sideout-classifier-ybranch-tiered-FFF-FFF-FFF-FFF-out_conv-stage0-train3shot-test_stop-cos_trial2/epoch-25.pth --method='cos' --dist_func_list='none' --logits_coeff_list='20' --disable_out_feat_flatten --sideout --feat_source_list='final' --branch_list='2'


    python train_classifier_sideout_classifier_ybranch_tiered.py --gpu 4,5,6,7 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_ybranch_tiered/FFF-FFF-FFF-FFF-out_conv-stage0-train7shot-test_stop-cos-debug.yaml

    


    
6. No Y-branch.
    ```bash
    # moss102
    ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_tiered.sh 0,1 FFF-TTT-FFF-FFF-out_conv-test_stop trial2 0
    ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_tiered.sh 2,4 FFF-FFF-TTT-FFF-out_conv-test_stop trial2 2
    ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_tiered.sh 3,6 FFF-FFF-FFF-TTT-out_conv-test_stop trial2 3

    # moss101
    ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_tiered.sh 0,6 FFF-FFF-TTT-FFF-out_conv-firstTconcat-test_stop-cluster351 trial2 0
    
    # moss103
    ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_tiered.sh 1,3 TFF-FFF-FFF-FFF-out_conv-firstTconcat-test_stop trial2 1

    # Debug:
    python train_classifier_sideout_classifier_temp_tiered.py --gpu 3,4 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_tiered/FFF-FFF-TTT-FFF-out_conv-firstTconcat-test_stop.yaml --update_freq=2 --num_workers=0 --vscode_debug --tag=another
    ```

7. Y-branch additional experiments.
    
    ```bash
    # moss105
    ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_ybranch_tiered.sh 0,1,2,3 FFF-FFF-FFF-FFF-out_conv-stage3-train7shot-test_stop-cos trial2 0;
    ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_ybranch_tiered.sh 0,1,2,3 FFF-FFF-FFF-FFF-out_conv-stage3-train7shot-test_stop-cos trial3 0

    # moss106
    ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_ybranch_tiered.sh 0,1,2,3 FFF-FFF-FFF-FFF-out_conv-stage4-train7shot-test_stop-cos trial2 0;
    ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_ybranch_tiered.sh 0,1,2,3 FFF-FFF-FFF-FFF-out_conv-stage4-train7shot-test_stop-cos trial3 0
    ```

8. No Y-branch additional experiments.
    ```bash
    # moss101 1-3/2-4/5-7/8-9
    # moss102 2-4/5-7/8-9
    # moss103 0-6/2-4/5-7/8-9
    # moss105 0-1/4-5/6-7
    # moss106 no nvlink

    # moss103
    (stopped) ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_tiered.sh 0,6 FFF-FFF-FFF-TTT-out_conv-firstTconcat-test_stop-sideout_losses_coeff0.01 trial2 0

    (running @ moss103) ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_tiered.sh 5,7 FFF-FFF-FFF-TTT-out_conv-firstTconcat-test_stop-sideout_losses_coeff0.1 trial2 2

    (running @ moss103) ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_tiered.sh 8,9 FFF-FFF-TTT-FFF-out_conv-firstTconcat-test_stop-sideout_losses_coeff0.01 trial2 5

    (running @ moss103) ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_tiered.sh 0,6 FFF-FFF-TTT-FFF-out_conv-firstTconcat-test_stop-sideout_losses_coeff0.1 trial2 0

    # Debug:
    python train_classifier_sideout_classifier_temp_tiered.py --gpu 0,6 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_tiered/FFF-FFF-FFF-TTT-out_conv-firstTconcat-test_stop-sideout_losses_coeff0.01.yaml --update_freq=2 --num_workers=0 --vscode_debug

    # Eval.
    ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_tiered.sh 5,7 FFF-FFF-FFF-TTT-out_conv-test_stop-sideout_losses_coeff0.1 trial2 4
    ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_tiered.sh 8,9 FFF-FFF-TTT-FFF-out_conv-test_stop-sideout_losses_coeff0.01 trial2 5
    ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_tiered.sh 0,6 FFF-FFF-TTT-FFF-out_conv-test_stop-sideout_losses_coeff0.1 trial2 6
    ```

9. Y-branch evaluation.
    ```bash
    python test_few_shot_ybranch.py --shot 1 --gpu 0 --config ./configs/current_configs/test_few_shot_tiered_general.yaml --load=./save/res12featcluster-minibatch-sideout-classifier-ybranch-tiered-FFF-FFF-FFF-FFF-out_conv-stage2-train3shot-test_stop-cos_trial2/max-f-va.pth --method='cos,cos' --dist_func_list='none,none' --logits_coeff_list='1,1' --disable_out_feat_flatten --sideout --feat_source_list='final,final' --branch_list='1,2'

    python test_few_shot_ybranch.py --shot 1 --gpu 1 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load=./save/res12featcluster-minibatch-sideout-classifier-ybranch-tiered-FFF-FFF-FFF-FFF-out_conv-stage2-train3shot-test_stop-cos_trial2/max-f-va.pth --method='cos,cos' --dist_func_list='none,none' --logits_coeff_list='1,1' --disable_out_feat_flatten --sideout --feat_source_list='final,final' --branch_list='1,2'

    python test_few_shot_ybranch.py --shot 1 --gpu 2 --config ./configs/current_configs/test_few_shot_tiered_general.yaml --load=./save/res12featcluster-minibatch-sideout-classifier-ybranch-tiered-TFF-FFF-TTT-FFF-out_conv-stage2-firstTconcat-train3shot-test_stop-cos_trial2/max-f-va.pth --method='cos,cos' --dist_func_list='none,none' --logits_coeff_list='1,1' --disable_out_feat_flatten --sideout --feat_source_list='final,final' --branch_list='1,2'

    python test_few_shot_ybranch.py --shot 1 --gpu 3 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load=./save/res12featcluster-minibatch-sideout-classifier-ybranch-tiered-TFF-FFF-TTT-FFF-out_conv-stage2-firstTconcat-train3shot-test_stop-cos_trial2/max-f-va.pth --method='cos,cos' --dist_func_list='none,none' --logits_coeff_list='1,1' --disable_out_feat_flatten --sideout --feat_source_list='final,final' --branch_list='1,2'
    ```

10. Check the randomness of the training schedule.
    ```bash
    python train_classifier_sideout_classifier_ybranch.py --gpu 0,1 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_ybranch/FFF-FFF-FFF-FFF-out_conv-stage0-debug.yaml --vscode_debug --num_workers=0 --train_batches=10
    ```

11. Merge sideout layers to a unified classifier.
    - mini-ImageNet
        ```bash
        # moss101 1-3/2-4/5-7/8-9
        # moss102 2-4/5-7/8-9
        # moss103 0-6/2-4/5-7/8-9
        # moss105 0-1/4-5/6-7
        # moss106 no nvlink

        # moss101
        ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier.sh 0,6 TTT-FFF-FFF-FFF-out_conv-merge-linear
        ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier.sh 1,3 FFF-TTT-FFF-FFF-out_conv-merge-linear
        ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier.sh 2,4 FFF-FFF-TTT-FFF-out_conv-merge-linear
        ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier.sh 5,7 FFF-FFF-FFF-TTT-out_conv-merge-linear

        # Debug:
        python train_classifier_sideout_classifier_temp.py --gpu 0,6 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier/TTT-FFF-FFF-FFF-out_conv-merge-linear.yaml

        python test_few_shot.py --shot 1 --gpu 2,3 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/res12featcluster-minibatch-sideout-classifier-FFF-FFF-TTT-FFF-out_conv-merge-linear_trial2/max-va.pth
        ```
    - Tiered-ImageNet
        ```bash
        # moss102
        ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_tiered.sh 0,3 TTT-FFF-FFF-FFF-out_conv-test_stop-merge-linear trial2 0
        ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_tiered.sh 2,4 FFF-TTT-FFF-FFF-out_conv-test_stop-merge-linear trial2 2
        ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_tiered.sh 5,7 FFF-FFF-TTT-FFF-out_conv-test_stop-merge-linear trial2 5
        ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_tiered.sh 8,9 FFF-FFF-FFF-TTT-out_conv-test_stop-merge-linear trial2 8

        # Eval.
        # moss105
        ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_tiered.sh 0,3 TTT-FFF-FFF-FFF-out_conv-test_stop-merge-linear trial2 0
        ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_tiered.sh 2,4 FFF-TTT-FFF-FFF-out_conv-test_stop-merge-linear trial2 1
        ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_tiered.sh 5,7 FFF-FFF-TTT-FFF-out_conv-test_stop-merge-linear trial2 2
        ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_tiered.sh 8,9 FFF-FFF-FFF-TTT-out_conv-test_stop-merge-linear trial2 3


        # Debug:
        python train_classifier_sideout_classifier_temp_tiered.py --gpu 0,3 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_tiered/TTT-FFF-FFF-FFF-out_conv-test_stop-merge-linear.yaml --update_freq=2

        python test_few_shot.py --shot 1 --gpu 2 --config ./configs/current_configs/test_few_shot_tiered_general.yaml --load_encoder=./save/res12featcluster-minibatch-sideout-classifier-tiered-TTT-FFF-FFF-FFF-out_conv-test_stop-merge-linear/max-va.pth
        ```

12. Sideout supervision without feat clustering.
    ```bash
    # moss105
    ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_tiered.sh 4,5 TTT-FFF-FFF-FFF-dsn-relu trial2 4
    ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_tiered.sh 6,7 FFF-TTT-FFF-FFF-dsn-relu trial2 6
    # moss101
    ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_tiered.sh 8,9 FFF-FFF-TTT-FFF-dsn-relu trial2 8
    # moss102
    ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_tiered.sh 1,6 FFF-FFF-FFF-TTT-dsn-relu trial2 1

    # Eval.
    # moss101
    ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_tiered.sh 4,5 TTT-FFF-FFF-FFF-dsn-relu trial2 0
    ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_tiered.sh 6,7 FFF-TTT-FFF-FFF-dsn-relu trial2 1
    ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_tiered.sh 8,9 FFF-FFF-TTT-FFF-dsn-relu trial2 2
    ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_tiered.sh 1,6 FFF-FFF-FFF-TTT-dsn-relu trial2 3

    # Debug:
    python train_classifier_sideout_classifier_temp_tiered.py --gpu 4,5 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_tiered/FFF-FFF-TTT-FFF-dsn-relu.yaml --update_freq=2
    
    python test_few_shot.py --shot 1 --gpu 6 --config ./configs/current_configs/test_few_shot_tiered_general.yaml --load_encoder=./save/res12featcluster-minibatch-sideout-classifier-tiered-FFF-FFF-TTT-FFF-dsn-relu/max-va.pth
    ```

13. Yifan's suggested experiments: TTT sideout classifier right after the stem.
    ```bash
    # moss101 1-3/2-4/5-7/8-9
    # moss102 2-4/5-7/8-9
    # moss103 0-6/2-4/5-7/8-9
    # moss105 0-1/4-5/6-7
    # moss106 no nvlink

    # moss106
    ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_ybranch_tiered.sh 4,5,6,7 FFF-FFF-FFF-TTT-out_conv-stage3-train7shot-test_stop-cos trial2 4;
    # moss102 1324 / 5789
    ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_ybranch_tiered.sh 1,3,2,4 FFF-FFF-TTT-FFF-out_conv-stage2-train7shot-test_stop-cos trial2 1;
    ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_ybranch_tiered.sh 5,7,8,9 FFF-TTT-FFF-FFF-out_conv-stage1-train7shot-test_stop-cos trial2 5;

    # Debug:
    python train_classifier_sideout_classifier_ybranch_tiered.py --gpu 4,5,6,7 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_ybranch_tiered/FFF-FFF-FFF-TTT-out_conv-stage3-train7shot-test_stop-cos.yaml
    ```

14. More experiments about the smaller coefficients.
    ```bash
    # moss101 1-3/2-4/5-7/8-9
    # moss102 2-4/5-7/8-9
    # moss103 0-6/2-4/5-7/8-9
    # moss105 0-1/4-5/6-7
    # moss106 no nvlink

    # moss102
    ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_tiered.sh 0,1,2,3 FFF-FFF-FFF-TTT-out_conv-test_stop-sideout_losses_coeff0.01 trial3 6

    ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_tiered.sh 5,7,8,9 FFF-FFF-FFF-TTT-out_conv-test_stop-sideout_losses_coeff0.1 trial3 5

    # moss105
    ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_tiered.sh 0,1,2,3 FFF-FFF-TTT-FFF-out_conv-test_stop-sideout_losses_coeff0.01 trial3 0

    ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_tiered.sh 4,5,6,7 FFF-FFF-TTT-FFF-out_conv-test_stop-sideout_losses_coeff0.1 trial3 4

    # moss103
    ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_tiered.sh 0,6,2,4 FFF-TTT-FFF-FFF-out_conv-test_stop-sideout_losses_coeff0.01 trial3 0

    ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_tiered.sh 5,7,8,9 FFF-TTT-FFF-FFF-out_conv-test_stop-sideout_losses_coeff0.1 trial3 5
    ```

15. CIFAR-FS experiments.
    ```bash
    # Debug:
    python train_classifier_sideout_classifier_temp_cifar-fs.py --gpu 0 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_cifar-fs/FFF-FFF-FFF-FFF-out_conv-test_stop-lee-opt.yaml

    python test_few_shot.py --shot 1 --gpu 0 --config ./configs/current_configs/test_few_shot_cifar-fs_general.yaml --load_encoder=./save/res12featcluster-minibatch-sideout-classifier-cifar-fs-FFF-FFF-FFF-FFF-out_conv-test_stop-lee-opt/max-va.pth

    # moss105 - No Y-branch
    ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_cifar-fs.sh 0 FFF-FFF-FFF-FFF-out_conv-test_stop-lee-opt trial2 0;
    ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_cifar-fs.sh 0 FFF-FFF-FFF-FFF-out_conv-test_stop-lee-opt trial3 0;
    ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_cifar-fs.sh 0 FFF-FFF-FFF-FFF-out_conv-test_stop-lee-opt trial4 0;

    ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_cifar-fs.sh 1 FFF-FFF-FFF-FFF-out_conv-test_stop trial2 1;
    ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_cifar-fs.sh 1 FFF-FFF-FFF-FFF-out_conv-test_stop trial3 1;
    ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_cifar-fs.sh 1 FFF-FFF-FFF-FFF-out_conv-test_stop trial4 1;

    ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_cifar-fs.sh 2 FFF-FFF-TTT-FFF-out_conv-test_stop-lee-opt trial2 2;
    ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_cifar-fs.sh 2 FFF-FFF-TTT-FFF-out_conv-test_stop-lee-opt trial3 2;
    ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_cifar-fs.sh 2 FFF-FFF-TTT-FFF-out_conv-test_stop-lee-opt trial4 2;

    ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_cifar-fs.sh 3 FFF-FFF-TTT-FFF-out_conv-test_stop trial2 3;
    ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_cifar-fs.sh 3 FFF-FFF-TTT-FFF-out_conv-test_stop trial3 3;
    ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_cifar-fs.sh 3 FFF-FFF-TTT-FFF-out_conv-test_stop trial4 3;

    ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_cifar-fs.sh 4 TFF-FFF-TTT-FFF-out_conv-test_stop-firstTconcat-lee-opt trial2 4;
    ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_cifar-fs.sh 4 TFF-FFF-TTT-FFF-out_conv-test_stop-firstTconcat-lee-opt trial3 4;
    ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_cifar-fs.sh 4 TFF-FFF-TTT-FFF-out_conv-test_stop-firstTconcat-lee-opt trial4 4;

    ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_cifar-fs.sh 5 TFF-FFF-TTT-FFF-out_conv-test_stop-firstTconcat trial2 5;
    ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_cifar-fs.sh 5 TFF-FFF-TTT-FFF-out_conv-test_stop-firstTconcat trial3 5;
    ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_cifar-fs.sh 5 TFF-FFF-TTT-FFF-out_conv-test_stop-firstTconcat trial4 5;

    # moss101 - No Y-branch
    ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_cifar-fs_3trials.sh 1 FFF-FFF-FFF-TTT-out_conv-test_stop 1
    ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_cifar-fs_3trials.sh 2 FFF-TTT-FFF-FFF-out_conv-test_stop 2
    ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_cifar-fs_3trials.sh 3 TTT-FFF-FFF-FFF-out_conv-test_stop 3
    ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_cifar-fs_3trials.sh 4 TFF-FFF-FFF-FFF-out_conv-test_stop-firstTconcat 4

    # moss106 - Y-branch
    ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_ybranch_cifar-fs_3trials.sh 1 FFF-FFF-FFF-FFF-out_conv-stage0-train7shot-test_stop-cos 1;
    ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_ybranch_cifar-fs_3trials.sh 2 FFF-FFF-FFF-FFF-out_conv-stage1-train7shot-test_stop-cos 2;
    ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_ybranch_cifar-fs_3trials.sh 3 FFF-FFF-FFF-FFF-out_conv-stage2-train7shot-test_stop-cos 3;
    ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_ybranch_cifar-fs_3trials.sh 4 FFF-FFF-FFF-FFF-out_conv-stage3-train7shot-test_stop-cos 4;
    ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_ybranch_cifar-fs_3trials.sh 5 FFF-FFF-FFF-FFF-out_conv-stage4-train7shot-test_stop-cos 5;
    ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_ybranch_cifar-fs_3trials.sh 6 FFF-FFF-TTT-FFF-out_conv-stage2-firstTconcat-train7shot-test_stop-cos 6;
    ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_ybranch_cifar-fs_3trials.sh 7 TFF-FFF-FFF-FFF-out_conv-stage2-firstTconcat-train7shot-test_stop-cos 7;

    # moss101 - Y-branch
    ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_ybranch_cifar-fs_3trials.sh 0 TFF-FFF-TTT-FFF-out_conv-stage2-firstTconcat-train7shot-test_stop-cos 0;

    # moss102 - Y-branch
    ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_ybranch_cifar-fs_3trials.sh 0  FFF-FFF-FFF-FFF-out_conv-stage0-train7shot-test_stop-sqr 0

    ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_ybranch_cifar-fs_3trials.sh 1 FFF-FFF-FFF-FFF-out_conv-stage2-train7shot-test_stop-sqr 1

    ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_ybranch_cifar-fs_3trials.sh 2  TFF-FFF-TTT-FFF-out_conv-stage2-firstTconcat-train7shot-test_stop-sqr 2

    # Debug:
    python train_classifier_sideout_classifier_ybranch_cifar-fs.py --gpu 0 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_ybranch_cifar-fs/TFF-FFF-TTT-FFF-out_conv-stage2-firstTconcat-train7shot-test_stop-cos.yaml --train_batches=10

    python test_few_shot_ybranch.py --shot 1 --gpu 0 --config ./configs/current_configs/test_few_shot_cifar-fs_general.yaml --load=./save/res12featcluster-minibatch-sideout-classifier-ybranch-cifar-fs-FFF-FFF-FFF-FFF-out_conv-stage0-train7shot-test_stop-cos/max-f-va.pth --method='cos' --dist_func_list='none' --logits_coeff_list='20' --disable_out_feat_flatten --sideout --feat_source_list='final' --branch_list='1'


    rsync -av --progress /home/shawn/cube/capsule/fewshot/materials/cifar-fs ~/weijian/datasets/;
    rsync -av --progress /home/shawn/cube/capsule/fewshot/materials/fc100 ~/weijian/datasets/;
    ln -s ~/weijian/datasets/cifar-fs ~/weijian/pentagon/code/Capsules/capsule/fewshot/materials/;
    ln -s ~/weijian/datasets/fc100 ~/weijian/pentagon/code/Capsules/capsule/fewshot/materials/

    python test_few_shot_ybranch.py --shot 1 --gpu 0 --config ./configs/current_configs/test_few_shot_cifar-fs_general.yaml --load=./save/res12featcluster-minibatch-sideout-classifier-ybranch-cifar-fs-FFF-FFF-FFF-FFF-out_conv-stage2-train7shot-test_stop-sqr_trial2/max-f-va.pth --method='cos' --dist_func_list='none' --logits_coeff_list='20' --disable_out_feat_flatten --sideout --feat_source_list='final' --branch_list='2'
    ```

16. Feature Augmentation for CIFAR-FS.
    - Flatten + AvgPool
        ```bash
        ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_ybranch_cifar-fs_eval_feataug.sh 0 TFF-FFF-TTT-FFF-out_conv-stage2-firstTconcat-train7shot-test_stop-cos trial2 0

        ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_ybranch_cifar-fs_eval_feataug.sh 1 TFF-FFF-TTT-FFF-out_conv-stage2-firstTconcat-train7shot-test_stop-cos trial3 1

        ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_ybranch_cifar-fs_eval_feataug.sh 2 TFF-FFF-TTT-FFF-out_conv-stage2-firstTconcat-train7shot-test_stop-cos trial4 2




        ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_ybranch_cifar-fs_eval_feataug.sh 3 TFF-FFF-TTT-FFF-out_conv-stage2-firstTconcat-train7shot-test_stop-sqr trial2 3

        ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_ybranch_cifar-fs_eval_feataug.sh 4 TFF-FFF-TTT-FFF-out_conv-stage2-firstTconcat-train7shot-test_stop-sqr trial3 4

        ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_ybranch_cifar-fs_eval_feataug.sh 5 TFF-FFF-TTT-FFF-out_conv-stage2-firstTconcat-train7shot-test_stop-sqr trial4 5
        ```

17. Experiments for FC100.
    ```bash
     # Debug:
    python train_classifier_sideout_classifier_temp_fc100.py --gpu 0 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_fc100/FFF-FFF-FFF-FFF-out_conv-test_stop.yaml

    python test_few_shot.py --shot 1 --gpu 1 --config ./configs/current_configs/test_few_shot_fc100_general.yaml --load_encoder=./save/res12featcluster-minibatch-sideout-classifier-fc100-FFF-FFF-FFF-FFF-out_conv-test_stop/max-va.pth

    # moss106 - No Y-branch
    #./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_fc100_3trials.sh 0 FFF-FFF-FFF-FFF-out_conv-test_stop 0
    ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_fc100_3trials.sh 1 FFF-FFF-FFF-TTT-out_conv-test_stop 1
    ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_fc100_3trials.sh 2 FFF-FFF-TTT-FFF-out_conv-test_stop 2
    ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_fc100_3trials.sh 3 FFF-TTT-FFF-FFF-out_conv-test_stop 3
    ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_fc100_3trials.sh 4 TTT-FFF-FFF-FFF-out_conv-test_stop 4
    ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_fc100_3trials.sh 5 TFF-FFF-FFF-FFF-out_conv-test_stop-firstTconcat 5
    ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_fc100_3trials.sh 6 TFF-FFF-TTT-FFF-out_conv-test_stop-firstTconcat 6
    ```

18. Add sideout classifier for the meta branch.
    ```bash
    # moss105
    ./scripts/traintest-featcluster_minibatch_res12_classifier-and-meta_sideout_classifier_ybranch_cifar-fs.sh 0 FFF-FFF-TTTTTT-FFF-out_conv-stage2-firstTconcat-train7shot-test_stop-cos trial2 0

    ./scripts/traintest-featcluster_minibatch_res12_classifier-and-meta_sideout_classifier_ybranch_cifar-fs.sh 1 FFF-FFF-TTTTTT-FFF-out_conv-stage2-firstTconcat-train7shot-test_stop-cos trial3 1

    ./scripts/traintest-featcluster_minibatch_res12_classifier-and-meta_sideout_classifier_ybranch_cifar-fs.sh 2 FFF-FFF-TTTTTT-FFF-out_conv-stage2-firstTconcat-train7shot-test_stop-cos trial4 2

    ./scripts/traintest-featcluster_minibatch_res12_classifier-and-meta_sideout_classifier_ybranch_cifar-fs.sh 3 FFF-FFF-FFFTTT-FFF-out_conv-stage2-firstTconcat-train7shot-test_stop-cos trial2 3

    ./scripts/traintest-featcluster_minibatch_res12_classifier-and-meta_sideout_classifier_ybranch_cifar-fs.sh 4 FFF-FFF-FFFTTT-FFF-out_conv-stage2-firstTconcat-train7shot-test_stop-cos trial3 4

    ./scripts/traintest-featcluster_minibatch_res12_classifier-and-meta_sideout_classifier_ybranch_cifar-fs.sh 5 FFF-FFF-FFFTTT-FFF-out_conv-stage2-firstTconcat-train7shot-test_stop-cos trial4 5

    ./scripts/traintest-featcluster_minibatch_res12_classifier-and-meta_sideout_classifier_ybranch_cifar-fs_3trials.sh 6 TFF-FFF-FFFTTT-FFF-out_conv-stage2-firstTconcat-train7shot-test_stop-cos 6

    ./scripts/traintest-featcluster_minibatch_res12_classifier-and-meta_sideout_classifier_ybranch_cifar-fs_3trials.sh 7 TFF-FFF-TTTTTT-FFF-out_conv-stage2-firstTconcat-train7shot-test_stop-cos 7

    # Debug:
    python train_classifier-and-meta_sideout_classifier_ybranch_cifar-fs.py --gpu 0 --config ./configs/current_configs/featcluster_minibatch_res12_classifier-and-meta_sideout_classifier_ybranch_cifar-fs/FFF-FFF-TTTTTT-FFF-out_conv-stage2-firstTconcat-train7shot-test_stop-cos.yaml --train_batches=10 --num_workers=0 --vscode_debug

    python test_few_shot_ybranch.py --shot 1 --gpu 0 --config ./configs/current_configs/test_few_shot_cifar-fs_general.yaml --load=./save/res12featcluster-minibatch-classifier-and-meta-sideout-classifier-ybranch-cifar-fs-FFF-FFF-TTTTTT-FFF-out_conv-stage2-firstTconcat-train7shot-test_stop-cos/max-f-va.pth --method='cos' --dist_func_list='none' --logits_coeff_list='20' --disable_out_feat_flatten --sideout --feat_source_list='final' --branch_list='1'
    ```

19. Tiered-ImageNet 
    ```bash
    # moss101 1-3/2-4/5-7/8-9
    # moss102 2-4/5-7/8-9
    # moss103 0-6/2-4/5-7/8-9
    # moss105 0-1/4-5/6-7
    # moss106 no nvlink

    # moss101
    ./scripts/traintest-featcluster_minibatch_res12_classifier-and-meta_sideout_classifier_ybranch_tiered.sh 1,3,2,4 FFF-FFF-FFFTTT-FFF-out_conv-stage2-firstTconcat-train7shot-test_stop-cos trial2 1
    ./scripts/traintest-featcluster_minibatch_res12_classifier-and-meta_sideout_classifier_ybranch_tiered.sh 5,7,8,9 FFF-FFF-FFFTTT-FFF-out_conv-stage2-firstTconcat-train7shot-test_stop-cos trial3 5
    
    # moss102
    ./scripts/traintest-featcluster_minibatch_res12_classifier-and-meta_sideout_classifier_ybranch_tiered.sh 0,1,2,4 FFF-FFF-TTTTTT-FFF-out_conv-stage2-firstTconcat-train7shot-test_stop-cos trial2 0
    ./scripts/traintest-featcluster_minibatch_res12_classifier-and-meta_sideout_classifier_ybranch_tiered.sh 5,7,8,9 FFF-FFF-TTTTTT-FFF-out_conv-stage2-firstTconcat-train7shot-test_stop-cos trial3 5

    # moss103
    ./scripts/traintest-featcluster_minibatch_res12_classifier-and-meta_sideout_classifier_ybranch_tiered.sh 0,1,2,3,4,5,6,7 TFF-FFF-TTTTTT-FFF-out_conv-stage2-firstTconcat-train7shot-test_stop-cos trial2 0

    # moss106
    ./scripts/traintest-featcluster_minibatch_res12_classifier-and-meta_sideout_classifier_ybranch_tiered.sh 0,1,2,3,4,5,6,7 TFF-FFF-FFFTTT-FFF-out_conv-stage2-firstTconcat-train7shot-test_stop-cos trial2 0
    ```
    
20. Tiered-ImageNet test time analysis.
    ```bash
    python test_few_shot_ybranch.py --shot 1 --gpu 0 --config ./configs/current_configs/test_few_shot_tiered_general.yaml --load=/home/shawn/cube/weijian/fewshot-save/res12featcluster-minibatch-sideout-classifier-ybranch-tiered-FFF-FFF-FFF-FFF-out_conv-stage2-train3shot-test_stop_trial2/max-f-va.pth --method='cos' --dist_func_list='none' --logits_coeff_list='20' --disable_out_feat_flatten --sideout --feat_source_list='final' --branch_list='1'

    python test_few_shot_ybranch_for_bn_test.py --shot 1 --gpu 0 --config ./configs/current_configs/test_few_shot_tiered_general.yaml --load=/home/shawn/cube/weijian/fewshot-save/res12featcluster-minibatch-sideout-classifier-ybranch-tiered-FFF-FFF-FFF-FFF-out_conv-stage2-train3shot-test_stop_trial2/max-f-va.pth --method='cos' --dist_func_list='none' --logits_coeff_list='20' --disable_out_feat_flatten --sideout --feat_source_list='final' --branch_list='1' --bn_test='all-eval'

    python test_few_shot_ybranch_for_bn_test.py --shot 1 --gpu 1 --config ./configs/current_configs/test_few_shot_tiered_general.yaml --load=/home/shawn/cube/weijian/fewshot-save/res12featcluster-minibatch-sideout-classifier-ybranch-tiered-FFF-FFF-FFF-FFF-out_conv-stage2-train3shot-test_stop_trial2/max-f-va.pth --method='cos' --dist_func_list='none' --logits_coeff_list='20' --disable_out_feat_flatten --sideout --feat_source_list='final' --branch_list='1' --bn_test='all-train'

    python test_few_shot_ybranch_for_bn_test.py --shot 1 --gpu 2 --config ./configs/current_configs/test_few_shot_tiered_general.yaml --load=/home/shawn/cube/weijian/fewshot-save/res12featcluster-minibatch-sideout-classifier-ybranch-tiered-FFF-FFF-FFF-FFF-out_conv-stage2-train3shot-test_stop_trial2/max-f-va.pth --method='cos' --dist_func_list='none' --logits_coeff_list='20' --disable_out_feat_flatten --sideout --feat_source_list='final' --branch_list='1' --bn_test='stem[1]-train'

    python test_few_shot_ybranch_for_bn_test.py --shot 1 --gpu 3 --config ./configs/current_configs/test_few_shot_tiered_general.yaml --load=/home/shawn/cube/weijian/fewshot-save/res12featcluster-minibatch-sideout-classifier-ybranch-tiered-FFF-FFF-FFF-FFF-out_conv-stage2-train3shot-test_stop_trial2/max-f-va.pth --method='cos' --dist_func_list='none' --logits_coeff_list='20' --disable_out_feat_flatten --sideout --feat_source_list='final' --branch_list='1' --bn_test='stem[0]-train'






    python test_few_shot_ybranch_for_bn_test.py --shot 1 --gpu 0 --config ./configs/current_configs/test_few_shot_tiered_general.yaml --load=/home/shawn/cube/weijian/fewshot-save/res12featcluster-minibatch-sideout-classifier-ybranch-tiered-FFF-FFF-FFF-FFF-out_conv-stage2-train3shot-test_stop_trial2/max-f-va.pth --method='cos' --dist_func_list='none' --logits_coeff_list='20' --disable_out_feat_flatten --sideout --feat_source_list='final' --branch_list='1' --bn_test='all-eval'

    python test_few_shot_ybranch_for_bn_test.py --shot 1 --gpu 1 --config ./configs/current_configs/test_few_shot_tiered_general.yaml --load=/home/shawn/cube/weijian/fewshot-save/res12featcluster-minibatch-sideout-classifier-ybranch-tiered-FFF-FFF-FFF-FFF-out_conv-stage2-train3shot-test_stop_trial2/max-f-va.pth --method='cos' --dist_func_list='none' --logits_coeff_list='20' --disable_out_feat_flatten --sideout --feat_source_list='final' --branch_list='1' --bn_test='stem-train'

    python test_few_shot_ybranch_for_bn_test.py --shot 1 --gpu 2 --config ./configs/current_configs/test_few_shot_tiered_general.yaml --load=/home/shawn/cube/weijian/fewshot-save/res12featcluster-minibatch-sideout-classifier-ybranch-tiered-FFF-FFF-FFF-FFF-out_conv-stage2-train3shot-test_stop_trial2/max-f-va.pth --method='cos' --dist_func_list='none' --logits_coeff_list='20' --disable_out_feat_flatten --sideout --feat_source_list='final' --branch_list='1' --bn_test='stem[1]-train'

    python test_few_shot_ybranch_for_bn_test.py --shot 1 --gpu 2 --config ./configs/current_configs/test_few_shot_tiered_general.yaml --load=/home/shawn/cube/weijian/fewshot-save/res12featcluster-minibatch-sideout-classifier-ybranch-tiered-FFF-FFF-FFF-FFF-out_conv-stage2-train3shot-test_stop_trial2/max-f-va.pth --method='cos' --dist_func_list='none' --logits_coeff_list='20' --disable_out_feat_flatten --sideout --feat_source_list='final' --branch_list='1' --bn_test='stem[1]-bn-train'

    python test_few_shot_ybranch_for_bn_test.py --shot 1 --gpu 3 --config ./configs/current_configs/test_few_shot_tiered_general.yaml --load=/home/shawn/cube/weijian/fewshot-save/res12featcluster-minibatch-sideout-classifier-ybranch-tiered-FFF-FFF-FFF-FFF-out_conv-stage2-train3shot-test_stop_trial2/max-f-va.pth --method='cos' --dist_func_list='none' --logits_coeff_list='20' --disable_out_feat_flatten --sideout --feat_source_list='final' --branch_list='1' --bn_test='stem[0]-train'





    python test_few_shot_ybranch_for_bn_test.py --shot 1 --gpu 4 --config ./configs/current_configs/test_few_shot_tiered_general.yaml --load=/home/shawn/cube/weijian/fewshot-save/res12featcluster-minibatch-sideout-classifier-ybranch-tiered-FFF-FFF-FFF-FFF-out_conv-stage2-train3shot-test_stop_trial2/max-f-va.pth --method='cos' --dist_func_list='none' --logits_coeff_list='20' --disable_out_feat_flatten --sideout --feat_source_list='final' --branch_list='1' --bn_test='all-eval' --ep_per_batch=4

    python test_few_shot_ybranch_for_bn_test.py --shot 1 --gpu 5 --config ./configs/current_configs/test_few_shot_tiered_general.yaml --load=/home/shawn/cube/weijian/fewshot-save/res12featcluster-minibatch-sideout-classifier-ybranch-tiered-FFF-FFF-FFF-FFF-out_conv-stage2-train3shot-test_stop_trial2/max-f-va.pth --method='cos' --dist_func_list='none' --logits_coeff_list='20' --disable_out_feat_flatten --sideout --feat_source_list='final' --branch_list='1' --bn_test='stem-train' --ep_per_batch=4

    python test_few_shot_ybranch_for_bn_test.py --shot 1 --gpu 6 --config ./configs/current_configs/test_few_shot_tiered_general.yaml --load=/home/shawn/cube/weijian/fewshot-save/res12featcluster-minibatch-sideout-classifier-ybranch-tiered-FFF-FFF-FFF-FFF-out_conv-stage2-train3shot-test_stop_trial2/max-f-va.pth --method='cos' --dist_func_list='none' --logits_coeff_list='20' --disable_out_feat_flatten --sideout --feat_source_list='final' --branch_list='1' --bn_test='stem[1]-train' --ep_per_batch=4

    python test_few_shot_ybranch_for_bn_test.py --shot 1 --gpu 6 --config ./configs/current_configs/test_few_shot_tiered_general.yaml --load=/home/shawn/cube/weijian/fewshot-save/res12featcluster-minibatch-sideout-classifier-ybranch-tiered-FFF-FFF-FFF-FFF-out_conv-stage2-train3shot-test_stop_trial2/max-f-va.pth --method='cos' --dist_func_list='none' --logits_coeff_list='20' --disable_out_feat_flatten --sideout --feat_source_list='final' --branch_list='1' --bn_test='stem[1]-bn-train' --ep_per_batch=4

    python test_few_shot_ybranch_for_bn_test.py --shot 1 --gpu 7 --config ./configs/current_configs/test_few_shot_tiered_general.yaml --load=/home/shawn/cube/weijian/fewshot-save/res12featcluster-minibatch-sideout-classifier-ybranch-tiered-FFF-FFF-FFF-FFF-out_conv-stage2-train3shot-test_stop_trial2/max-f-va.pth --method='cos' --dist_func_list='none' --logits_coeff_list='20' --disable_out_feat_flatten --sideout --feat_source_list='final' --branch_list='1' --bn_test='stem[0]-train' --ep_per_batch=4

    python test_few_shot_ybranch_for_bn_test.py --shot 1 --gpu 5 --config ./configs/current_configs/test_few_shot_tiered_general.yaml --load=/home/shawn/cube/weijian/fewshot-save/res12featcluster-minibatch-sideout-classifier-ybranch-tiered-FFF-FFF-FFF-FFF-out_conv-stage2-train3shot-test_stop_trial2/max-f-va.pth --method='cos' --dist_func_list='none' --logits_coeff_list='20' --disable_out_feat_flatten --sideout --feat_source_list='final' --branch_list='1' --bn_test='stem[0]-bn-train' --ep_per_batch=4
    ```

21. ImageNet-800 experiments.
    ```bash
    python train_classifier_sideout_classifier_temp_imagenet.py --gpu 0,1,2,3,4,5,6,7 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_imagenet/FFF-FFF-FFF-FFF-out_conv-test_stop.yaml
    
    python train_classifier_sideout_classifier_temp_imagenet.py --gpu 0,1,2,3,4,5,6,7 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_imagenet/FFF-FFF-TTT-FFF-out_conv-test_stop-firstTconcat-local.yaml --num_workers=16
    ```

22. CIFAR-FS experiments to validate the effectivenesss of clustering features for next layer.
    ```bash
    # (finished) ./scripts/traintest-featcluster_minibatch_res12_classifier-and-meta_sideout_classifier_ybranch_cifar-fs.sh 3 FFF-FFF-FFFTTT-FFF-out_conv-stage2-firstTconcat-train7shot-test_stop-cos trial2 3

    # (finished) ./scripts/traintest-featcluster_minibatch_res12_classifier-and-meta_sideout_classifier_ybranch_cifar-fs.sh 4 FFF-FFF-FFFTTT-FFF-out_conv-stage2-firstTconcat-train7shot-test_stop-cos trial3 4

    # (finished) ./scripts/traintest-featcluster_minibatch_res12_classifier-and-meta_sideout_classifier_ybranch_cifar-fs.sh 5 FFF-FFF-FFFTTT-FFF-out_conv-stage2-firstTconcat-train7shot-test_stop-cos trial4 5

    ./scripts/traintest-featcluster_minibatch_res12_classifier-and-meta_sideout_classifier_ybranch_cifar-fs.sh 0 FFF-FFF-FFFTTT-FFF-out_conv-stage2-TTTwithconcat-train7shot-test_stop-cos trial2 0

    ./scripts/traintest-featcluster_minibatch_res12_classifier-and-meta_sideout_classifier_ybranch_cifar-fs.sh 1 FFF-FFF-FFFTTT-FFF-out_conv-stage2-TTTwithconcat-train7shot-test_stop-cos trial3 1

    ./scripts/traintest-featcluster_minibatch_res12_classifier-and-meta_sideout_classifier_ybranch_cifar-fs.sh 2 FFF-FFF-FFFTTT-FFF-out_conv-stage2-TTTwithconcat-train7shot-test_stop-cos trial4 2

    ./scripts/traintest-featcluster_minibatch_res12_classifier-and-meta_sideout_classifier_ybranch_cifar-fs.sh 3 FFF-FFF-FFFTTT-FFF-out_conv-stage2-TTTonlyconcat-train7shot-test_stop-cos trial2 3

    ./scripts/traintest-featcluster_minibatch_res12_classifier-and-meta_sideout_classifier_ybranch_cifar-fs.sh 4 FFF-FFF-FFFTTT-FFF-out_conv-stage2-TTTonlyconcat-train7shot-test_stop-cos trial3 4

    ./scripts/traintest-featcluster_minibatch_res12_classifier-and-meta_sideout_classifier_ybranch_cifar-fs.sh 5 FFF-FFF-FFFTTT-FFF-out_conv-stage2-TTTonlyconcat-train7shot-test_stop-cos trial4 5
    
    # Debug:
    python train_classifier-and-meta_sideout_classifier_ybranch_cifar-fs.py --gpu 0 --config ./configs/current_configs/featcluster_minibatch_res12_classifier-and-meta_sideout_classifier_ybranch_cifar-fs/FFF-FFF-FFFTTT-FFF-out_conv-stage2-TTTwithconcat-train7shot-test_stop-cos.yaml --tag=debug --vscode_debug --num_workers=0

    python train_classifier-and-meta_sideout_classifier_ybranch_cifar-fs.py --gpu 1 --config ./configs/current_configs/featcluster_minibatch_res12_classifier-and-meta_sideout_classifier_ybranch_cifar-fs/FFF-FFF-FFFTTT-FFF-out_conv-stage2-TTTonlyconcat-train7shot-test_stop-cos.yaml --tag=debug --vscode_debug --num_workers=0
    ```

23. Additional experiments for tiered-ImageNet.
    ```bash
    # moss101 1-3/2-4/5-7/8-9
    # moss102 2-4/5-7/8-9
    # moss103 0-6/2-4/5-7/8-9
    # moss105 0-1/4-5/6-7
    # moss106 no nvlink

    # moss101
    ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_ybranch_tiered.sh 1,3,2,4 FFF-FFF-FFF-FFF-out_conv-stage2-train7shot-test_stop-sqr trial2 1;

    # moss102
    ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_ybranch_tiered.sh 0,1,2,4 FFF-FFF-TTT-FFF-out_conv-stage2-firstTconcat-train7shot-test_stop-sqr trial2 0;

    ./scripts/traintest-featcluster_minibatch_res12_classifier-and-meta_sideout_classifier_ybranch_tiered.sh 5,7,8,9 FFF-FFF-FFFTTT-FFF-out_conv-stage2-firstTconcat-train7shot-test_stop-sqr trial2 5

    # moss103
    # (cancelled) ./scripts/traintest-featcluster_minibatch_res12_classifier-and-meta_sideout_classifier_ybranch_tiered.sh 0,6,2,4 FFF-FFF-TTTTTT-FFF-out_conv-stage2-firstTconcat-train7shot-test_stop-sqr-lastTTTwithconcat trial2 0

    # (cancelled) ./scripts/traintest-featcluster_minibatch_res12_classifier-and-meta_sideout_classifier_ybranch_tiered.sh 5,7,8,9 FFF-FFF-TTTTTT-FFF-out_conv-stage2-firstTconcat-train7shot-test_stop-sqr-TTTTTTwithconcat trial2 5

    # moss105
    ./scripts/traintest-featcluster_minibatch_res12_classifier-and-meta_sideout_classifier_ybranch_tiered.sh 0,1,2,3 FFF-FFF-TTTTTT-FFF-out_conv-stage2-firstTconcat-train7shot-test_stop-sqr trial2 0

    ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_ybranch_tiered.sh 4,5,6,7 FFF-FFF-FFF-FFF-out_conv-stage2-train15shot-test_stop-sqr trial2 5;

    # moss105 (new)
    ./scripts/traintest-featcluster_minibatch_res12_classifier-and-meta_sideout_classifier_ybranch_tiered.sh 0,6,2,4 FFF-FFF-TTTTTT-FFF-out_conv-stage2-firstTconcat-train7shot-test_stop-cos-lastTTTwithconcat trial2 0

    ./scripts/traintest-featcluster_minibatch_res12_classifier-and-meta_sideout_classifier_ybranch_tiered.sh 5,7,8,9 FFF-FFF-TTTTTT-FFF-out_conv-stage2-firstTconcat-train7shot-test_stop-cos-TTTTTTwithconcat trial2 5

    # Debug:
    python train_classifier-and-meta_sideout_classifier_ybranch_tiered.py --gpu 0,1,2,3 --config ./configs/current_configs/featcluster_minibatch_res12_classifier-and-meta_sideout_classifier_ybranch_tiered/FFF-FFF-TTTTTT-FFF-out_conv-stage2-firstTconcat-train7shot-test_stop-sqr-lastTTTwithconcat-debug.yaml --tag=debug --num_workers=0 --vscode_debug


    ```

24. Additional experiments for CIFAR-FS to verify the effectiveness of concatenation.
    ```bash
    # moss102
    ./scripts/traintest-featcluster_minibatch_res12_classifier-and-meta_sideout_classifier_ybranch_cifar-fs.sh 0 FFF-FFF-FFF-FFFTTT-out_conv-stage2-TTTnoconcat-train7shot-test_stop-cos trial2 0
    ./scripts/traintest-featcluster_minibatch_res12_classifier-and-meta_sideout_classifier_ybranch_cifar-fs.sh 1 FFF-FFF-FFF-FFFTTT-out_conv-stage2-TTTnoconcat-train7shot-test_stop-cos trial3 1
    ./scripts/traintest-featcluster_minibatch_res12_classifier-and-meta_sideout_classifier_ybranch_cifar-fs.sh 2 FFF-FFF-FFF-FFFTTT-out_conv-stage2-TTTnoconcat-train7shot-test_stop-cos trial4 2

    ./scripts/traintest-featcluster_minibatch_res12_classifier-and-meta_sideout_classifier_ybranch_cifar-fs.sh 3 FFF-FFF-FFF-FFFTTT-out_conv-stage2-TTTwithconcat-train7shot-test_stop-cos trial2 3
    ./scripts/traintest-featcluster_minibatch_res12_classifier-and-meta_sideout_classifier_ybranch_cifar-fs.sh 4 FFF-FFF-FFF-FFFTTT-out_conv-stage2-TTTwithconcat-train7shot-test_stop-cos trial3 4
    ./scripts/traintest-featcluster_minibatch_res12_classifier-and-meta_sideout_classifier_ybranch_cifar-fs.sh 5 FFF-FFF-FFF-FFFTTT-out_conv-stage2-TTTwithconcat-train7shot-test_stop-cos trial4 5

    
    ./scripts/traintest-featcluster_minibatch_res12_classifier-and-meta_sideout_classifier_ybranch_cifar-fs.sh 6 FFF-FFFTTT-FFF-FFF-out_conv-stage0-TTTnoconcat-train7shot-test_stop-cos trial2 6
    ./scripts/traintest-featcluster_minibatch_res12_classifier-and-meta_sideout_classifier_ybranch_cifar-fs.sh 7 FFF-FFFTTT-FFF-FFF-out_conv-stage0-TTTnoconcat-train7shot-test_stop-cos trial3 7
    ./scripts/traintest-featcluster_minibatch_res12_classifier-and-meta_sideout_classifier_ybranch_cifar-fs.sh 8 FFF-FFFTTT-FFF-FFF-out_conv-stage0-TTTnoconcat-train7shot-test_stop-cos trial4 8

    # moss103
    ./scripts/traintest-featcluster_minibatch_res12_classifier-and-meta_sideout_classifier_ybranch_cifar-fs.sh 0 FFF-FFFTTT-FFF-FFF-out_conv-stage0-TTTwithconcat-train7shot-test_stop-cos trial2 0
    ./scripts/traintest-featcluster_minibatch_res12_classifier-and-meta_sideout_classifier_ybranch_cifar-fs.sh 1 FFF-FFFTTT-FFF-FFF-out_conv-stage0-TTTwithconcat-train7shot-test_stop-cos trial3 1
    ./scripts/traintest-featcluster_minibatch_res12_classifier-and-meta_sideout_classifier_ybranch_cifar-fs.sh 2 FFF-FFFTTT-FFF-FFF-out_conv-stage0-TTTwithconcat-train7shot-test_stop-cos trial4 2

    ./scripts/traintest-featcluster_minibatch_res12_classifier-and-meta_sideout_classifier_ybranch_cifar-fs.sh 3 FFF-FFF-FFF-FFFTTT-out_conv-stage0-TTTnoconcat-train7shot-test_stop-cos trial2 3
    ./scripts/traintest-featcluster_minibatch_res12_classifier-and-meta_sideout_classifier_ybranch_cifar-fs.sh 4 FFF-FFF-FFF-FFFTTT-out_conv-stage0-TTTnoconcat-train7shot-test_stop-cos trial3 4
    ./scripts/traintest-featcluster_minibatch_res12_classifier-and-meta_sideout_classifier_ybranch_cifar-fs.sh 5 FFF-FFF-FFF-FFFTTT-out_conv-stage0-TTTnoconcat-train7shot-test_stop-cos trial4 5

    ./scripts/traintest-featcluster_minibatch_res12_classifier-and-meta_sideout_classifier_ybranch_cifar-fs.sh 6 FFF-FFF-FFF-FFFTTT-out_conv-stage0-TTTwithconcat-train7shot-test_stop-cos trial2 6
    ./scripts/traintest-featcluster_minibatch_res12_classifier-and-meta_sideout_classifier_ybranch_cifar-fs.sh 7 FFF-FFF-FFF-FFFTTT-out_conv-stage0-TTTwithconcat-train7shot-test_stop-cos trial3 7
    ./scripts/traintest-featcluster_minibatch_res12_classifier-and-meta_sideout_classifier_ybranch_cifar-fs.sh 8 FFF-FFF-FFF-FFFTTT-out_conv-stage0-TTTwithconcat-train7shot-test_stop-cos trial4 8

    # moss106
    ./scripts/traintest-featcluster_minibatch_res12_classifier-and-meta_sideout_classifier_ybranch_cifar-fs.sh 0 FFF-FFFTTT-FFFTTT-FFFTTT-out_conv-stage0-TTTnoconcat-train7shot-test_stop-cos trial2 0
    ./scripts/traintest-featcluster_minibatch_res12_classifier-and-meta_sideout_classifier_ybranch_cifar-fs.sh 1 FFF-FFFTTT-FFFTTT-FFFTTT-out_conv-stage0-TTTnoconcat-train7shot-test_stop-cos trial3 1
    ./scripts/traintest-featcluster_minibatch_res12_classifier-and-meta_sideout_classifier_ybranch_cifar-fs.sh 2 FFF-FFFTTT-FFFTTT-FFFTTT-out_conv-stage0-TTTnoconcat-train7shot-test_stop-cos trial4 2

    ./scripts/traintest-featcluster_minibatch_res12_classifier-and-meta_sideout_classifier_ybranch_cifar-fs.sh 3 FFF-FFFTTT-FFFTTT-FFFTTT-out_conv-stage0-TTTwithconcat-train7shot-test_stop-cos trial2 3
    ./scripts/traintest-featcluster_minibatch_res12_classifier-and-meta_sideout_classifier_ybranch_cifar-fs.sh 4 FFF-FFFTTT-FFFTTT-FFFTTT-out_conv-stage0-TTTwithconcat-train7shot-test_stop-cos trial3 4
    ./scripts/traintest-featcluster_minibatch_res12_classifier-and-meta_sideout_classifier_ybranch_cifar-fs.sh 5 FFF-FFFTTT-FFFTTT-FFFTTT-out_conv-stage0-TTTwithconcat-train7shot-test_stop-cos trial4 5

    # moss101
    ./scripts/traintest-featcluster_minibatch_res12_classifier-and-meta_sideout_classifier_ybranch_cifar-fs.sh 1 FFF-FFFTTT-FFFTTT-FFFTTT-out_conv-stage0-TTTnoconcat-train7shot-test_stop-cos-sideout0.333 trial2 1
    ./scripts/traintest-featcluster_minibatch_res12_classifier-and-meta_sideout_classifier_ybranch_cifar-fs.sh 5 FFF-FFFTTT-FFFTTT-FFFTTT-out_conv-stage0-TTTnoconcat-train7shot-test_stop-cos-sideout0.333 trial3 5
    ./scripts/traintest-featcluster_minibatch_res12_classifier-and-meta_sideout_classifier_ybranch_cifar-fs.sh 6 FFF-FFFTTT-FFFTTT-FFFTTT-out_conv-stage0-TTTnoconcat-train7shot-test_stop-cos-sideout0.333 trial4 6

    ./scripts/traintest-featcluster_minibatch_res12_classifier-and-meta_sideout_classifier_ybranch_cifar-fs.sh 7 FFF-FFFTTT-FFFTTT-FFFTTT-out_conv-stage0-TTTwithconcat-train7shot-test_stop-cos-sideout0.333 trial2 7
    ./scripts/traintest-featcluster_minibatch_res12_classifier-and-meta_sideout_classifier_ybranch_cifar-fs.sh 8 FFF-FFFTTT-FFFTTT-FFFTTT-out_conv-stage0-TTTwithconcat-train7shot-test_stop-cos-sideout0.333 trial3 8
    ./scripts/traintest-featcluster_minibatch_res12_classifier-and-meta_sideout_classifier_ybranch_cifar-fs.sh 9 FFF-FFFTTT-FFFTTT-FFFTTT-out_conv-stage0-TTTwithconcat-train7shot-test_stop-cos-sideout0.333 trial4 9


    # moss106
    ./scripts/traintest-featcluster_minibatch_res12_classifier-and-meta_sideout_classifier_ybranch_cifar-fs.sh 0 FFF-FFF-FFFTTT-FFF-out_conv-stage0-TTTwithconcat-train7shot-test_stop-cos trial2 0
    ./scripts/traintest-featcluster_minibatch_res12_classifier-and-meta_sideout_classifier_ybranch_cifar-fs.sh 1 FFF-FFF-FFFTTT-FFF-out_conv-stage0-TTTwithconcat-train7shot-test_stop-cos trial3 1
    ./scripts/traintest-featcluster_minibatch_res12_classifier-and-meta_sideout_classifier_ybranch_cifar-fs.sh 2 FFF-FFF-FFFTTT-FFF-out_conv-stage0-TTTwithconcat-train7shot-test_stop-cos trial4 2

    # moss103
    ./scripts/traintest-featcluster_minibatch_res12_classifier-and-meta_sideout_classifier_ybranch_cifar-fs.sh 3 FFF-FFF-FFFTTT-FFF-out_conv-stage0-TTTnoconcat-train7shot-test_stop-cos trial2 3
    ./scripts/traintest-featcluster_minibatch_res12_classifier-and-meta_sideout_classifier_ybranch_cifar-fs.sh 4 FFF-FFF-FFFTTT-FFF-out_conv-stage0-TTTnoconcat-train7shot-test_stop-cos trial3 4
    ./scripts/traintest-featcluster_minibatch_res12_classifier-and-meta_sideout_classifier_ybranch_cifar-fs.sh 5 FFF-FFF-FFFTTT-FFF-out_conv-stage0-TTTnoconcat-train7shot-test_stop-cos trial4 5


    # Debug:
    python train_classifier-and-meta_sideout_classifier_ybranch_cifar-fs.py --gpu 0 --config ./configs/current_configs/featcluster_minibatch_res12_classifier-and-meta_sideout_classifier_ybranch_cifar-fs/FFF-FFFTTT-FFFTTT-FFFTTT-out_conv-stage0-TTTnoconcat-train7shot-test_stop-cos-sideout0.333.yaml --tag=debug --num_workers=0 --vscode_debug

    python train_classifier-and-meta_sideout_classifier_ybranch_cifar-fs.py --gpu 0 --config ./configs/current_configs/featcluster_minibatch_res12_classifier-and-meta_sideout_classifier_ybranch_cifar-fs/FFF-FFFTTT-FFFTTT-FFFTTT-out_conv-stage0-TTTwithconcat-train7shot-test_stop-cos-sideout0.333.yaml --tag=debug --num_workers=0 --vscode_debug
    ```

25. Revisit Conv-4 on mini-ImageNet - Delay feature clustering.
    ```bash
    # Note: We have two versions of PyTorch on machines, which may also cause issues.
    #       Docker should be a good solution.
    
    # Perhaps it is due to the version of PyTorch that causes the CPU utilization issue!

    # moss101 (pytorch 1.1)
    # (done) ./scripts/traintest-featcluster_minibatch_trial_delay.sh 0 FFFF-out_conv trial2
    ./scripts/traintest-featcluster_minibatch_trial_delay.sh 0 FFFF-out_conv trial3
    ./scripts/traintest-featcluster_minibatch_trial_delay.sh 1 FFFF-out_conv trial4

    # moss102 (pytorch 1.4)
    # (done) ./scripts/traintest-featcluster_minibatch_trial_delay.sh 0 FFFF-out_conv trial2-pt1.4
    ./scripts/traintest-featcluster_minibatch_trial_delay.sh 0 FFFF-out_conv-merge_conv_identity trial2
    ./scripts/traintest-featcluster_minibatch_trial_delay.sh 1 FFFF-out_conv-merge_conv_identity trial3
    ./scripts/traintest-featcluster_minibatch_trial_delay.sh 2 FFFF-out_conv-merge_conv_identity trial4

    # moss103 (pytorch 1.1)
    # (done) ./scripts/traintest-featcluster_minibatch_trial_delay.sh 0 TFFF-out_conv trial2
    ./scripts/traintest-featcluster_minibatch_trial_delay.sh 0 TFFF-out_conv trial3
    ./scripts/traintest-featcluster_minibatch_trial_delay.sh 1 TFFF-out_conv trial4

    # moss105 (pytorch 1.4)
    # (done) ./scripts/traintest-featcluster_minibatch_trial_delay.sh 0 TFFF-out_conv trial2-pt1.4
    ./scripts/traintest-featcluster_minibatch_trial_delay.sh 0 TFFF-out_conv-merge_conv_identity trial2
    ./scripts/traintest-featcluster_minibatch_trial_delay.sh 1 TFFF-out_conv-merge_conv_identity trial3
    ./scripts/traintest-featcluster_minibatch_trial_delay.sh 2 TFFF-out_conv-merge_conv_identity trial4

    ./scripts/traintest-featcluster_minibatch_trial_delay.sh 3 TFFF-out_conv-delay trial2
    ./scripts/traintest-featcluster_minibatch_trial_delay.sh 4 TFFF-out_conv-delay trial3
    ./scripts/traintest-featcluster_minibatch_trial_delay.sh 5 TFFF-out_conv-delay trial4

    ./scripts/traintest-featcluster_minibatch_trial_delay.sh 6 TTTT-out_conv-delay trial2
    ./scripts/traintest-featcluster_minibatch_trial_delay.sh 7 TTTT-out_conv-delay trial3
    
    # moss106 (pytorch 1.3.1)
    ./scripts/traintest-featcluster_minibatch_trial_delay.sh 0 TTTT-out_conv-delay trial4

    ./scripts/traintest-featcluster_minibatch_trial_delay.sh 1 TTTT-out_conv trial2
    ./scripts/traintest-featcluster_minibatch_trial_delay.sh 2 TTTT-out_conv trial3
    ./scripts/traintest-featcluster_minibatch_trial_delay.sh 3 TTTT-out_conv trial4

    ./scripts/traintest-featcluster_minibatch_trial_delay.sh 4 TTTT-out_conv-merge_conv_identity trial2
    ./scripts/traintest-featcluster_minibatch_trial_delay.sh 5 TTTT-out_conv-merge_conv_identity trial3
    ./scripts/traintest-featcluster_minibatch_trial_delay.sh 6 TTTT-out_conv-merge_conv_identity trial4


    python train_classifier.py --gpu 0 --config ./configs/current_configs/featcluster_minibatch_delay/TFFF-out_conv.yaml --tag=debug --vscode_debug --num_workers=0

    python train_classifier.py --gpu 0 --config ./configs/current_configs/featcluster_minibatch/TFFF-out_conv-no-eval.yaml

    python test_few_shot.py --shot 1 --gpu 0 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4featcluster-minibatch-delay-TFFF-out_conv_debug/max-va.pth

    python test_few_shot.py --shot 1 --gpu 3 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4featcluster-minibatch-delay-TFFF-out_conv_trial2/max-va.pth


    python test_few_shot.py --shot 1 --gpu 3 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4featcluster-minibatch-TFFF-out_conv-no-eval/epoch-last.pth

    python test_few_shot_delay.py --shot 1 --gpu 3 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4featcluster-minibatch-delay-TTTT-out_conv-delay_trial2/epoch-last.pth
    ```