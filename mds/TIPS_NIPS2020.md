1. Re-run experiments with the correct implementation of the feature clustering.
   - mini-ImageNet
   ```bash
   # moss105 0-1/4-5/6-7
   ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_trial.sh 0 TFF-FFF-TTT-FFF-out_conv-firstTconcat trial2_fix
   ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_trial.sh 1 TFF-FFF-TTT-FFF-out_conv-firstTconcat trial3_fix
   ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_trial.sh 2 TFF-FFF-TTT-FFF-out_conv-firstTconcat trial4_fix
   ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_trial.sh 3 FFF-FFF-FFF-FFF-out_conv trial2_fix
   
   python train_classifier_sideout_classifier_ybranch.py --gpu 4,5 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_ybranch/TFF-FFF-TTT-FFF-out_conv-stage2-firstTconcat-train3shot-60epoch.yaml --tag=trial2_fix;

   python train_classifier_sideout_classifier_ybranch.py --gpu 6,7 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_ybranch/TFF-FFF-TTT-FFF-out_conv-stage2-firstTconcat-train3shot-60epoch.yaml --tag=trial3_fix;

   # moss101
   ./scripts/test-featcluster_minibatch_res12_sideout_classifier_ybranch.sh 0 TFF-FFF-TTT-FFF-out_conv-stage2-firstTconcat-train3shot-60epoch_trial2_fix

   ./scripts/test-featcluster_minibatch_res12_sideout_classifier_ybranch.sh 1 TFF-FFF-TTT-FFF-out_conv-stage2-firstTconcat-train3shot-60epoch_trial3_fix



   python test_few_shot_ybranch.py --shot 1 --gpu 0 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load=./save/res12featcluster-minibatch-sideout-classifier-ybranch-TFF-FFF-TTT-FFF-out_conv-stage2-firstTconcat-train3shot-60epoch_trial2_fix/max-f-va.pth --method='cos,cos' --dist_func_list='none,none' --logits_coeff_list='20,20' --disable_out_feat_flatten --sideout --feat_source_list='final,final' --branch_list='1,2'

   python test_few_shot_ybranch.py --shot 1 --gpu 1 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load=./save/res12featcluster-minibatch-sideout-classifier-ybranch-TFF-FFF-TTT-FFF-out_conv-stage2-firstTconcat-train3shot-60epoch_trial3_fix/max-f-va.pth --method='cos,cos' --dist_func_list='none,none' --logits_coeff_list='20,20' --disable_out_feat_flatten --sideout --feat_source_list='final,final' --branch_list='1,2'

   python test_few_shot_ybranch.py --shot 1 --gpu 2 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load=./save/res12featcluster-minibatch-sideout-classifier-ybranch-TFF-FFF-TTT-FFF-out_conv-stage2-firstTconcat-train3shot-60epoch_trial3/max-f-va.pth --method='cos,cos' --dist_func_list='none,none' --logits_coeff_list='20,20' --disable_out_feat_flatten --sideout --feat_source_list='final,final' --branch_list='1,2'





   python test_few_shot_ybranch.py --shot 1 --gpu 2 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load=./save/res12featcluster-minibatch-sideout-classifier-ybranch-TFF-FFF-TTT-FFF-out_conv-stage2-firstTconcat-train3shot-60epoch_trial2_fix/max-f-va.pth --method='cos,cos,cos,cos' --dist_func_list='none,none,none,none' --logits_coeff_list='20,20,20,20' --disable_out_feat_flatten --sideout --feat_source_list='final,before_avgpool,final,before_avgpool' --branch_list='1,1,2,2'

   python test_few_shot_ybranch.py --shot 1 --gpu 3 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load=./save/res12featcluster-minibatch-sideout-classifier-ybranch-TFF-FFF-TTT-FFF-out_conv-stage2-firstTconcat-train3shot-60epoch_trial3_fix/max-f-va.pth --method='cos,cos,cos,cos' --dist_func_list='none,none,none,none' --logits_coeff_list='20,20,20,20' --disable_out_feat_flatten --sideout --feat_source_list='final,before_avgpool,final,before_avgpool' --branch_list='1,1,2,2'


   python test_few_shot_ybranch.py --shot 5 --gpu 2 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load=./save/res12featcluster-minibatch-sideout-classifier-ybranch-TFF-FFF-TTT-FFF-out_conv-stage2-firstTconcat-train3shot-60epoch_trial2_fix/max-f-va.pth --method='cos,cos,cos,cos' --dist_func_list='none,none,none,none' --logits_coeff_list='20,20,20,20' --disable_out_feat_flatten --sideout --feat_source_list='final,before_avgpool,final,before_avgpool' --branch_list='1,1,2,2'

   python test_few_shot_ybranch.py --shot 5 --gpu 3 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load=./save/res12featcluster-minibatch-sideout-classifier-ybranch-TFF-FFF-TTT-FFF-out_conv-stage2-firstTconcat-train3shot-60epoch_trial3_fix/max-f-va.pth --method='cos,cos,cos,cos' --dist_func_list='none,none,none,none' --logits_coeff_list='20,20,20,20' --disable_out_feat_flatten --sideout --feat_source_list='final,before_avgpool,final,before_avgpool' --branch_list='1,1,2,2'



   python test_few_shot_ybranch.py --shot 5 --gpu 2 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load=./save/res12featcluster-minibatch-sideout-classifier-ybranch-TFF-FFF-TTT-FFF-out_conv-stage2-firstTconcat-train3shot-60epoch_trial2_fix/max-f-va.pth --method='cos,cos' --dist_func_list='none,none' --logits_coeff_list='20,20' --disable_out_feat_flatten --sideout --feat_source_list='final,before_avgpool' --branch_list='1,1'

   python test_few_shot_ybranch.py --shot 5 --gpu 3 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load=./save/res12featcluster-minibatch-sideout-classifier-ybranch-TFF-FFF-TTT-FFF-out_conv-stage2-firstTconcat-train3shot-60epoch_trial3_fix/max-f-va.pth --method='cos,cos' --dist_func_list='none,none' --logits_coeff_list='20,20' --disable_out_feat_flatten --sideout --feat_source_list='final,before_avgpool' --branch_list='1,1'


   ```
   Debug:
   ```bash
   python train_classifier_sideout_classifier_temp.py --gpu 0 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier/TFF-FFF-TTT-FFF-out_conv-firstTconcat.yaml --vscode_debug --num_workers=0
   ```

   - CIFAR-FS: Conv-4.
    ```bash
   # moss102
   ./scripts/traintest-featcluster_minibatch_trial_cifar-fs.sh 0 FFFF trial2_fix 0
   ./scripts/traintest-featcluster_minibatch_trial_cifar-fs.sh 1 FFFF trial3_fix 1
   ./scripts/traintest-featcluster_minibatch_trial_cifar-fs.sh 2 FFFF trial4_fix 2

   ./scripts/traintest-featcluster_minibatch_trial_cifar-fs.sh 3 TTTT-out_conv trial2_fix 3
   ./scripts/traintest-featcluster_minibatch_trial_cifar-fs.sh 4 TTTT-out_conv trial3_fix 4
   ./scripts/traintest-featcluster_minibatch_trial_cifar-fs.sh 5 TTTT-out_conv trial4_fix 5
   
   # Epoch95 evaluation for a wrong script.
   ./scripts/test-featcluster_minibatch_trial_cifar-fs_test_epoch95.sh 0 FFFF trial2_fix 0
   ./scripts/test-featcluster_minibatch_trial_cifar-fs_test_epoch95.sh 1 FFFF trial3_fix 1
   ./scripts/test-featcluster_minibatch_trial_cifar-fs_test_epoch95.sh 2 FFFF trial4_fix 2

   ./scripts/test-featcluster_minibatch_trial_cifar-fs_test_epoch95.sh 3 TTTT-out_conv trial2_fix 3
   ./scripts/test-featcluster_minibatch_trial_cifar-fs_test_epoch95.sh 4 TTTT-out_conv trial3_fix 4
   ./scripts/test-featcluster_minibatch_trial_cifar-fs_test_epoch95.sh 5 TTTT-out_conv trial4_fix 5
   ```

   - Fix the performance drop in the model.
   ```bash
   # moss105
   ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_trial.sh 0 TFF-FFF-TTT-FFF-out_conv-firstTconcat-old-featcluster trial2
   ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_trial.sh 1 TFF-FFF-TTT-FFF-out_conv-firstTconcat-cluster128 trial2_fix
   ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_trial.sh 2 TFF-FFF-TTT-FFF-out_conv-firstTconcat-cluster32 trial2_fix
   ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_trial.sh 3 TTT-FFF-TTT-FFF-out_conv-firstTTTconcat trial2_fix
   ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_trial.sh 4 FFF-FFF-TTT-FFF-out_conv trial2_fix
   ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_trial.sh 5 FFF-FFF-FFF-TTT-out_conv trial2_fix
   ./scripts/traintest-featcluster_minibatch_res12_trial.sh 6 TFF-FFF-FFF-FFF-out_conv 2_fix
   ./scripts/traintest-featcluster_minibatch_res12_trial.sh 7 TTT-FFF-FFF-FFF-out_conv 2_fix


   python train_classifier_sideout_classifier_ybranch.py --gpu 0,1 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_ybranch/TFF-FFF-TTT-FFF-out_conv-stage2-firstTconcat-train3shot-60epoch-128clusters.yaml --tag=trial2_fix;


   # moss103
   ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_ybranch_cifar-fs.sh 0 TFF-FFF-TTT-FFF-out_conv-stage2-firstTconcat-train7shot-test_stop-cos trial2_fix 0

   ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_ybranch_cifar-fs.sh 1 TFF-FFF-TTT-FFF-out_conv-stage2-firstTconcat-train7shot-test_stop-cos trial3_fix 1

   ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_ybranch_cifar-fs.sh 2 TFF-FFF-TTT-FFF-out_conv-stage2-firstTconcat-train7shot-test_stop-cos trial4_fix 2



   ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_ybranch_cifar-fs.sh 3 TFF-FFF-TTT-FFF-out_conv-stage2-firstTconcat-train7shot-test_stop-cos trial2_fix 3

   ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_ybranch_cifar-fs.sh 4 TFF-FFF-TTT-FFF-out_conv-stage2-firstTconcat-train7shot-test_stop-cos trial3_fix 4

   ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_ybranch_cifar-fs.sh 5 TFF-FFF-TTT-FFF-out_conv-stage2-firstTconcat-train7shot-test_stop-cos trial4_fix 5




   ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_ybranch_cifar-fs_eval_feataug.sh 3 TFF-FFF-TTT-FFF-out_conv-stage2-firstTconcat-train7shot-test_stop-cos trial2_fix 3

   ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_ybranch_cifar-fs_eval_feataug.sh 4 TFF-FFF-TTT-FFF-out_conv-stage2-firstTconcat-train7shot-test_stop-cos trial3_fix 4

   ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_ybranch_cifar-fs_eval_feataug.sh 5 TFF-FFF-TTT-FFF-out_conv-stage2-firstTconcat-train7shot-test_stop-cos trial4_fix 5
   ```


2. Conv-4 Y-branch mini-ImageNet.
   ```bash
   ./scripts/traintest-featcluster_minibatch_sideout_classifier_ybranch.sh 0 TTTT-out_conv-stage2-firstTconcat-train3shot-60epoch trial2 0

   ./scripts/traintest-featcluster_minibatch_sideout_classifier_ybranch.sh 5,7 TTTT-out_conv-stage2-firstTconcat-train3shot-60epoch trial2_2gpu 5

   ./scripts/traintest-featcluster_minibatch_sideout_classifier_ybranch.sh 8,9 TTTT-out_conv-stage2-firstTconcat-train3shot-60epoch trial3_2gpu 8

   python train_classifier_sideout_classifier_ybranch.py --gpu 0 --config ./configs/current_configs/featcluster_minibatch_sideout_classifier_ybranch/TTTT-out_conv-stage2-firstTconcat-train3shot-60epoch.yaml --tag=debug

   # moss101 1-3/2-4/5-7/8-9
   # moss102 2-4/5-7/8-9
   # moss103 0-6/2-4/5-7/8-9
   # moss105 0-1/4-5/6-7
   # moss106 no nvlink
   
   # moss102
   python train_classifier_sideout_classifier_ybranch.py --gpu 8,9 --config ./configs/current_configs/featcluster_minibatch_sideout_classifier_ybranch/TTTT-out_conv-stage2-firstTconcat-train3shot-60epoch.yaml --tag=debug2

   # moss106
   ./scripts/traintest-featcluster_minibatch_sideout_classifier_ybranch_cifar-fs.sh 2 TTTT-out_conv-stage2-firstTconcat-train3shot-60epoch trial2 2
   ./scripts/traintest-featcluster_minibatch_sideout_classifier_ybranch_cifar-fs.sh 3 TTTT-out_conv-stage2-firstTconcat-train3shot-60epoch trial3 3
   ./scripts/traintest-featcluster_minibatch_sideout_classifier_ybranch_cifar-fs.sh 4 TTTT-out_conv-stage2-firstTconcat-train3shot-60epoch trial4 4

   python train_classifier_sideout_classifier_ybranch_cifar-fs.py --gpu 1 --config ./configs/current_configs/featcluster_minibatch_sideout_classifier_ybranch_cifar-fs/TTTT-out_conv-stage2-firstTconcat-train3shot-60epoch.yaml --tag=debug
   ```

3. Ablation: Num of clusters.
   ```bash
   # moss106
   ./scripts/traintest-featcluster_minibatch_trial_delay.sh 0 TTTT-out_conv-32clusters trial2;
   ./scripts/traintest-featcluster_minibatch_trial_delay.sh 0 TTTT-out_conv-32clusters trial3;
   ./scripts/traintest-featcluster_minibatch_trial_delay.sh 0 TTTT-out_conv-32clusters trial4

   ./scripts/traintest-featcluster_minibatch_trial_delay.sh 1 TTTT-out_conv-64clusters trial2;
   ./scripts/traintest-featcluster_minibatch_trial_delay.sh 1 TTTT-out_conv-64clusters trial3;
   ./scripts/traintest-featcluster_minibatch_trial_delay.sh 1 TTTT-out_conv-64clusters trial4

   ./scripts/traintest-featcluster_minibatch_trial_delay.sh 5 TTTT-out_conv-128clusters trial2;
   ./scripts/traintest-featcluster_minibatch_trial_delay.sh 5 TTTT-out_conv-128clusters trial3;
   ./scripts/traintest-featcluster_minibatch_trial_delay.sh 5 TTTT-out_conv-128clusters trial4

   ./scripts/traintest-featcluster_minibatch_trial_delay.sh 6 TTTT-out_conv-256clusters trial2;
   ./scripts/traintest-featcluster_minibatch_trial_delay.sh 6 TTTT-out_conv-256clusters trial3;
   ./scripts/traintest-featcluster_minibatch_trial_delay.sh 6 TTTT-out_conv-256clusters trial4

   ./scripts/traintest-featcluster_minibatch_trial_delay.sh 7 TTFF-out_conv trial2;
   ./scripts/traintest-featcluster_minibatch_trial_delay.sh 7 TTFF-out_conv trial3;
   ./scripts/traintest-featcluster_minibatch_trial_delay.sh 7 TTFF-out_conv trial4

   # moss101
   ./scripts/traintest-featcluster_minibatch_trial_delay.sh 0 TTTF-out_conv trial2;
   ./scripts/traintest-featcluster_minibatch_trial_delay.sh 0 TTTF-out_conv trial3;
   ./scripts/traintest-featcluster_minibatch_trial_delay.sh 0 TTTF-out_conv trial4


   ```

   Some evaluations.
   ```bash
   python test_few_shot_ybranch.py --shot 1 --gpu 1 --config ./configs/current_configs/test_few_shot_cifar-fs_general.yaml --load=./save/conv4featcluster-minibatch-sideout-classifier-ybranch-cifar-fs-TTTT-out_conv-stage2-firstTconcat-train3shot-60epoch_trial2/max-f-va.pth --method='cos,cos,cos,cos' --dist_func_list='none,none,none,none' --logits_coeff_list='20,20,20,20' --disable_out_feat_flatten --sideout --feat_source_list='final,before_avgpool,final,before_avgpool' --branch_list='1,1,2,2'

   python test_few_shot_ybranch.py --shot 1 --gpu 2 --config ./configs/current_configs/test_few_shot_cifar-fs_general.yaml --load=./save/conv4featcluster-minibatch-sideout-classifier-ybranch-cifar-fs-TTTT-out_conv-stage2-firstTconcat-train3shot-60epoch_trial3/max-f-va.pth --method='cos,cos,cos,cos' --dist_func_list='none,none,none,none' --logits_coeff_list='20,20,20,20' --disable_out_feat_flatten --sideout --feat_source_list='final,before_avgpool,final,before_avgpool' --branch_list='1,1,2,2'

   python test_few_shot_ybranch.py --shot 1 --gpu 3 --config ./configs/current_configs/test_few_shot_cifar-fs_general.yaml --load=./save/conv4featcluster-minibatch-sideout-classifier-ybranch-cifar-fs-TTTT-out_conv-stage2-firstTconcat-train3shot-60epoch_trial4/max-f-va.pth --method='cos,cos,cos,cos' --dist_func_list='none,none,none,none' --logits_coeff_list='20,20,20,20' --disable_out_feat_flatten --sideout --feat_source_list='final,before_avgpool,final,before_avgpool' --branch_list='1,1,2,2'





   python test_few_shot_ybranch.py --shot 5 --gpu 4 --config ./configs/current_configs/test_few_shot_cifar-fs_general.yaml --load=./save/conv4featcluster-minibatch-sideout-classifier-ybranch-cifar-fs-TTTT-out_conv-stage2-firstTconcat-train3shot-60epoch_trial2/max-f-va.pth --method='cos,cos,cos,cos' --dist_func_list='none,none,none,none' --logits_coeff_list='20,20,20,20' --disable_out_feat_flatten --sideout --feat_source_list='final,before_avgpool,final,before_avgpool' --branch_list='1,1,2,2'

   python test_few_shot_ybranch.py --shot 5 --gpu 6 --config ./configs/current_configs/test_few_shot_cifar-fs_general.yaml --load=./save/conv4featcluster-minibatch-sideout-classifier-ybranch-cifar-fs-TTTT-out_conv-stage2-firstTconcat-train3shot-60epoch_trial3/max-f-va.pth --method='cos,cos,cos,cos' --dist_func_list='none,none,none,none' --logits_coeff_list='20,20,20,20' --disable_out_feat_flatten --sideout --feat_source_list='final,before_avgpool,final,before_avgpool' --branch_list='1,1,2,2'

   python test_few_shot_ybranch.py --shot 5 --gpu 1 --config ./configs/current_configs/test_few_shot_cifar-fs_general.yaml --load=./save/conv4featcluster-minibatch-sideout-classifier-ybranch-cifar-fs-TTTT-out_conv-stage2-firstTconcat-train3shot-60epoch_trial4/max-f-va.pth --method='cos,cos,cos,cos' --dist_func_list='none,none,none,none' --logits_coeff_list='20,20,20,20' --disable_out_feat_flatten --sideout --feat_source_list='final,before_avgpool,final,before_avgpool' --branch_list='1,1,2,2'







   python test_few_shot_ybranch.py --shot 1 --gpu 1 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load=./save/conv4featcluster-minibatch-sideout-classifier-ybranch-TTTT-out_conv-stage2-firstTconcat-train3shot-60epoch_trial2_2gpu/max-f-va.pth --method='cos,cos,cos,cos' --dist_func_list='none,none,none,none' --logits_coeff_list='20,20,20,20' --disable_out_feat_flatten --sideout --feat_source_list='final,before_avgpool,final,before_avgpool' --branch_list='1,1,2,2'

   python test_few_shot_ybranch.py --shot 1 --gpu 2 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load=./save/conv4featcluster-minibatch-sideout-classifier-ybranch-TTTT-out_conv-stage2-firstTconcat-train3shot-60epoch_trial3_2gpu/max-f-va.pth --method='cos,cos,cos,cos' --dist_func_list='none,none,none,none' --logits_coeff_list='20,20,20,20' --disable_out_feat_flatten --sideout --feat_source_list='final,before_avgpool,final,before_avgpool' --branch_list='1,1,2,2'




   python test_few_shot_ybranch.py --shot 5 --gpu 3 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load=./save/conv4featcluster-minibatch-sideout-classifier-ybranch-TTTT-out_conv-stage2-firstTconcat-train3shot-60epoch_trial2_2gpu/max-f-va.pth --method='cos,cos,cos,cos' --dist_func_list='none,none,none,none' --logits_coeff_list='20,20,20,20' --disable_out_feat_flatten --sideout --feat_source_list='final,before_avgpool,final,before_avgpool' --branch_list='1,1,2,2'

   python test_few_shot_ybranch.py --shot 5 --gpu 4 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load=./save/conv4featcluster-minibatch-sideout-classifier-ybranch-TTTT-out_conv-stage2-firstTconcat-train3shot-60epoch_trial3_2gpu/max-f-va.pth --method='cos,cos,cos,cos' --dist_func_list='none,none,none,none' --logits_coeff_list='20,20,20,20' --disable_out_feat_flatten --sideout --feat_source_list='final,before_avgpool,final,before_avgpool' --branch_list='1,1,2,2'


   


   ```