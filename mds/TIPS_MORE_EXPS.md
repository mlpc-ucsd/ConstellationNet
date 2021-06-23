1. More experiments.


```bash
# moss101 1-3/2-4/5-7/8-9
# moss102 2-4/5-7/8-9
# moss103 0-6/2-4/5-7/8-9
# moss105 0-1/4-5/6-7
# moss106 no nvlink

# moss106
./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_trial.sh 0 TFF-FFF-TTT-FFF-out_conv-firstTconcat trial5_fix
./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_trial.sh 1 TFF-FFF-TTT-FFF-out_conv-firstTconcat trial6_fix
./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_trial.sh 2 TFF-FFF-TTT-FFF-out_conv-firstTconcat trial7_fix

# moss101
./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_trial.sh 0 TTT-FFF-FFF-FFF-out_conv trial2_fix
./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_trial.sh 1 TTT-FFF-FFF-FFF-out_conv trial3_fix
./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_trial.sh 2 TTT-FFF-FFF-FFF-out_conv trial4_fix

./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_trial.sh 3 FFF-TTT-FFF-FFF-out_conv trial2_fix
./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_trial.sh 4 FFF-TTT-FFF-FFF-out_conv trial3_fix
./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_trial.sh 5 FFF-TTT-FFF-FFF-out_conv trial4_fix

# moss102
./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_trial.sh 0 FFF-FFF-TTT-FFF-out_conv trial2_fix
./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_trial.sh 1 FFF-FFF-TTT-FFF-out_conv trial3_fix
./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_trial.sh 2 FFF-FFF-TTT-FFF-out_conv trial4_fix

./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_trial.sh 3 FFF-FFF-FFF-TTT-out_conv trial2_fix
./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_trial.sh 4 FFF-FFF-FFF-TTT-out_conv trial3_fix
./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_trial.sh 5 FFF-FFF-FFF-TTT-out_conv trial4_fix

./scripts/traintest-featcluster_minibatch_res12_fix.sh 0 TTT-TTT-TTT-TTT-out_conv
./scripts/traintest-featcluster_minibatch_res12_fix.sh 1 TTT-TTT-TTT-FFF-out_conv
./scripts/traintest-featcluster_minibatch_res12_fix.sh 2 TTT-TTT-FFF-FFF-out_conv
./scripts/traintest-featcluster_minibatch_res12_fix.sh 3 TTT-FFF-FFF-FFF-out_conv
./scripts/traintest-featcluster_minibatch_res12_fix.sh 4 TTF-FFF-FFF-FFF-out_conv
./scripts/traintest-featcluster_minibatch_res12_fix.sh 5 TFF-FFF-FFF-FFF-out_conv
```