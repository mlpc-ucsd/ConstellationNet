#!/bin/bash
echo "Train GPU index:" $1 "Experiment:" $2 "Test GPU index:" $3
bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_ybranch_fc100_self_att.sh $1 $2 trial1 $3
bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_ybranch_fc100_self_att.sh $1 $2 trial2 $3