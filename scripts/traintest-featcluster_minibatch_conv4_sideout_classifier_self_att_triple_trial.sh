#!/bin/bash
echo "GPU index:" $1 "Experiment:" $2 
bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_self_att.sh $1 $2 trial1
bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_self_att.sh $1 $2 trial2
bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_self_att.sh $1 $2 trial3