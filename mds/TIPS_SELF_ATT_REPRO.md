-Mini-ImageNet


1. baseline
```
  
   # res12 single branch
  bash ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_self_att_trial.sh 0,6,2,4 TFF-FFF-TTT-FFF-out_conv-firstTconcat-no_att-init_rp trial1_new
  


   # conv4 y-branch 
   bash ./scripts/traintest-featcluster_minibatch_sideout_classifier_ybranch_self_att.sh 4 TTTT-out_conv-stage2-firstTconcat-train3shot-60epoch-no_att-init_rp trial1_new 4
```




2. Res12, Conv4

```
   #res12 + single branch + self_attention(sine positional encoding + post normalization + 8-head)
   bash ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_self_att_trial.sh 0,6,2,4 TFF-FFF-TTT-FFF-self-uv_dist-att-FFF-FFF-TTT-FFF-out_conv-firstTconcat-sine_pe-pre_normF-init_rp trial1_new
   

 
   #conv4+ y_branch + self_attention(no positional encoding + post normalization + 1-head)
   bash ./scripts/traintest-featcluster_minibatch_sideout_classifier_ybranch_self_att.sh 2 TTTT-self-uv_dist-FFTT-att-out_conv-stage2-firstTconcat-train3shot-60epoch-no_pe-pre_normF-init_rp-1_head trial1_new 2

```




  

 
