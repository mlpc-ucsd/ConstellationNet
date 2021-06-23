1. baseline
```bash
  
   # res12 single branch
  bash ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_self_att.sh 0 TFF-FFF-TTT-FFF-out_conv-firstTconcat-no_att-init_rp trial1

   # conv4 y-branch 
   bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_ybranch_self_att.sh 0 TTTT-out_conv-stage2-firstTconcat-train3shot-60epoch-no_att-init_rp trial1 0
```


2. (Mini-Imagenet)  

``` res12 + self_attention(sine positional encoding + post normalization + 8-head) (table3)
   bash ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_self_att.sh 5,7 TFF-FFF-TTT-FFF-self-uv_dist-att-FFF-FFF-TTT-FFF-out_conv-firstTconcat-sine_pe-pre_normF-init_rp trial1
``` 
``` res12 without constellation
   bash ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_self_att.sh 5,7 FFF-FFF-FFF-FFF-self-uv_dist-att-FFF-FFF-FFF-FFF-out_conv-firstTconcat-sine_pe-pre_normF-init_rp trial1
``` 

   


``` res12 single-branch only attention (first T clustering)
   bash ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_self_att.sh 0,6 TFF-FFF-FFF-FFF-self-uv_dist-att-FFF-FFF-TTT-FFF-out_conv-firstTconcat-sine_pe-pre_normF-init_rp trial1


```
```res12 single-branch only attention (first T not clustering)(rebuttal)

bash ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_self_att.sh 7,9 FFF-FFF-FFF-FFF-self-uv_dist-att-FFF-FFF-TTT-FFF-out_conv-firstTconcat-sine_pe-pre_normF-init_rp trial4

```

```res12 single-branch only attention +1x1 conv without sideout loss(first T not clustering)(rebuttal) 

bash ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_self_att.sh 0,2,4,6 FFF-FFF-FFF-FFF-self-uv_dist-att-FFF-FFF-TTT-FFF-out_conv-firstTconcat-1x1conv-sine_pe-pre_normF-init_rp trial1

```

```res12 single-branch only attention +1x1 conv with sideout loss(first T not clustering)(rebuttal) 

bash ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_self_att.sh 0,2,4,6 FFF-FFF-FFF-FFF-self-uv_dist-att-FFF-FFF-TTT-FFF-out_conv-firstTconcat-sideout-1x1conv-sine_pe-pre_normF-init_rp trial3

```
```res12 single-branch neg-cosine adam(rebuttal) 

bash ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_self_att.sh 0,1 TFF-FFF-TTT-FFF-self-uv_dist-att-FFF-FFF-TTT-FFF-out_conv-firstTconcat-sine_pe-pre_normF-init_rp-cosineface-adam trial1

```

``` res12 single-branch

   # different clusters num
   bash ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_self_att.sh 2 TFF-FFF-TTT-FFF-self-uv_dist-att-FFF-FFF-TTT-FFF-out_conv-firstTconcat-sine_pe-pre_normF-init_rp-clusters_128 trial1

```
```res12 y-branch + param reduced (table 1) (table3)

   bash ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_ybranch_self_att.sh 1,3 TFF-FFF-TTT-FFF-self-uv_dist-att-FFF-FFF-TTT-FFF-out_conv-stage2-firstTconcat-otherTconcat_regular-train3shot-60epoch-sine_pe-pre_normF-init_rp-param_reduced trial1 1
   
```
```res12 y-branch
  
  bash ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_ybranch_self_att.sh 5,6 TFF-FFF-TTT-FFF-self-uv_dist-att-FFF-FFF-TTT-FFF-out_conv-stage2-firstTconcat-otherTconcat_regular-train3shot-60epoch-sine_pe-pre_normF-init_rp trial1 5
```



```  conv4 ybranch + self_attention(no_pe + post normalization + 1-head)
   ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_ybranch_self_att.sh 0 TTTT-self-uv_dist-FFTT-att-out_conv-stage2-firstTconcat-train3shot-60epoch-no_pe-pre_normF-init_rp-1_head trial1 0

```

```conv4 ybranch + self_attention(sine_pe + pre_normF + 8-head) 
   
  bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_ybranch_self_att.sh 7 TTTT-self-uv_dist-FFTT-att-out_conv-stage2-firstTconcat-train3shot-60epoch-sine_pe-pre_normF-init_rp-8_head trial1 7
  
```

``` conv4 single branch sideout classifier

   bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_self_att.sh 0 TTTT-out_conv-self-uv_dist-att-FFTT-sine_pe-pre_normF-init_rp-clusters_64 trial1

```
```
 # conv4 single branch + sine pe + preNormF +8-head(table1,3)
   bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_self_att.sh 4 TTTT-out_conv-self-uv_dist-att-FFTT-sine_pe-pre_normF-init_rp trial3
   
   bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_self_att.sh 0 TTTT-out_conv-self-uv_dist-att-FFTT-sine_pe-pre_normF-init_rp trial2
 ```  
 
``` conv4 single branch only attention (rebuttal)
   # first two T
       bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_self_att.sh 8,9 TTFF-out_conv-self-uv_dist-att-FFTT-sine_pe-pre_normF-init_rp trial1
   # without first two T
    bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_self_att.sh 8,9 FFFF-out_conv-self-uv_dist-att-FFTT-sine_pe-pre_normF-init_rp trial2
  

``` conv4 single branch no pe (rebuttal)

    bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_self_att.sh 8,9 TTTT-out_conv-self-uv_dist-att-FFTT-no_pe-pre_normF-init_rp trial1

```


``` conv4 single branch sine pe amp=0.2 (rebuttal)

    bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_self_att.sh 0,1 TTTT-out_conv-self-uv_dist-att-FFTT-sine_pe-amp_0.2-pre_normF-init_rp trial1

```

``` conv4 single branch sine pe amp=0.1 (rebuttal)

    bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_self_att.sh 3,4 TTTT-out_conv-self-uv_dist-att-FFTT-sine_pe-amp_0.1-pre_normF-init_rp trial1

```

``` conv4 single branch no pe FFFF-FFTT(rebuttal)

    bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_self_att.sh 3,4 FFFF-out_conv-self-uv_dist-att-FFTT-no_pe-pre_normF-init_rp trial1

```

``` conv4 single branch sine pe 1-head (rebuttal)

    bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_self_att.sh 5,6 TTTT-out_conv-self-uv_dist-att-FFTT-sine_pe-pre_normF-init_rp-1_head trial1

```

``` conv4 single branch sine pe TTTT-FFTT att after concat (rebuttal)

    bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_self_att.sh 5,6 TTTT-out_conv-self-uv_dist-att-FFTT-sine_pe-pre_normF-init_rp-att_after_concat trial1

```
   
``` conv4 single branch similar 21 classes(rebuttal)
    
    # 0 clusters
    bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_self_att_triple_trial.sh 0 FFFF-out_conv-self-uv_dist-att-FFFF-sine_pe-pre_normF-init_rp-clusters_0-similar_21
    
    # 8 clusters
    bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_self_att_triple_trial.sh 1 TTTT-out_conv-self-uv_dist-att-FFTT-sine_pe-pre_normF-init_rp-clusters_8-similar_21
    
    # 16 clusters
    bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_self_att_triple_trial.sh 2 TTTT-out_conv-self-uv_dist-att-FFTT-sine_pe-pre_normF-init_rp-clusters_16-similar_21
    
    # 32 clusters
    bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_self_att_triple_trial.sh 3 TTTT-out_conv-self-uv_dist-att-FFTT-sine_pe-pre_normF-init_rp-clusters_32-similar_21
    
    # 64 clusters
    bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_self_att_triple_trial.sh 4 TTTT-out_conv-self-uv_dist-att-FFTT-sine_pe-pre_normF-init_rp-clusters_64-similar_21
    
    # 128 clusters
    bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_self_att_triple_trial.sh 5 TTTT-out_conv-self-uv_dist-att-FFTT-sine_pe-pre_normF-init_rp-clusters_128-similar_21

```

``` conv4 single branch random 21 classes(rebuttal)

     # 0 clusters
    bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_self_att_triple_trial.sh 6 FFFF-out_conv-self-uv_dist-att-FFFF-sine_pe-pre_normF-init_rp-clusters_0-random_21
    
    # 8 clusters
    bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_self_att_triple_trial.sh 7 TTTT-out_conv-self-uv_dist-att-FFTT-sine_pe-pre_normF-init_rp-clusters_8-random_21
    
    # 16 clusters
    bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_self_att_triple_trial.sh 8 TTTT-out_conv-self-uv_dist-att-FFTT-sine_pe-pre_normF-init_rp-clusters_16-random_21
    
    # 32 clusters
    bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_self_att_triple_trial.sh 9 TTTT-out_conv-self-uv_dist-att-FFTT-sine_pe-pre_normF-init_rp-clusters_32-random_21
    
    # 64 clusters
    bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_self_att_triple_trial.sh 0 TTTT-out_conv-self-uv_dist-att-FFTT-sine_pe-pre_normF-init_rp-clusters_64-random_21
    
    # 128 clusters
    bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_self_att_triple_trial.sh 1 TTTT-out_conv-self-uv_dist-att-FFTT-sine_pe-pre_normF-init_rp-clusters_128-random_21

```
``` conv4 +constell single branch consineface(rebuttal)
   bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_self_att.sh 1 TTTT-out_conv-self-uv_dist-att-FFTT-sine_pe-pre_normF-init_rp-cosineface_classifier trial1
     
```
``` conv4 +constell single branch consineface adam(rebuttal)
   bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_self_att.sh 2 TTTT-out_conv-self-uv_dist-att-FFTT-sine_pe-pre_normF-init_rp-cosineface_classifier-adam trial1
     
```
``` conv4 +constell single branch consineface adam-dc1e-4(rebuttal)
   bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_self_att.sh 3 TTTT-out_conv-self-uv_dist-att-FFTT-sine_pe-pre_normF-init_rp-cosineface_classifier-adam_dc1e-4 trial1
     
```

``` conv4 +constell single branch consineface long-0.01(rebuttal)
   bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_self_att.sh 4 TTTT-out_conv-self-uv_dist-att-FFTT-sine_pe-pre_normF-init_rp-cosineface_classifier-long_0.01 trial1
     
```

``` conv4 +constell single branch consineface long-0.001(rebuttal)
   bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_self_att.sh 5 TTTT-out_conv-self-uv_dist-att-FFTT-sine_pe-pre_normF-init_rp-cosineface_classifier-long_0.001 trial1
     
```


``` conv4 single branch consineface adam(rebuttal)
   bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_self_att.sh 3 FFFF-out_conv-self-uv_dist-att-FFFF-sine_pe-pre_normF-init_rp-cosineface_classifier-adam trial1    
```
``` conv4 single branch linear adam(rebuttal)
   bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_self_att.sh 4 FFFF-out_conv-self-uv_dist-att-FFFF-sine_pe-pre_normF-init_rp-adam trial1    
```


``` conv4  single branch consineface(rebuttal)
   bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_self_att.sh 0 FFFF-out_conv-self-uv_dist-att-FFFF-sine_pe-pre_normF-init_rp-cosineface_classifier trial1
     


```
``` conv4  single branch negsoftmax(rebuttal)
   bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_self_att.sh 0 FFFF-out_conv-self-uv_dist-att-FFFF-sine_pe-pre_normF-init_rp-negsoftmax_classifier trial1

```

``` conv4  single branch no constell negsoftmax scale 1.0 margin -0.01(rebuttal)
   bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_self_att.sh 0 FFFF-out_conv-self-uv_dist-att-FFFF-sine_pe-pre_normF-init_rp-negsoftmax_classifier-scale_1.0-margin_-0.01 trial1

```
``` conv4  single branch no constell negsoftmax scale 1.0 margin -0.02(rebuttal)
   bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_self_att.sh 1 FFFF-out_conv-self-uv_dist-att-FFFF-sine_pe-pre_normF-init_rp-negsoftmax_classifier-scale_1.0-margin_-0.02 trial1

```
``` conv4  single branch no constell negsoftmax scale 1.0 margin -0.03(rebuttal)
   bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_self_att.sh 2 FFFF-out_conv-self-uv_dist-att-FFFF-sine_pe-pre_normF-init_rp-negsoftmax_classifier-scale_1.0-margin_-0.03 trial1

```
``` conv4  single branch no constell negsoftmax scale 1.0 margin -0.1(rebuttal)
   bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_self_att.sh 3 FFFF-out_conv-self-uv_dist-att-FFFF-sine_pe-pre_normF-init_rp-negsoftmax_classifier-scale_1.0-margin_-0.1 trial1

```

``` conv4  single branch no constell negsoftmax scale 1.0 margin -0.15(rebuttal)
   bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_self_att_triple_trial.sh 1 FFFF-out_conv-self-uv_dist-att-FFFF-sine_pe-pre_normF-init_rp-negsoftmax_classifier-scale_1.0-margin_-0.15 

```

``` conv4  single branch no constell negsoftmax scale 1.0 margin -0.2(rebuttal)
   bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_self_att_triple_trial.sh 2 FFFF-out_conv-self-uv_dist-att-FFFF-sine_pe-pre_normF-init_rp-negsoftmax_classifier-scale_1.0-margin_-0.2

```

``` conv4  single branch no constell negsoftmax scale 1.0 margin -0.3(rebuttal)
   bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_self_att_triple_trial.sh 9 FFFF-out_conv-self-uv_dist-att-FFFF-sine_pe-pre_normF-init_rp-negsoftmax_classifier-scale_1.0-margin_-0.3

```

``` conv4  single branch no constell negsoftmax scale 1.0 margin -0.05(rebuttal)
   bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_self_att_triple_trial.sh 3 FFFF-out_conv-self-uv_dist-att-FFFF-sine_pe-pre_normF-init_rp-negsoftmax_classifier-scale_1.0-margin_-0.05

```
``` conv4  single branch no constell cosineface scale 10.0 margin -0.01(rebuttal)
   bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_self_att_triple_trial.sh 4 FFFF-out_conv-self-uv_dist-att-FFFF-sine_pe-pre_normF-init_rp-cosineface_classifier-scale_10.0-margin_-0.01

```
``` conv4  single branch no constell cosineface scale 20.0 margin -0.01(rebuttal)
   bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_self_att_triple_trial.sh 5 FFFF-out_conv-self-uv_dist-att-FFFF-sine_pe-pre_normF-init_rp-cosineface_classifier-scale_20.0-margin_-0.01

```


``` conv4  single branch no constell cosineface scale 30.0 margin -0.01(rebuttal)
   bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_self_att_triple_trial.sh 6 FFFF-out_conv-self-uv_dist-att-FFFF-sine_pe-pre_normF-init_rp-cosineface_classifier-scale_30.0-margin_-0.01

```

``` conv4  single branch no constell cosineface scale 40.0 margin -0.01(rebuttal)
   bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_self_att_triple_trial.sh 7 FFFF-out_conv-self-uv_dist-att-FFFF-sine_pe-pre_normF-init_rp-cosineface_classifier-scale_40.0-margin_-0.01

```

``` conv4  single branch no constell cosineface scale 50.0 margin -0.01(rebuttal)
   bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_self_att_triple_trial.sh 8 FFFF-out_conv-self-uv_dist-att-FFFF-sine_pe-pre_normF-init_rp-cosineface_classifier-scale_50.0-margin_-0.01

```

``` conv4  single branch no constell cosineface scale 10.0 margin -0.02(rebuttal)
   bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_self_att.sh 0 FFFF-out_conv-self-uv_dist-att-FFFF-sine_pe-pre_normF-init_rp-cosineface_classifier-scale_10.0-margin_-0.02 trial1

```
``` conv4  single branch no constell cosineface scale 20.0 margin -0.02(rebuttal)
   bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_self_att.sh 0 FFFF-out_conv-self-uv_dist-att-FFFF-sine_pe-pre_normF-init_rp-cosineface_classifier-scale_20.0-margin_-0.02 trial1

```


``` conv4  single branch no constell cosineface scale 30.0 margin -0.02(rebuttal)
   bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_self_att.sh 1 FFFF-out_conv-self-uv_dist-att-FFFF-sine_pe-pre_normF-init_rp-cosineface_classifier-scale_30.0-margin_-0.02 trial1

```

``` conv4  single branch no constell cosineface scale 40.0 margin -0.02(rebuttal)
   bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_self_att.sh 2 FFFF-out_conv-self-uv_dist-att-FFFF-sine_pe-pre_normF-init_rp-cosineface_classifier-scale_40.0-margin_-0.02 trial1

```

``` conv4  single branch no constell cosineface scale 50.0 margin -0.02(rebuttal)
   bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_self_att.sh 3 FFFF-out_conv-self-uv_dist-att-FFFF-sine_pe-pre_normF-init_rp-cosineface_classifier-scale_50.0-margin_-0.02 trial1

```





``` conv4  single branch no constell negsoftmax scale 10.0 margin -0.01(rebuttal)
   bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_self_att.sh 4 FFFF-out_conv-self-uv_dist-att-FFFF-sine_pe-pre_normF-init_rp-negsoftmax_classifier-scale_10.0-margin_-0.01 trial1

```

``` conv4  single branch no constell cosineface scale 1.0 margin -0.01(rebuttal)
   bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_self_att.sh 5 FFFF-out_conv-self-uv_dist-att-FFFF-sine_pe-pre_normF-init_rp-cosineface_classifier-scale_1.0-margin_-0.01 trial1

```

``` conv4  single branch + constell negsoftmax scale 1.0 margin -0.01(rebuttal)
   bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_self_att_triple_trial.sh 4 TTTT-out_conv-self-uv_dist-att-FFTT-sine_pe-pre_normF-init_rp-negsoftmax_classifier-scale_1.0-margin_-0.01

```
``` conv4  single branch + constell cosineface scale 10.0 margin -0.01(rebuttal)
   bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_self_att_triple_trial.sh 1 TTTT-out_conv-self-uv_dist-att-FFTT-sine_pe-pre_normF-init_rp-cosineface_classifier-scale_10.0-margin_-0.01

```
``` conv4  single branch + constell cosineface scale 5.0 margin -0.02(rebuttal)
   bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_self_att_triple_trial.sh 1 TTTT-out_conv-self-uv_dist-att-FFTT-sine_pe-pre_normF-init_rp-cosineface_classifier-scale_5.0-margin_-0.02

```

``` conv4  single branch + constell cosineface scale 5.0 margin -0.01(rebuttal)
   bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_self_att_triple_trial.sh 0 TTTT-out_conv-self-uv_dist-att-FFTT-sine_pe-pre_normF-init_rp-cosineface_classifier-scale_5.0-margin_-0.01

```

``` conv4  single branch + constell cosineface scale 1.0 margin -0.4(rebuttal)
   bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_self_att_triple_trial.sh 0 TTTT-out_conv-self-uv_dist-att-FFTT-sine_pe-pre_normF-init_rp-negsoftmax_classifier-scale_1.0-margin_-0.4

```

``` conv4  single branch + constell cosineface scale 2.0 margin -0.3(rebuttal)
   bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_self_att_triple_trial.sh 2 TTTT-out_conv-self-uv_dist-att-FFTT-sine_pe-pre_normF-init_rp-negsoftmax_classifier-scale_2.0-margin_-0.3

```

``` conv4  single branch + constell cosineface scale 2.0 margin -0.5(rebuttal)
   bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_self_att_triple_trial.sh 3 TTTT-out_conv-self-uv_dist-att-FFTT-sine_pe-pre_normF-init_rp-negsoftmax_classifier-scale_2.0-margin_-0.5

```

``` conv4  single branch + constell cosineface scale 2.0 margin -0.4(rebuttal)
   bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_self_att_triple_trial.sh 4 TTTT-out_conv-self-uv_dist-att-FFTT-sine_pe-pre_normF-init_rp-negsoftmax_classifier-scale_2.0-margin_-0.4

```



``` conv4  single branch + constell cosineface scale 1.0 margin -0.5(rebuttal)
   bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_self_att_triple_trial.sh 1 TTTT-out_conv-self-uv_dist-att-FFTT-sine_pe-pre_normF-init_rp-negsoftmax_classifier-scale_1.0-margin_-0.5

```

``` conv4  single branch + constell cosineface scale 1.0 margin -0.6(rebuttal)
   bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_self_att_triple_trial.sh 5 TTTT-out_conv-self-uv_dist-att-FFTT-sine_pe-pre_normF-init_rp-negsoftmax_classifier-scale_1.0-margin_-0.6

```

``` conv4  single branch + constell cosineface scale 1.0 margin -0.7(rebuttal)
   bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_self_att_triple_trial.sh 6 TTTT-out_conv-self-uv_dist-att-FFTT-sine_pe-pre_normF-init_rp-negsoftmax_classifier-scale_1.0-margin_-0.7

```

``` conv4  single branch + constell cosineface scale 1.0 margin -0.8(rebuttal)
   bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_self_att_triple_trial.sh 0 TTTT-out_conv-self-uv_dist-att-FFTT-sine_pe-pre_normF-init_rp-negsoftmax_classifier-scale_1.0-margin_-0.8

```

``` conv4  single branch + constell cosineface scale 1.0 margin -0.9(rebuttal)
   bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_self_att_triple_trial.sh 1 TTTT-out_conv-self-uv_dist-att-FFTT-sine_pe-pre_normF-init_rp-negsoftmax_classifier-scale_1.0-margin_-0.9

```

``` conv4  single branch + constell cosineface scale 1.0 margin -1.0(rebuttal)
   bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_self_att_triple_trial.sh 2 TTTT-out_conv-self-uv_dist-att-FFTT-sine_pe-pre_normF-init_rp-negsoftmax_classifier-scale_1.0-margin_-1.0

```


``` conv4  single branch + constell cosineface scale 10.0 margin -0.02(rebuttal)
   bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_self_att_triple_trial.sh 3 TTTT-out_conv-self-uv_dist-att-FFTT-sine_pe-pre_normF-init_rp-cosineface_classifier-scale_10.0-margin_-0.02

```

``` conv4  single branch + constell cosineface scale 20.0 margin -0.02(rebuttal)
   bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_self_att_triple_trial.sh 3 TTTT-out_conv-self-uv_dist-att-FFTT-sine_pe-pre_normF-init_rp-cosineface_classifier-scale_20.0-margin_-0.02

```

``` conv4  single branch + constell negsoftmax scale 10.0 margin -0.1(rebuttal)
   bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_self_att_triple_trial.sh 3 TTTT-out_conv-self-uv_dist-att-FFTT-sine_pe-pre_normF-init_rp-negsoftmax_classifier-scale_10.0-margin_-0.1

```

``` conv4  single branch + constell negsoftmax scale 1.0 margin -0.3(rebuttal)
   bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_self_att_triple_trial.sh 3 TTTT-out_conv-self-uv_dist-att-FFTT-sine_pe-pre_normF-init_rp-negsoftmax_classifier-scale_1.0-margin_-0.3

```



``` conv4  single branch + constell negsoftmax scale 1.0 margin -0.2(rebuttal)
   bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_self_att_triple_trial.sh 2 TTTT-out_conv-self-uv_dist-att-FFTT-sine_pe-pre_normF-init_rp-negsoftmax_classifier-scale_1.0-margin_-0.2

```

``` conv4  single branch + constell negsoftmax scale 1.0 margin 0.1(rebuttal)
   bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_self_att_triple_trial.sh 5 TTTT-out_conv-self-uv_dist-att-FFTT-sine_pe-pre_normF-init_rp-negsoftmax_classifier-scale_1.0-margin_0.1

```

``` conv4  single branch + constell negsoftmax scale 1.0 margin 0.01(rebuttal)
   bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_self_att_triple_trial.sh 6 TTTT-out_conv-self-uv_dist-att-FFTT-sine_pe-pre_normF-init_rp-negsoftmax_classifier-scale_1.0-margin_0.01

```

``` conv4  single branch no constell negsoftmax scale 1.0 margin -0.5(rebuttal)
   bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_self_att.sh 2 FFFF-out_conv-self-uv_dist-att-FFFF-sine_pe-pre_normF-init_rp-negsoftmax_classifier-scale_1.0-margin_-0.5 trial3 

```




``` conv4 single branch ablation (Figure 2)

   # vanilla
   bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_self_att.sh 4 TTTT-out_conv-self-uv_dist-att-FFTT-sine_pe-pre_normF-init_rp trial4

   # cluster type input
   bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_self_att.sh 1 TTTT-input-self-uv_dist-att-FFTT-sine_pe-pre_normF-init_rp trial1
   
   # cluster type input 3x3
   bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_self_att.sh 1 TTTT-input3x3-self-uv_dist-att-FFTT-sine_pe-pre_normF-init_rp trial1
   


   # 1-head
   bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_self_att.sh 0 TTTT-out_conv-self-uv_dist-att-FFTT-sine_pe-pre_normF-init_rp-1_head trial1
   
   bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_self_att.sh 0 TTTT-out_conv-self-uv_dist-att-FFTT-sine_pe-pre_normF-init_rp-1_head trial3
   
   # 2-head
   bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_self_att.sh 1 TTTT-out_conv-self-uv_dist-att-FFTT-sine_pe-pre_normF-init_rp-2_head trial1
   
   bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_self_att.sh 1 TTTT-out_conv-self-uv_dist-att-FFTT-sine_pe-pre_normF-init_rp-2_head trial3
   
   #4-head
   bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_self_att.sh 2 TTTT-out_conv-self-uv_dist-att-FFTT-sine_pe-pre_normF-init_rp-4_head trial1
   
   bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_self_att.sh 2 TTTT-out_conv-self-uv_dist-att-FFTT-sine_pe-pre_normF-init_rp-4_head trial3
   
   #16-head
   bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_self_att.sh 3 TTTT-out_conv-self-uv_dist-att-FFTT-sine_pe-pre_normF-init_rp-16_head trial1
   
   bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_self_att.sh 6 TTTT-out_conv-self-uv_dist-att-FFTT-sine_pe-pre_normF-init_rp-16_head trial3
   
   
   
   # TFFF(FFFF)
   bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_self_att.sh 1 TFFF-out_conv-self-uv_dist-att-FFFF-sine_pe-pre_normF-init_rp trial1
   
   #TTFF(FFFF)
   bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_self_att.sh 1 TTFF-out_conv-self-uv_dist-att-FFFF-sine_pe-pre_normF-init_rp trial1
   
   #TTTF(FFTF)
   bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_self_att.sh 1 TTTF-out_conv-self-uv_dist-att-FFTF-sine_pe-pre_normF-init_rp trial1
   
   #FFFT(FFFT)
   bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_self_att.sh 1 FFFT-out_conv-self-uv_dist-att-FFFT-sine_pe-pre_normF-init_rp trial1
   
   #FFTT(FFTT)
   bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_self_att.sh 0 FFTT-out_conv-self-uv_dist-att-FFTT-sine_pe-pre_normF-init_rp trial1
   
   #FTTT(FFTT)
   bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_self_att.sh 2 FTTT-out_conv-self-uv_dist-att-FFTT-sine_pe-pre_normF-init_rp trial1
   
   #FFFF(FFFF)
   bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_self_att.sh 3 FFFF-out_conv-self-uv_dist-att-FFFF-sine_pe-pre_normF-init_rp trial1
   
   # 1x1 conv replace clustering (rebuttal)
   bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_self_att.sh 0,2,4,6 FFFF-out_conv-self-uv_dist-att-FFTT-1x1conv-sine_pe-pre_normF-init_rp trial1

  
   
   # 8 clusters
   bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_self_att.sh 0 TTTT-out_conv-self-uv_dist-att-FFTT-sine_pe-pre_normF-init_rp-clusters_8 trial1
   
   bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_self_att.sh 1 TTTT-out_conv-self-uv_dist-att-FFTT-sine_pe-pre_normF-init_rp-clusters_8 trial2
   
   # 16 clusters
   bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_self_att.sh 2 TTTT-out_conv-self-uv_dist-att-FFTT-sine_pe-pre_normF-init_rp-clusters_16 trial1
   
   bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_self_att.sh 3 TTTT-out_conv-self-uv_dist-att-FFTT-sine_pe-pre_normF-init_rp-clusters_16 trial2
   
   # 32 clusters
   bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_self_att.sh 0 TTTT-out_conv-self-uv_dist-att-FFTT-sine_pe-pre_normF-init_rp-clusters_32 trial1
   
   bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_self_att.sh 5 TTTT-out_conv-self-uv_dist-att-FFTT-sine_pe-pre_normF-init_rp-clusters_32 trial2
   
   # 128 clusters
   bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_self_att.sh 2 TTTT-out_conv-self-uv_dist-att-FFTT-sine_pe-pre_normF-init_rp-clusters_128 trial1
   
   bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_self_att.sh 7 TTTT-out_conv-self-uv_dist-att-FFTT-sine_pe-pre_normF-init_rp-clusters_128 trial2
   
   
   # learned pe + preNormF 
   bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_self_att.sh 0 TTTT-out_conv-self-uv_dist-att-FFTT-learned_pe-pre_normF-init_rp trial1
   
   # no pe + preNormF 
   bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_self_att.sh 1 TTTT-out_conv-self-uv_dist-att-FFTT-no_pe-pre_normF-init_rp trial1
   
   # sine pe + preNormT 
   bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_self_att.sh 2 TTTT-out_conv-self-uv_dist-att-FFTT-sine_pe-pre_normT-init_rp trial1
   
   # learned pe + preNormT + triple trial
   bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_self_att.sh 3 TTTT-out_conv-self-uv_dist-att-FFTT-learned_pe-pre_normT-init_rp
   
   # no pe + preNormT + triple trial
   bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_self_att.sh 4 TTTT-out_conv-self-uv_dist-att-FFTT-no_pe-pre_normT-init_rp
   
   
```

``` conv4 ybranch 2-att layer

   bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_ybranch_self_att.sh 4 TTTT-self-uv_dist-FFTT-att-out_conv-stage2-firstTconcat-train3shot-60epoch-sine_pe-pre_normF-init_rp-8_head-2_att_layer

```


```conv4 ybranch + ablation on  num clusters
  
  # 48
  bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_ybranch_self_att.sh 6 TTTT-self-uv_dist-FFTT-att-out_conv-stage2-firstTconcat-train3shot-60epoch-sine_pe-pre_normF-init_rp-8_head-clusters_48 trial1 6
  
  # 96
  bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_ybranch_self_att.sh 7 TTTT-self-uv_dist-FFTT-att-out_conv-stage2-firstTconcat-train3shot-60epoch-sine_pe-pre_normF-init_rp-8_head-clusters_96 trial1 7

```

```conv4 ybranch + param reduced (table 1)(table3)
  
  # 64 64 42 42 
  bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_ybranch_self_att.sh 1,3 TTTT-self-uv_dist-FFTT-att-out_conv-stage2-firstTconcat-train3shot-60epoch-sine_pe-pre_normF-init_rp-8_head-param_reduced trial1
  
  bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_ybranch_self_att.sh 2,3 TTTT-self-uv_dist-FFTT-att-out_conv-stage2-firstTconcat-train3shot-60epoch-sine_pe-pre_normF-init_rp-8_head-param_reduced trial3 2
``` 


3. (Cifar-fs)
``` res 12 y-branch
   
   # full version
   bash ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_ybranch_cifar-fs_self_att.sh 3 TFF-FFF-TTT-FFF-self-uv_dist-att-FFF-FFF-TTT-FFF-out_conv-stage2-firstTconcat-train7shot-test_stop-cos-sine_pe-pre_normF-init_rp trial1 3
   
   
   
   # train 60 epochs no sideout
   
   bash ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_ybranch_cifar-fs_self_att.sh 2 TFF-FFF-TTT-FFF-self-uv_dist-att-FFF-FFF-TTT-FFF-out_conv-stage2-onlyconcat-train7shot-test_stop-cos-sine_pe-pre_normF-init_rp trial1 2
   
   # train 60 epochs no sideout param-reduced
   bash ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_ybranch_cifar-fs_self_att.sh 5 TFF-FFF-TTT-FFF-self-uv_dist-att-FFF-FFF-TTT-FFF-out_conv-stage2-onlyconcat-train7shot-test_stop-cos-sine_pe-pre_normF-init_rp-param_reduced trial1 5
   
   
   # train 40 epochs no sideout
   bash ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_ybranch_cifar-fs_self_att.sh 0 TFF-FFF-TTT-FFF-self-uv_dist-att-FFF-FFF-TTT-FFF-out_conv-stage2-onlyconcat-train7shot-test_stop-cos-sine_pe-pre_normF-init_rp-train40epochs trial1 0
   
   # train 40 epochs 
   bash ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_ybranch_cifar-fs_self_att.sh 1 TFF-FFF-TTT-FFF-self-uv_dist-att-FFF-FFF-TTT-FFF-out_conv-stage2-firstTconcat-train7shot-test_stop-cos-sine_pe-pre_normF-init_rp-train40epochs trial1 1
```   
   
```   
   # param reduced (table2)
    bash ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_ybranch_cifar-fs_self_att.sh 1 TFF-FFF-TTT-FFF-self-uv_dist-att-FFF-FFF-TTT-FFF-out_conv-stage2-firstTconcat-train7shot-test_stop-cos-sine_pe-pre_normF-init_rp-param_reduced trial1 1
   
  
   
   # no sideout classifier
   bash ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_ybranch_cifar-fs_self_att.sh 6 TFF-FFF-TTT-FFF-self-uv_dist-att-FFF-FFF-TTT-FFF-out_conv-stage2-onlyconcat-train7shot-test_stop-cos-sine_pe-pre_normF-init_rp trial1 6
   

```

``` conv4 baseline

bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_ybranch_cifar-fs_self_att.sh 2 TTTT-no_att-out_conv-stage2-firstTconcat-train3shot-60epoch-init_rp trial2 2

```

``` conv4 ybranch param reduced (table2)

  bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_ybranch_cifar-fs_self_att.sh 1 TTTT-self-uv_dist-FFTT-att-out_conv-stage2-firstTconcat-train3shot-60epoch-sine_pe-pre_normF-init_rp-8_head-param_reduced trial1
  

```

```conv4 ybranch ablation

    # 1-head + no pe + prenormF
    bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_ybranch_cifar-fs_self_att.sh 4 TTTT-self-uv_dist-FFTT-att-out_conv-stage2-firstTconcat-train3shot-60epoch-no_pe-pre_normF-init_rp-1_head trial1 

    # 8-head + sine pe + prenormF
    bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_ybranch_cifar-fs_self_att.sh 2 TTTT-self-uv_dist-FFTT-att-out_conv-stage2-firstTconcat-train3shot-60epoch-sine_pe-pre_normF-init_rp-8_head trial3 

    # 8-head + no pe + prenormT
    bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_ybranch_cifar-fs_self_att.sh 4 TTTT-self-uv_dist-FFTT-att-out_conv-stage2-firstTconcat-train3shot-60epoch-no_pe-pre_normT-init_rp-8_head trial2 

    # 8-head + sine pe + prenormT
    bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_ybranch_cifar-fs_self_att.sh 2 TTTT-self-uv_dist-FFTT-att-out_conv-stage2-firstTconcat-train3shot-60epoch-sine_pe-pre_normT-init_rp-8_head trial1 

    # 8-head + sine pe + prenormT 
    bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_ybranch_cifar-fs_self_att.sh 6 TTTT-self-uv_dist-FFTT-att-out_conv-stage2-firstTconcat-train3shot-60epoch-sine_pe-pre_normT-init_rp-8_head trial1


```

4. fc100
``` fc100 res12 single branch 
   bash ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_fc100_self_att.sh 6 TFF-FFF-TTT-FFF-self-uv_dist-att-FFF-FFF-TTT-FFF-out_conv-test_stop-firstTconcat-sine_pe-pre_normF-init_rp trial1 6
   bash ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_fc100_self_att.sh 6 TFF-FFF-TTT-FFF-self-uv_dist-att-FFF-FFF-TTT-FFF-out_conv-test_stop-firstTconcat-sine_pe-pre_normF-init_rp-4_head trial1 6
   
   # 128 clusters
   bash ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_fc100_self_att.sh 2 TFF-FFF-TTT-FFF-self-uv_dist-att-FFF-FFF-TTT-FFF-out_conv-test_stop-firstTconcat-sine_pe-pre_normF-init_rp-clusters_128 trial1 2
   
   #only concat(table 2)
   bash ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_fc100_self_att.sh 1 TFF-FFF-TTT-FFF-self-uv_dist-att-FFF-FFF-TTT-FFF-out_conv-test_stop-onlyconcat-sine_pe-pre_normF-init_rp trial1 1
   
   # only first T
   bash ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_fc100_self_att.sh 6,7 TFF-FFF-FFF-FFF-self-uv_dist-att-TFF-FFF-FFF-FFF-out_conv-test_stop-firstTconcat-sine_pe-pre_normF-init_rp trial1 6
   
   # only concat 128 clusters
   bash ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_fc100_self_att.sh 0 TFF-FFF-TTT-FFF-self-uv_dist-att-FFF-FFF-TTT-FFF-out_conv-test_stop-onlyconcat-sine_pe-pre_normF-init_rp-clusters_128 0
   
   # only concat 32 clusters
   bash ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_fc100_self_att.sh 1 TFF-FFF-TTT-FFF-self-uv_dist-att-FFF-FFF-TTT-FFF-out_conv-test_stop-onlyconcat-sine_pe-pre_normF-init_rp-clusters_32 1
   
   # only concat 16 clusters
   bash ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_fc100_self_att.sh 2 TFF-FFF-TTT-FFF-self-uv_dist-att-FFF-FFF-TTT-FFF-out_conv-test_stop-onlyconcat-sine_pe-pre_normF-init_rp-clusters_16 2
   
   

```
```fc100 conv4 ybranch
   
   # full version
   bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_ybranch_fc100_self_att.sh 5 TTTT-self-uv_dist-FFTT-att-out_conv-stage2-firstTconcat-train3shot-60epoch-sine_pe-pre_normF-init_rp-8_head 5
   
   # param reduced
   bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_ybranch_fc100_self_att.sh 7 TTTT-self-uv_dist-FFTT-att-out_conv-stage2-firstTconcat-train3shot-60epoch-sine_pe-pre_normF-init_rp-8_head-param_reduced 7


```


``` fc100 res12 ybranch
  # full version (moss5)
  bash ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_ybranch_fc100_self_att.sh 0,1 TFF-FFF-TTT-FFF-self-uv_dist-att-FFF-FFF-TTT-FFF-out_conv-stage2-onlyconcat-train3shot-60epoch-sine_pe-pre_normF-init_rp trial1 0
  
  
  # full version trial 2,3
  bash ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_ybranch_fc100_self_att.sh 0,1 TFF-FFF-TTT-FFF-self-uv_dist-att-FFF-FFF-TTT-FFF-out_conv-stage2-onlyconcat-train3shot-60epoch-sine_pe-pre_normF-init_rp 0
  
  
  # param reduced trial 1
  bash ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_ybranch_fc100_self_att.sh 3 TFF-FFF-TTT-FFF-self-uv_dist-att-FFF-FFF-TTT-FFF-out_conv-stage2-onlyconcat-train3shot-60epoch-sine_pe-pre_normF-init_rp-param_reduced trial1 3
  
  # param reduced trial 2,3
  bash ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_ybranch_fc100_self_att.sh 2 TFF-FFF-TTT-FFF-self-uv_dist-att-FFF-FFF-TTT-FFF-out_conv-stage2-onlyconcat-train3shot-60epoch-sine_pe-pre_normF-init_rp-param_reduced 2



```
```
   # y-branch param reduced cluster number ablation (rebuttal)
  
   # 8-clusters
   bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_ybranch_fc100_self_att_double_trial.sh 0 TTTT-self-uv_dist-FFTT-att-out_conv-stage2-firstTconcat-train7shot-60epoch-sine_pe-pre_normF-init_rp-8_head-clusters_8-param_reduced 0
   # 16-clusters
   bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_ybranch_fc100_self_att_double_trial.sh 1 TTTT-self-uv_dist-FFTT-att-out_conv-stage2-firstTconcat-train7shot-60epoch-sine_pe-pre_normF-init_rp-8_head-clusters_16-param_reduced 1
   # 32-clusters
   bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_ybranch_fc100_self_att_double_trial.sh 2 TTTT-self-uv_dist-FFTT-att-out_conv-stage2-firstTconcat-train7shot-60epoch-sine_pe-pre_normF-init_rp-8_head-clusters_32-param_reduced 2
   
   # 128-clusters
   bash ./scripts/traintest-featcluster_minibatch_conv4_sideout_classifier_ybranch_fc100_self_att_double_trial.sh 3 TTTT-self-uv_dist-FFTT-att-out_conv-stage2-firstTconcat-train7shot-60epoch-sine_pe-pre_normF-init_rp-8_head-clusters_128-param_reduced 3
   
   
   
```
``` fc100 res12 single branch(rebuttal)
   # reproduce default (table2)
   bash ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_fc100_self_att.sh 0 TFF-FFF-TTT-FFF-self-uv_dist-att-FFF-FFF-TTT-FFF-out_conv-test_stop-onlyconcat-sine_pe-pre_normF-init_rp-clusters_64 trial1 0
   
   
   # similar 30 classes 
   
   # cluster num 0
   bash ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_fc100_self_att.sh 4 FFF-FFF-FFF-FFF-self-uv_dist-att-FFF-FFF-FFF-FFF-out_conv-test_stop-onlyconcat-sine_pe-pre_normF-init_rp-clusters_0-similar_30 trial1 4
   bash ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_fc100_self_att.sh 7 FFF-FFF-FFF-FFF-self-uv_dist-att-FFF-FFF-FFF-FFF-out_conv-test_stop-onlyconcat-sine_pe-pre_normF-init_rp-clusters_0-similar_30 trial2 7
   bash ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_fc100_self_att.sh 8 FFF-FFF-FFF-FFF-self-uv_dist-att-FFF-FFF-FFF-FFF-out_conv-test_stop-onlyconcat-sine_pe-pre_normF-init_rp-clusters_0-similar_30 trial3 8
   #cluster num 8
   bash ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_fc100_self_att_triple_trial.sh 0 TFF-FFF-TTT-FFF-self-uv_dist-att-FFF-FFF-TTT-FFF-out_conv-test_stop-onlyconcat-sine_pe-pre_normF-init_rp-clusters_8-similar_30 0
   
   #cluster num 16
   bash ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_fc100_self_att_triple_trial.sh 1 TFF-FFF-TTT-FFF-self-uv_dist-att-FFF-FFF-TTT-FFF-out_conv-test_stop-onlyconcat-sine_pe-pre_normF-init_rp-clusters_16-similar_30 1
   
   #cluster num 32
   bash ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_fc100_self_att_triple_trial.sh 2 TFF-FFF-TTT-FFF-self-uv_dist-att-FFF-FFF-TTT-FFF-out_conv-test_stop-onlyconcat-sine_pe-pre_normF-init_rp-clusters_32-similar_30 2
   
   #cluster num 64
   bash ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_fc100_self_att_triple_trial.sh 3 TFF-FFF-TTT-FFF-self-uv_dist-att-FFF-FFF-TTT-FFF-out_conv-test_stop-onlyconcat-sine_pe-pre_normF-init_rp-clusters_64-similar_30 3
   
   #cluster num 128
   bash ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_fc100_self_att_triple_trial.sh 4 TFF-FFF-TTT-FFF-self-uv_dist-att-FFF-FFF-TTT-FFF-out_conv-test_stop-onlyconcat-sine_pe-pre_normF-init_rp-clusters_128-similar_30 4
   
   # random 30 classes 
   
   # cluster num 0
   bash ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_fc100_self_att.sh 0 FFF-FFF-FFF-FFF-self-uv_dist-att-FFF-FFF-FFF-FFF-out_conv-test_stop-onlyconcat-sine_pe-pre_normF-init_rp-clusters_0-random_30 trial1 0
   bash ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_fc100_self_att.sh 1 FFF-FFF-FFF-FFF-self-uv_dist-att-FFF-FFF-FFF-FFF-out_conv-test_stop-onlyconcat-sine_pe-pre_normF-init_rp-clusters_0-random_30 trial2 1
   bash ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_fc100_self_att.sh 3 FFF-FFF-FFF-FFF-self-uv_dist-att-FFF-FFF-FFF-FFF-out_conv-test_stop-onlyconcat-sine_pe-pre_normF-init_rp-clusters_0-random_30 trial3 3
   
   #cluster num 8
   bash ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_fc100_self_att_triple_trial.sh 9 TFF-FFF-TTT-FFF-self-uv_dist-att-FFF-FFF-TTT-FFF-out_conv-test_stop-onlyconcat-sine_pe-pre_normF-init_rp-clusters_8-random_30 9
   
   
   #cluster num 16
   bash ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_fc100_self_att_triple_trial.sh 8 TFF-FFF-TTT-FFF-self-uv_dist-att-FFF-FFF-TTT-FFF-out_conv-test_stop-onlyconcat-sine_pe-pre_normF-init_rp-clusters_16-random_30 8
   
   #cluster num 32
   bash ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_fc100_self_att_triple_trial.sh 7 TFF-FFF-TTT-FFF-self-uv_dist-att-FFF-FFF-TTT-FFF-out_conv-test_stop-onlyconcat-sine_pe-pre_normF-init_rp-clusters_32-random_30 7
   
   #cluster num 64
   bash ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_fc100_self_att_triple_trial.sh 6 TFF-FFF-TTT-FFF-self-uv_dist-att-FFF-FFF-TTT-FFF-out_conv-test_stop-onlyconcat-sine_pe-pre_normF-init_rp-clusters_64-random_30 6
   
   #cluster num 128
   bash ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier_fc100_self_att_triple_trial.sh 5 TFF-FFF-TTT-FFF-self-uv_dist-att-FFF-FFF-TTT-FFF-out_conv-test_stop-onlyconcat-sine_pe-pre_normF-init_rp-clusters_128-random_30 5
   

```

