1. original
```
  # image features
        python feat_extract.py --gpu 3 --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/conv4featcluster-minibatch-TTTT-out_conv_trial2/max-va.pth --split train --sideout --feat_source="3.conv" --save_path=/home/shawn/weijian/data/conv4featcluster-minibatch-TTTT-out_conv_trial2

        python feat_extract.py --gpu 2 --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/conv4featcluster-minibatch-TTTT-out_conv_trial2/max-va.pth --split train --sideout --feat_source="2.conv" --save_path=/home/shawn/weijian/data/conv4featcluster-minibatch-TTTT-out_conv_trial2

        python feat_extract.py --gpu 1 --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/conv4featcluster-minibatch-TTTT-out_conv_trial2/max-va.pth --split train --sideout --feat_source="1.conv" --save_path=/home/shawn/weijian/data/conv4featcluster-minibatch-TTTT-out_conv_trial2

        python feat_extract.py --gpu 0 --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/conv4featcluster-minibatch-TTTT-out_conv_trial2/max-va.pth --split train --sideout --feat_source="0.conv" --save_path=/home/shawn/weijian/data/conv4featcluster-minibatch-TTTT-out_conv_trial2
  
```

```
  # cluster centers
  python weight_extract.py --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/conv4featcluster-minibatch-TTTT-out_conv_trial2/max-va.pth --layer_str="blocks.0.feat_cluster.V_buffer" #--vscode_debug

   python weight_extract.py --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/conv4featcluster-minibatch-TTTT-out_conv_trial2/max-va.pth --layer_str="blocks.1.feat_cluster.V_buffer" 

   python weight_extract.py --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/conv4featcluster-minibatch-TTTT-out_conv_trial2/max-va.pth --layer_str="blocks.2.feat_cluster.V_buffer" 

   python weight_extract.py --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/conv4featcluster-minibatch-TTTT-out_conv_trial2/max-va.pth --layer_str="blocks.3.feat_cluster.V_buffer"


```
2. Attention

```
  # image features
        python feat_extract.py --gpu 4 --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/conv4featcluster-minibatch-TTTT-out_conv-self-uv_dist-att-FFTT-sine_pe-pre_normF-init_rp_trial4/max-va.pth --split train --sideout --feat_source="3.conv" --save_path=/home/shawn/cube/capsule_new/fewshot/visualization/data/conv4featcluster-minibatch-TTTT-out_conv-self-uv_dist-att-FFTT-sine_pe-pre_normF-init_rp_trial4
        
        python feat_extract.py --gpu 1 --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/conv4featcluster-minibatch-TTTT-out_conv-self-uv_dist-att-FFTT-sine_pe-pre_normF-init_rp-clusters_128_trial1/max-va.pth --split train --sideout --feat_source="3.conv" --save_path=/home/shawn/cube/capsule_new/fewshot/visualization/data/conv4featcluster-minibatch-TTTT-out_conv-self-uv_dist-att-FFTT-sine_pe-pre_normF-init_rp-clusters_128_trial1
        
        python feat_extract.py --gpu 1 --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/conv4featcluster-minibatch-TTTT-out_conv-self-uv_dist-att-FFTT-sine_pe-pre_normF-init_rp_trial4/max-va.pth --split train --sideout --feat_source="3.conv" --save_path=/home/shawn/cube/capsule_new/fewshot/visualization/data/conv4featcluster-minibatch-TTTT-out_conv-self-uv_dist-att-FFTT-sine_pe-pre_normF-init_rp_trial4
        
        
        
        
  
```

```
  # cluster centers


  
  # uv_dist ([B, C', H, W])


python feat_extract.py --gpu 1 --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/conv4featcluster-minibatch-TTTT-out_conv-self-uv_dist-att-FFTT-sine_pe-pre_normF-init_rp-clusters_128_trial1/max-va.pth --split train --sideout --feat_source="2.featcluster" --save_path=/home/shawn/cube/capsule_new/fewshot/visualization/data/conv4featcluster-minibatch-TTTT-out_conv-self-uv_dist-att-FFTT-sine_pe-pre_normF-init_rp-clusters_128_trial1

python feat_extract.py --gpu 1 --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/conv4featcluster-minibatch-TTTT-out_conv-self-uv_dist-att-FFTT-sine_pe-pre_normF-init_rp-clusters_128_trial1/max-va.pth --split train --sideout --feat_source="3.featcluster" --save_path=/home/shawn/cube/capsule_new/fewshot/visualization/data/conv4featcluster-minibatch-TTTT-out_conv-self-uv_dist-att-FFTT-sine_pe-pre_normF-init_rp-clusters_128_trial1

python feat_extract.py --gpu 2 --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/conv4featcluster-minibatch-TTTT-out_conv-self-uv_dist-att-FFTT-sine_pe-pre_normF-init_rp_trial4/max-va.pth --split train --sideout --feat_source="2.featcluster" --save_path=/home/shawn/cube/capsule_new/fewshot/visualization/data/conv4featcluster-minibatch-TTTT-out_conv-self-uv_dist-att-FFTT-sine_pe-pre_normF-init_rp_trial4

python feat_extract.py --gpu 1 --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/conv4featcluster-minibatch-TTTT-out_conv-self-uv_dist-att-FFTT-sine_pe-pre_normF-init_rp_trial4/max-va.pth --split train --sideout --feat_source="3.featcluster" --save_path=/home/shawn/cube/capsule_new/fewshot/visualization/data/conv4featcluster-minibatch-TTTT-out_conv-self-uv_dist-att-FFTT-sine_pe-pre_normF-init_rp_trial4

```

4.att_map
```
python feat_extract.py --gpu 0 --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/conv4featcluster-minibatch-TTTT-out_conv-self-uv_dist-att-FFTT-sine_pe-pre_normF-init_rp_trial1/max-va.pth --split train --sideout --save_attention --feat_source="2.sideoutfeature.att_map" --save_path=/home/shawn/cube/capsule_new/capsule/ConstellationNets/visualization/data/conv4featcluster-minibatch-TTTT-out_conv-self-uv_dist-att-FFTT-sine_pe-pre_normF-init_rp_trial1
```
```
python feat_extract.py --gpu 0 --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/conv4featcluster-minibatch-TTTT-out_conv-self-uv_dist-att-FFTT-sine_pe-pre_normF-init_rp_trial1/max-va.pth --split train --sideout --save_attention --feat_source="3.sideoutfeature.att_map" --save_path=/home/shawn/cube/capsule_new/capsule/ConstellationNets/visualization/data/conv4featcluster-minibatch-TTTT-out_conv-self-uv_dist-att-FFTT-sine_pe-pre_normF-init_rp_trial1
```

```
python feat_extract.py --gpu 0 --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/conv4featcluster-minibatch-TTTT-out_conv-self-uv_dist-att-FFTT-sine_pe-pre_normF-init_rp_trial1/max-va.pth --split train --sideout --save_attention --feat_source="3.sideoutfeature.query" --save_path=/home/shawn/cube/capsule_new/capsule/ConstellationNets/visualization/data/conv4featcluster-minibatch-TTTT-out_conv-self-uv_dist-att-FFTT-sine_pe-pre_normF-init_rp_trial1
```

```
python feat_extract.py --gpu 0 --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/conv4featcluster-minibatch-TTTT-out_conv-self-uv_dist-att-FFTT-sine_pe-pre_normF-init_rp_trial1/max-va.pth --split train --sideout --save_attention --feat_source="3.sideoutfeature.input" --save_path=/home/shawn/cube/capsule_new/capsule/ConstellationNets/visualization/data/conv4featcluster-minibatch-TTTT-out_conv-self-uv_dist-att-FFTT-sine_pe-pre_normF-init_rp_trial1


```

```
python feat_extract.py --gpu 5 --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/conv4featcluster-minibatch-TTTT-out_conv-self-uv_dist-att-FFTT-no_pe-pre_normF-init_rp_trial1/max-va.pth --split train --sideout --save_attention --feat_source="3.sideoutfeature.att_map" --save_path=/home/shawn/cube/capsule_new/capsule/ConstellationNets/visualization/data/conv4featcluster-minibatch-TTTT-out_conv-self-uv_dist-att-FFTT-no_pe-pre_normF-init_rp_trial1
```


```
python feat_extract.py --gpu 7 --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/conv4featcluster-minibatch-FFFF-out_conv-self-uv_dist-att-FFTT-no_pe-pre_normF-init_rp_trial1/max-va.pth --split train --sideout --save_attention --feat_source="3.sideoutfeature.att_map" --save_path=/home/shawn/cube/capsule_new/capsule/ConstellationNets/visualization/data/conv4featcluster-minibatch-FFFF-out_conv-self-uv_dist-att-FFTT-no_pe-pre_normF-init_rp_trial1
```

```
python feat_extract.py --gpu 1 --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/conv4featcluster-minibatch-TTFF-out_conv-self-uv_dist-att-FFTT-sine_pe-pre_normF-init_rp_trial1/max-va.pth --split train --sideout --save_attention --feat_source="3.sideoutfeature.att_map" --save_path=/home/shawn/cube/capsule_new/capsule/ConstellationNets/visualization/data/conv4featcluster-minibatch-TTFF-out_conv-self-uv_dist-att-FFTT-sine_pe-pre_normF-init_rp_trial1
```

```
python feat_extract.py --gpu 1 --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/conv4featcluster-minibatch-TTFF-out_conv-self-uv_dist-att-FFTT-sine_pe-pre_normF-init_rp_trial1/max-va.pth --split train --sideout --save_attention --feat_source="3.sideoutfeature.att_map" --save_path=/home/shawn/cube/capsule_new/capsule/ConstellationNets/visualization/data/conv4featcluster-minibatch-TTFF-out_conv-self-uv_dist-att-FFTT-sine_pe-pre_normF-init_rp_trial1
```
```
python feat_extract.py --gpu 1 --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/conv4featcluster-minibatch-FTTT-out_conv-self-uv_dist-att-FFTT-sine_pe-pre_normF-init_rp_trial1/max-va.pth --split train --sideout --save_attention --feat_source="3.sideoutfeature.att_map" --save_path=/home/shawn/cube/capsule_new/capsule/ConstellationNets/visualization/data/conv4featcluster-minibatch-FTTT-out_conv-self-uv_dist-att-FFTT-sine_pe-pre_normF-init_rp_trial1
```


```
python feat_extract.py --gpu 1 --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/conv4featcluster-minibatch-TTTT-out_conv-self-uv_dist-att-FFTT-sine_pe-amp_0.2-pre_normF-init_rp_trial1/max-va.pth --split train --sideout --save_attention --feat_source="3.sideoutfeature.att_map" --save_path=/home/shawn/cube/capsule_new/capsule/ConstellationNets/visualization/data/conv4featcluster-minibatch-TTTT-out_conv-self-uv_dist-att-FFTT-sine_pe-amp_0.2-pre_normF-init_rp_trial1
```

```
python feat_extract.py --gpu 1 --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/conv4featcluster-minibatch-TTTT-out_conv-self-uv_dist-att-FFTT-sine_pe-pre_normF-init_rp-1_head_trial1/max-va.pth --split train --sideout --save_attention --feat_source="3.sideoutfeature.att_map" --save_path=/home/shawn/cube/capsule_new/capsule/ConstellationNets/visualization/data/conv4featcluster-minibatch-TTTT-out_conv-self-uv_dist-att-FFTT-sine_pe-pre_normF-init_rp-1_head_trial1
```

```
python feat_extract.py --gpu 1 --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/conv4featcluster-minibatch-TTTT-out_conv-self-uv_dist-att-FFTT-sine_pe-pre_normF-init_rp_trial1/max-va.pth --split train --sideout --save_attention --feat_source="3.sideoutfeature.query" --save_path=/home/shawn/cube/capsule_new/capsule/ConstellationNets/visualization/data/conv4featcluster-minibatch-TTTT-out_conv-self-uv_dist-att-FFTT-sine_pe-pre_normF-init_rp_trial1
```

```
python feat_extract.py --gpu 1 --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/conv4featcluster-minibatch-TTTT-out_conv-self-uv_dist-att-FFTT-sine_pe-pre_normF-init_rp_trial1/max-va.pth --split train --sideout --save_attention --feat_source="3.sideoutfeature.key" --save_path=/home/shawn/cube/capsule_new/capsule/ConstellationNets/visualization/data/conv4featcluster-minibatch-TTTT-out_conv-self-uv_dist-att-FFTT-sine_pe-pre_normF-init_rp_trial1
```

```
python feat_extract.py --gpu 1 --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/conv4featcluster-minibatch-TTTT-out_conv-self-uv_dist-att-FFTT-sine_pe-pre_normF-init_rp_trial1/max-va.pth --split train --sideout --save_attention --feat_source="3.sideoutfeature.value" --save_path=/home/shawn/cube/capsule_new/capsule/ConstellationNets/visualization/data/conv4featcluster-minibatch-TTTT-out_conv-self-uv_dist-att-FFTT-sine_pe-pre_normF-init_rp_trial1
```

```
python feat_extract.py --gpu 0 --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/conv4featcluster-minibatch-TTTT-out_conv-self-uv_dist-att-FFTT-sine_pe-pre_normF-init_rp_trial1/max-va.pth --split train --sideout --save_attention --feat_source="2.sideoutfeature.att_map" --save_path=/home/shawn/cube/capsule_new/capsule/ConstellationNets/visualization/data/conv4featcluster-minibatch-TTTT-out_conv-self-uv_dist-att-FFTT-sine_pe-pre_normF-init_rp_trial1 --label_slt
```


```
python feat_extract.py --gpu 0 --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/res12featcluster-minibatch-sideout-classifier-TFF-FFF-FFF-FFF-self-uv_dist-att-FFF-FFF-TTT-FFF-out_conv-firstTconcat-sine_pe-pre_normF-init_rp_trial1/max-va.pth --split train --sideout --save_attention --feat_source="2.mergeblock1.sideoutfeature.att_map" --save_path=/home/shawn/cube/capsule_new/capsule/ConstellationNets/visualization/data/res12featcluster-minibatch-sideout-classifier-TFF-FFF-FFF-FFF-self-uv_dist-att-FFF-FFF-TTT-FFF-out_conv-firstTconcat-sine_pe-pre_normF-init_rp_trial1 --label_slt
```

```
python feat_extract.py --gpu 0 --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/res12featcluster-minibatch-sideout-classifier-TFF-FFF-FFF-FFF-self-uv_dist-att-FFF-FFF-TTT-FFF-out_conv-firstTconcat-sine_pe-pre_normF-init_rp_trial1/max-va.pth --split train --sideout --save_attention --feat_source="2.mergeblock3.sideoutfeature.att_map" --save_path=/home/shawn/cube/capsule_new/capsule/ConstellationNets/visualization/data/res12featcluster-minibatch-sideout-classifier-TFF-FFF-FFF-FFF-self-uv_dist-att-FFF-FFF-TTT-FFF-out_conv-firstTconcat-sine_pe-pre_normF-init_rp_trial1_visual-label_0,1,8,9 --label_slt --visual_label 0,1,8,9


```