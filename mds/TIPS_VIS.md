1. (delayed) Visualization - Full-image clustering vs. Patch-wise clustering
    - Res12
        ```bash
        python feat_extract.py --gpu 2 --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/res12featcluster-minibatch-sideout-classifier-FFF-FFF-FFF-FFF-out_conv_trial2/max-va.pth --split train --sideout --feat_source="3.drop_layer" --save_path=/home/shawn/weijian/data/res12featcluster-minibatch-sideout-classifier-FFF-FFF-FFF-FFF-out_conv_trial2 --batch_size=32


        python calc_clusters.py --feat_path="/home/shawn/weijian/data/res12featcluster-minibatch-sideout-classifier-FFF-FFF-FFF-FFF-out_conv_trial2/train_3.drop_layer_all_feat.pt" --feat_process="avgpool" --save_name "train_3.drop_layer_avgpool-subset0.01" --subset_ratio=0.01

        python calc_clusters.py --feat_path="/home/shawn/weijian/data/res12featcluster-minibatch-sideout-classifier-FFF-FFF-FFF-FFF-out_conv_trial2/train_3.drop_layer_all_feat.pt" --feat_process="avgpool" --save_name "train_3.drop_layer_avgpool"

        python calc_clusters.py --feat_path="/home/shawn/weijian/data/res12featcluster-minibatch-sideout-classifier-FFF-FFF-FFF-FFF-out_conv_trial2/train_3.drop_layer_all_feat.pt" --feat_process="avgpool" --save_name "train_3.drop_layer_avgpool"

        python calc_clusters.py --feat_path="/home/shawn/weijian/data/res12featcluster-minibatch-sideout-classifier-FFF-FFF-FFF-FFF-out_conv_trial2/train_3.drop_layer_all_feat.pt" --feat_process="avgpool" --save_name "train_3.maxpool_spatial_1x1_k1s1p0_sklearn-minibatch-kmeans-cos-batchsize128_trial2" --stage=3 --kernel_size=1 --stride=1 --padding=0 --cluster_mode='sklearn-minibatch-kmeans-cos' --batch_size=128
   
        python calc_clusters.py --feat_path="/home/shawn/weijian/data/conv4sideout_avgpool/train_3.maxpool_all_feat.pt" --feat_process="spatial" --save_name "train_3.maxpool_spatial_1x1_k1s1p0_sklearn-minibatch-kmeans-cos-batchsize128-cluster_init-custom-reassign0.0" --stage=3 --kernel_size=1 --stride=1 --padding=0 --cluster_mode='sklearn-minibatch-kmeans-cos' --batch_size=128 --cluster_init='custom' --reassignment_ratio=0.0
        ```

2. Conv-4 / Res-12 (FFFF, TTTT) Visualization.
    - Last layer embedding.
        ```bash
        # Feature extraction for Conv-4.
        python feat_extract.py --gpu 0 --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/conv4featcluster-minibatch-FFFF_trial2/max-va.pth --split test --sideout --feat_source="3.maxpool" --save_path=/home/shawn/weijian/data/conv4featcluster-minibatch-FFFF_trial2

        python feat_extract.py --gpu 1 --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/conv4featcluster-minibatch-TTTT-out_conv_trial2/max-va.pth --split test --sideout --feat_source="3.maxpool" --save_path=/home/shawn/weijian/data/conv4featcluster-minibatch-TTTT-out_conv_trial2

        # Feature extraction for Res-12
        python feat_extract.py --gpu 2 --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/res12featcluster-minibatch-FFF-FFF-FFF-FFF_trial2/max-va.pth --split test --sideout --feat_source="3.maxpool" --save_path=/home/shawn/weijian/data/res12featcluster-minibatch-FFF-FFF-FFF-FFF_trial2 

        python feat_extract.py --gpu 3 --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/res12featcluster-minibatch-TFF-FFF-FFF-FFF-out_conv_trial2/max-va.pth --split test --sideout --feat_source="3.maxpool" --save_path=/home/shawn/weijian/data/res12featcluster-minibatch-TFF-FFF-FFF-FFF-out_conv_trial2

        python feat_extract.py --gpu 4 --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/res12featcluster-minibatch-sideout-classifier-TFF-FFF-TTT-FFF-out_conv-firstTconcat_trial2/max-va.pth --split test --sideout --feat_source="3.maxpool" --save_path=/home/shawn/weijian/data/res12featcluster-minibatch-sideout-classifier-TFF-FFF-TTT-FFF-out_conv-firstTconcat_trial2

        # Test classifier to debug the feature extractor.
        python test_classifier.py --gpu 2 --config ./configs/current_configs/featcluster_minibatch/test_classifier_mini-debug.yaml --load=./save/res12featcluster-minibatch-FFF-FFF-FFF-FFF_trial2/max-va.pth --num_workers=0 --vscode_debug
        ```
    
    - Mid layer embedding.
        ```bash
        # Feature extraction for Conv-4.
        python feat_extract.py --gpu 0 --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/conv4featcluster-minibatch-FFFF_trial2/max-va.pth --split test --sideout --feat_source="3.conv" --save_path=/home/shawn/weijian/data/conv4featcluster-minibatch-FFFF_trial2

        python feat_extract.py --gpu 1 --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/conv4featcluster-minibatch-TTTT-out_conv_trial2/max-va.pth --split test --sideout --feat_source="0.conv" --save_path=/home/shawn/weijian/data/conv4featcluster-minibatch-TTTT-out_conv_trial2

        python feat_extract.py --gpu 2 --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/conv4featcluster-minibatch-TTTT-out_conv_trial2/max-va.pth --split test --sideout --feat_source="1.conv" --save_path=/home/shawn/weijian/data/conv4featcluster-minibatch-TTTT-out_conv_trial2

        python feat_extract.py --gpu 3 --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/conv4featcluster-minibatch-TTTT-out_conv_trial2/max-va.pth --split test --sideout --feat_source="2.conv" --save_path=/home/shawn/weijian/data/conv4featcluster-minibatch-TTTT-out_conv_trial2

        python feat_extract.py --gpu 4 --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/conv4featcluster-minibatch-TTTT-out_conv_trial2/max-va.pth --split test --sideout --feat_source="3.conv" --save_path=/home/shawn/weijian/data/conv4featcluster-minibatch-TTTT-out_conv_trial2
        
        python feat_extract.py --gpu 5 --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/conv4featcluster-minibatch-TFFF-out_conv_trial2/max-va.pth --split test --sideout --feat_source="0.conv" --save_path=/home/shawn/weijian/data/conv4featcluster-minibatch-TFFF-out_conv_trial2

        python feat_extract.py --gpu 3 --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/conv4featcluster-minibatch-TTTT-out_conv_trial2/max-va.pth --split train --sideout --feat_source="3.conv" --save_path=/home/shawn/weijian/data/conv4featcluster-minibatch-TTTT-out_conv_trial2

        python feat_extract.py --gpu 2 --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/conv4featcluster-minibatch-TTTT-out_conv_trial2/max-va.pth --split train --sideout --feat_source="2.conv" --save_path=/home/shawn/weijian/data/conv4featcluster-minibatch-TTTT-out_conv_trial2

        python feat_extract.py --gpu 1 --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/conv4featcluster-minibatch-TTTT-out_conv_trial2/max-va.pth --split train --sideout --feat_source="1.conv" --save_path=/home/shawn/weijian/data/conv4featcluster-minibatch-TTTT-out_conv_trial2

        python feat_extract.py --gpu 0 --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/conv4featcluster-minibatch-TTTT-out_conv_trial2/max-va.pth --split train --sideout --feat_source="0.conv" --save_path=/home/shawn/weijian/data/conv4featcluster-minibatch-TTTT-out_conv_trial2



        python feat_extract.py --gpu 3 --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/conv4featcluster-minibatch-FFFF_trial2/max-va.pth --split train --sideout --feat_source="3.relu" --save_path=/home/shawn/weijian/data/conv4featcluster-minibatch-FFFF_trial2

        python feat_extract.py --gpu 2 --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/conv4featcluster-minibatch-FFFF_trial2/max-va.pth --split train --sideout --feat_source="2.relu" --save_path=/home/shawn/weijian/data/conv4featcluster-minibatch-FFFF_trial2

        python feat_extract.py --gpu 1 --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/conv4featcluster-minibatch-FFFF_trial2/max-va.pth --split train --sideout --feat_source="1.relu" --save_path=/home/shawn/weijian/data/conv4featcluster-minibatch-FFFF_trial2

        python feat_extract.py --gpu 0 --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/conv4featcluster-minibatch-FFFF_trial2/max-va.pth --split train --sideout --feat_source="0.relu" --save_path=/home/shawn/weijian/data/cconv4featcluster-minibatch-FFFF_trial2

        

        # Feature extraction for Res-12
        python feat_extract.py --gpu 3 --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/res12featcluster-minibatch-TFF-FFF-FFF-FFF-out_conv_trial2/max-va.pth --split test --sideout --feat_source="0.mergeblock1.conv" --save_path=/home/shawn/weijian/data/res12featcluster-minibatch-TFF-FFF-FFF-FFF-out_conv_trial2

        python feat_extract.py --gpu 4 --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/res12featcluster-minibatch-sideout-classifier-TFF-FFF-TTT-FFF-out_conv-firstTconcat_trial2/max-va.pth --split test --sideout --feat_source="0.mergeblock1.conv" --save_path=/home/shawn/weijian/data/res12featcluster-minibatch-sideout-classifier-TFF-FFF-TTT-FFF-out_conv-firstTconcat_trial2

        python feat_extract.py --gpu 5 --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/res12featcluster-minibatch-sideout-classifier-TFF-FFF-TTT-FFF-out_conv-firstTconcat_trial2/max-va.pth --split test --sideout --feat_source="2.mergeblock1.conv" --save_path=/home/shawn/weijian/data/res12featcluster-minibatch-sideout-classifier-TFF-FFF-TTT-FFF-out_conv-firstTconcat_trial2

        python feat_extract.py --gpu 6 --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/res12featcluster-minibatch-sideout-classifier-TFF-FFF-TTT-FFF-out_conv-firstTconcat_trial2/max-va.pth --split test --sideout --feat_source="2.mergeblock2.conv" --save_path=/home/shawn/weijian/data/res12featcluster-minibatch-sideout-classifier-TFF-FFF-TTT-FFF-out_conv-firstTconcat_trial2

        python feat_extract.py --gpu 7 --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/res12featcluster-minibatch-sideout-classifier-TFF-FFF-TTT-FFF-out_conv-firstTconcat_trial2/max-va.pth --split test --sideout --feat_source="2.mergeblock3.conv" --save_path=/home/shawn/weijian/data/res12featcluster-minibatch-sideout-classifier-TFF-FFF-TTT-FFF-out_conv-firstTconcat_trial2

        # Weight extraction.
        python weight_extract.py --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/conv4featcluster-minibatch-TTTT-out_conv_trial2/max-va.pth --layer_str="blocks.0.feat_cluster.V_buffer"

        python weight_extract.py --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/conv4featcluster-minibatch-TFFF-out_conv_trial2/max-va.pth --layer_str="blocks.0.feat_cluster.V_buffer"

        python weight_extract.py --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/res12featcluster-minibatch-TFF-FFF-FFF-FFF-out_conv_trial2/max-va.pth --layer_str="blocks.0.mergeblock1.feat_cluster.V_buffer"

        python weight_extract.py --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/res12featcluster-minibatch-sideout-classifier-TFF-FFF-TTT-FFF-out_conv-firstTconcat_trial2/max-va.pth --layer_str="blocks.0.mergeblock1.feat_cluster.V_buffer"

        python weight_extract.py --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/res12featcluster-minibatch-sideout-classifier-TFF-FFF-TTT-FFF-out_conv-firstTconcat_trial2/max-va.pth --layer_str="blocks.2.mergeblock1.feat_cluster.V_buffer"

        python weight_extract.py --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/res12featcluster-minibatch-sideout-classifier-TFF-FFF-TTT-FFF-out_conv-firstTconcat_trial2/max-va.pth --layer_str="blocks.2.mergeblock2.feat_cluster.V_buffer"

        python weight_extract.py --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/res12featcluster-minibatch-sideout-classifier-TFF-FFF-TTT-FFF-out_conv-firstTconcat_trial2/max-va.pth --layer_str="blocks.2.mergeblock3.feat_cluster.V_buffer"
        ```