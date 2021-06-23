## Weijian's tips.

1. A demo classifier training: 3-class-64-linear-conv4-lee-84-long.
   - Training scheme: 64-way standard classification. 
   - Classification head: linear layer
   - Feature embedding: Conv-4
   - Data augmentation: Lee (84x84 image)
   - Optimization: Long

   1.1 Train and test:

      ```bash
      python train_classifier.py --gpu 0 --config ./configs/current_configs/train_classifier_mini_3-class-64-linear-conv4-lee-84-long.yaml

      python test_few_shot.py --shot 1 --gpu 0 --config ./configs/current_configs/test_few_shot_mini_3-class-64-linear-conv4-lee-84-long.yaml
      ```

      Results: 
      ```
      Train - epoch 100, train 1.2086|0.6595, val 1.4271|0.6199, fs 1: 0.4742 5: 0.6553, 49.4s 1.1h/1.1h
      Test - test epoch 10: acc=47.30 +- 0.21 (%), loss=1.3055 (@9)
      ```
   
   1.2 More trials:

      ```bash
      python train_classifier.py --gpu 1 --config ./configs/current_configs/train_classifier_mini_3-class-64-linear-conv4-lee-84-long.yaml --tag trial2

      python train_classifier.py --gpu 2 --config ./configs/current_configs/train_classifier_mini_3-class-64-linear-conv4-lee-84-long.yaml --tag trial3
      ```

   FIXME: CPU utilization is too high.

2. A demo meta training: 21-meta-1-sqr-res12-lee-84-lee.
   - Training scheme: Meta 1-shot. 
   - Classification head: Squared Euclidean distance.
   - Feature embedding: Res-12.
   - Data augmentation: Lee (84x84 image)
   - Optimization: Lee

   ```bash
   python train_meta.py --gpu 6 --config ./configs/current_configs/train_meta_mini_21-meta-1-sqr-res12-lee-84-lee.yaml
   ```
   
   
   A 3-shot version.
   ```bash
   # moss101 1-3/2-4/5-7/8-9
   python train_meta.py --gpu 1,3 --config ./configs/current_configs/train_meta_mini_22-meta-3-sqr-res12-lee-84-lee.yaml
   # 1-GPU for 1-shot or 2-GPU for 2-shot.
   ```

3. Add capsules - Conv-4.

   ```bash
   python train_classifier.py --gpu 0,1 --config ./configs/current_configs/capsules/train_classifier_mini_conv4merge-long-0.01-only_conv-conv1x1.yaml

   python train_classifier.py --gpu 2,3 --config ./configs/current_configs/capsules/train_classifier_mini_conv4merge-long-0.01-only_conv-sum.yaml

   python train_classifier.py --gpu 4,5 --config ./configs/current_configs/capsules/train_classifier_mini_conv4merge-long-0.01-only_caps-sum-out_n_caps-2.yaml

   python train_classifier.py --gpu 6,7 --config ./configs/current_configs/capsules/train_classifier_mini_conv4merge-long-0.01-only_caps-sum.yaml

   python train_classifier.py --gpu 0,1 --config ./configs/current_configs/capsules/train_classifier_mini_conv4merge-long-0.01-all-conv1x1.yaml

   python train_classifier.py --gpu 2,3 --config ./configs/current_configs/capsules/train_classifier_mini_conv4merge-long-0.01-all-sum.yaml

   python train_classifier.py --gpu 4,5 --config ./configs/current_configs/capsules/train_classifier_mini_conv4merge-long-0.01-only_caps-conv1x1.yaml

   python train_classifier.py --gpu 6,7 --config ./configs/current_configs/capsules/train_classifier_mini_conv4merge-long-0.01-only_caps-sum-out_n_caps-2-num_routing-2.yaml
   ```

   Evaluation:

   ```bash
   python test_few_shot.py --shot 1 --gpu 0 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4merge-long-0.01-only_conv-conv1x1/max-va.pth
   # test epoch 10: acc=49.90 +- 0.21 (%), loss=1.2585 (@9)

   python test_few_shot.py --shot 1 --gpu 1 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4merge-long-0.01-only_conv-sum/max-va.pth
   # test epoch 10: acc=47.97 +- 0.22 (%), loss=1.3047 (@9)

   python test_few_shot.py --shot 1 --gpu 2 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4merge-long-0.01-only_caps-sum-out_n_caps-2/max-va.pth
   # test epoch 10: acc=48.45 +- 0.22 (%), loss=1.2937 (@9)

   python test_few_shot.py --shot 1 --gpu 3 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4merge-long-0.01-only_caps-sum/max-va.pth
   # test epoch 10: acc=46.85 +- 0.21 (%), loss=1.3338 (@9)

   python test_few_shot.py --shot 1 --gpu 4 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4merge-long-0.01-all-conv1x1/max-va.pth
   # test epoch 10: acc=52.30 +- 0.21 (%), loss=1.2272 (@9)

   python test_few_shot.py --shot 1 --gpu 5 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4merge-long-0.01-all-sum/max-va.pth
   # test epoch 10: acc=49.36 +- 0.21 (%), loss=1.3048 (@9)

   python test_few_shot.py --shot 1 --gpu 0 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4merge-long-0.01-only_caps-conv1x1/max-va.pth
   # test epoch 10: acc=49.90 +- 0.22 (%), loss=1.2503 (@9)

   python test_few_shot.py --shot 1 --gpu 1 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4merge-long-0.01-only_caps-sum-out_n_caps-2-num_routing-2/max-va.pth
   ```

   Debug:

   ```bash
   python train_classifier.py --gpu 0 --config ./configs/current_configs/capsules/train_classifier_mini_conv4merge_debug.yaml --num_workers=0 --vscode_debug
   ```

4. Add capsules - Res-12.
   ```bash
   # moss103 - nvlink: 0-6 / 2-4 / 5-7 / 8-9   no nvlink: 1 3
   # moss105 - nvlink: 0-1 / 4-5 / 6-7         no nvlink: 2 3
   python train_classifier.py --gpu 1,3 --config ./configs/current_configs/capsules/train_classifier_mini_res12merge-long-0.01-only_conv-sum.yaml

   python train_classifier.py --gpu 0,6 --config ./configs/current_configs/capsules/train_classifier_mini_res12merge-long-0.01-only_caps-sum.yaml
   
   python train_classifier.py --gpu 2,4 --config ./configs/current_configs/capsules/train_classifier_mini_res12merge-long-0.01-all-sum.yaml
   
   python train_classifier.py --gpu 5,7 --config ./configs/current_configs/capsules/train_classifier_mini_res12merge-long-0.01-only_conv-conv1x1.yaml
   
   python train_classifier.py --gpu 8,9 --config ./configs/current_configs/capsules/train_classifier_mini_res12merge-long-0.01-only_caps-conv1x1.yaml
   
   python train_classifier.py --gpu 0,1 --config ./configs/current_configs/capsules/train_classifier_mini_res12merge-long-0.01-all-conv1x1.yaml

   python train_classifier.py --gpu 2,3 --config ./configs/current_configs/capsules/train_classifier_mini_res12merge-long-0.01-only_caps-sum-out_n_caps-2.yaml
   ```

   Evaluation:

   ```bash
   python test_few_shot.py --shot 1 --gpu 0 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/deprecated/21-meta-1-sqr-res12-lee-84-lee/max-va.pth

   python test_few_shot.py --shot 1 --gpu 0 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/res12merge-long-0.01-only_conv-sum/max-va.pth

   python test_few_shot.py --shot 1 --gpu 2 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/res12merge-long-0.01-only_caps-sum/max-va.pth

   python test_few_shot.py --shot 1 --gpu 3 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/res12merge-long-0.01-all-sum/max-va.pth

   python test_few_shot.py --shot 1 --gpu 4 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/res12merge-long-0.01-only_conv-conv1x1/max-va.pth

   python test_few_shot.py --shot 1 --gpu 5 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/res12merge-long-0.01-only_caps-conv1x1/max-va.pth

   python test_few_shot.py --shot 1 --gpu 0 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/res12merge-long-0.01-all-conv1x1/max-va.pth

   python test_few_shot.py --shot 1 --gpu 1 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/res12merge-long-0.01-only_caps-sum-out_n_caps-2/max-va.pth
   ```

   Debug:

   ```bash
   python train_classifier.py --gpu 0 --config ./configs/current_configs/capsules/train_classifier_mini_res12merge_debug.yaml --num_workers=0 --vscode_debug
   ```

5. Add capsules - Res-12 (long).
   ```bash
   # moss103 - nvlink: 0-6 / 2-4 / 5-7 / 8-9   no nvlink: 1 3
   # moss105 - nvlink: 0-1 / 4-5 / 6-7         no nvlink: 2 3
   python train_classifier.py --gpu 1,3 --config ./configs/current_configs/capsules/train_classifier_mini_res12merge-long-only_conv-sum.yaml

   python train_classifier.py --gpu 0,6 --config ./configs/current_configs/capsules/train_classifier_mini_res12merge-long-only_caps-sum.yaml
   
   python train_classifier.py --gpu 2,4 --config ./configs/current_configs/capsules/train_classifier_mini_res12merge-long-all-sum.yaml
   
   python train_classifier.py --gpu 5,7 --config ./configs/current_configs/capsules/train_classifier_mini_res12merge-long-only_conv-conv1x1.yaml
   
   python train_classifier.py --gpu 8,9 --config ./configs/current_configs/capsules/train_classifier_mini_res12merge-long-only_caps-conv1x1.yaml
   
   python train_classifier.py --gpu 0,1 --config ./configs/current_configs/capsules/train_classifier_mini_res12merge-long-all-conv1x1.yaml

   python train_classifier.py --gpu 4,5 --config ./configs/current_configs/capsules/train_classifier_mini_res12merge-long-only_caps-sum-out_n_caps-2.yaml

   python train_classifier.py --gpu 6,7 --config ./configs/current_configs/capsules/train_classifier_mini_res12.yaml

   python train_classifier.py --gpu 2 --config ./configs/current_configs/capsules/train_classifier_mini_res12.yaml --tag 1gpu
   ```

   Evaluation:

   ```bash
   python test_few_shot.py --shot 1 --gpu 0 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/res12merge-long-only_conv-sum/max-va.pth
   # test epoch 10: acc=60.82 +- 0.23 (%), loss=1.0146 (@9)

   python test_few_shot.py --shot 1 --gpu 1 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/res12merge-long-only_caps-sum/max-va.pth
   # test epoch 10: acc=51.54 +- 0.22 (%), loss=1.2232 (@9)

   python test_few_shot.py --shot 1 --gpu 2 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/res12merge-long-all-sum/max-va.pth
   # test epoch 10: acc=60.19 +- 0.22 (%), loss=1.0299 (@9)

   python test_few_shot.py --shot 1 --gpu 3 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/res12/max-va.pth
   # test epoch 10: acc=60.45 +- 0.23 (%), loss=1.0239 (@9)

   python test_few_shot.py --shot 1 --gpu 4 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/res12_1gpu/max-va.pth
   # test epoch 10: acc=61.24 +- 0.23 (%), loss=0.9965 (@9)
   ```

6. Feature clustering at last layer - Conv-4.
   ```bash
   python train_classifier.py --gpu 0 --config ./configs/current_configs/featcluster/train_classifier_mini_conv4featcluster-avgpool.yaml

   python train_classifier.py --gpu 1 --config ./configs/current_configs/featcluster/train_classifier_mini_conv4featcluster-baseline.yaml

   python train_classifier.py --gpu 2 --config ./configs/current_configs/featcluster/train_classifier_mini_conv4featcluster-lastlayer-argmax-kmeans-10iter-2cluster-detach_coeff.yaml

   python train_classifier.py --gpu 3 --config ./configs/current_configs/featcluster/train_classifier_mini_conv4featcluster-lastlayer-softmax-kmeans-5iter-2cluster-detach_coeff.yaml

   python train_classifier.py --gpu 4 --config ./configs/current_configs/featcluster/train_classifier_mini_conv4featcluster-lastlayer-softmax-kmeans-10iter-1cluster-detach_coeff.yaml

   python train_classifier.py --gpu 5 --config ./configs/current_configs/featcluster/train_classifier_mini_conv4featcluster-lastlayer-softmax-kmeans-10iter-2cluster-detach_coeff-eps0.0001.yaml

   python train_classifier.py --gpu 6 --config ./configs/current_configs/featcluster/train_classifier_mini_conv4featcluster-lastlayer-softmax-kmeans-10iter-2cluster-detach_coeff-K0.1.yaml

   python train_classifier.py --gpu 0 --config ./configs/current_configs/featcluster/train_classifier_mini_conv4featcluster-lastlayer-softmax-kmeans-10iter-2cluster-detach_coeff-K10.yaml

   python train_classifier.py --gpu 1 --config ./configs/current_configs/featcluster/train_classifier_mini_conv4featcluster-lastlayer-softmax-kmeans-10iter-2cluster-detach_coeff.yaml

   python train_classifier.py --gpu 2 --config ./configs/current_configs/featcluster/train_classifier_mini_conv4featcluster-lastlayer-softmax-kmeans-10iter-2cluster.yaml

   python train_classifier.py --gpu 3 --config ./configs/current_configs/featcluster/train_classifier_mini_conv4featcluster-lastlayer-softmax-kmeans-10iter-3cluster-detach_coeff.yaml

   python train_classifier.py --gpu 4 --config ./configs/current_configs/featcluster/train_classifier_mini_conv4featcluster-lastlayer-softmax-kmeans-20iter-2cluster-detach_coeff.yaml

   python train_classifier.py --gpu 5 --config ./configs/current_configs/featcluster/train_classifier_mini_conv4featcluster-lastlayer-softmax-routing-10iter-1cluster-detach_coeff.yaml

   python train_classifier.py --gpu 6 --config ./configs/current_configs/featcluster/train_classifier_mini_conv4featcluster-lastlayer-softmax-routing-10iter-2cluster-detach_coeff.yaml

   python train_classifier.py --gpu 7 --config ./configs/current_configs/featcluster/train_classifier_mini_conv4featcluster-lastlayer-softmax-routing-10iter-3cluster-detach_coeff.yaml
   ```

   ```bash
   python test_few_shot.py --shot 1 --gpu 0 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4featcluster-avgpool/max-va.pth
   # test epoch 10: acc=50.73 +- 0.22 (%), loss=1.2281 (@9)

   python test_few_shot.py --shot 1 --gpu 1 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4featcluster-baseline/max-va.pth
   # test epoch 10: acc=47.13 +- 0.21 (%), loss=1.3076 (@9)

   python test_few_shot.py --shot 1 --gpu 2 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4featcluster-lastlayer-argmax-kmeans-10iter-2cluster-detach_coeff/max-va.pth
   # test epoch 10: acc=49.36 +- 0.22 (%), loss=1.2509 (@9)

   python test_few_shot.py --shot 1 --gpu 3 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4featcluster-lastlayer-softmax-kmeans-5iter-2cluster-detach_coeff/max-va.pth
   # test epoch 10: acc=49.79 +- 0.22 (%), loss=1.2416 (@9)

   python test_few_shot.py --shot 1 --gpu 4 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4featcluster-lastlayer-softmax-kmeans-10iter-1cluster-detach_coeff/max-va.pth
   # test epoch 10: acc=51.49 +- 0.23 (%), loss=1.2145 (@9)

   python test_few_shot.py --shot 1 --gpu 5 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4featcluster-lastlayer-softmax-kmeans-10iter-2cluster-detach_coeff-eps0.0001/max-va.pth
   # test epoch 10: acc=50.16 +- 0.22 (%), loss=1.2363 (@9)

   python test_few_shot.py --shot 1 --gpu 6 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4featcluster-lastlayer-softmax-kmeans-10iter-2cluster-detach_coeff-K0.1/max-va.pth
   # test epoch 10: acc=49.51 +- 0.22 (%), loss=1.2544 (@9)

   python test_few_shot.py --shot 1 --gpu 7 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4featcluster-lastlayer-softmax-kmeans-10iter-2cluster-detach_coeff-K10/max-va.pth
   # test epoch 10: acc=50.08 +- 0.23 (%), loss=1.2361 (@9)

   python test_few_shot.py --shot 1 --gpu 0 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4featcluster-lastlayer-softmax-kmeans-10iter-2cluster-detach_coeff/max-va.pth
   # test epoch 10: acc=49.82 +- 0.23 (%), loss=1.2472 (@9)

   python test_few_shot.py --shot 1 --gpu 1 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4featcluster-lastlayer-softmax-kmeans-10iter-2cluster/max-va.pth
   # test epoch 10: acc=27.73 +- 0.17 (%), loss=2.5991 (@9

   python test_few_shot.py --shot 1 --gpu 2 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4featcluster-lastlayer-softmax-kmeans-10iter-3cluster-detach_coeff/max-va.pth
   # test epoch 10: acc=49.74 +- 0.22 (%), loss=1.2417 (@9)

   python test_few_shot.py --shot 1 --gpu 3 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4featcluster-lastlayer-softmax-kmeans-20iter-2cluster-detach_coeff/max-va.pth
   # test epoch 10: acc=49.90 +- 0.22 (%), loss=1.2382 (@9)
   ```


   ```bash
   python train_classifier.py --gpu 0 --config ./configs/current_configs/featcluster/train_classifier_mini_conv4featcluster-lastlayer-default-avgpool.yaml

   python train_classifier.py --gpu 1 --config ./configs/current_configs/featcluster/train_classifier_mini_conv4featcluster-lastlayer-default-feat_concat-allcluster-fix_init.yaml

   python train_classifier.py --gpu 2 --config ./configs/current_configs/featcluster/train_classifier_mini_conv4featcluster-lastlayer-default-feat_concat-allcluster.yaml

   python train_classifier.py --gpu 3 --config ./configs/current_configs/featcluster/train_classifier_mini_conv4featcluster-lastlayer-default-feat_concat-avgcluster+avgpool-fix_init.yaml

   python train_classifier.py --gpu 4 --config ./configs/current_configs/featcluster/train_classifier_mini_conv4featcluster-lastlayer-default-feat_concat-avgcluster+avgpool.yaml

   python train_classifier.py --gpu 5 --config ./configs/current_configs/featcluster/train_classifier_mini_conv4featcluster-lastlayer-default-fix_init.yaml

   python train_classifier.py --gpu 6 --config ./configs/current_configs/featcluster/train_classifier_mini_conv4featcluster-lastlayer-default-poolingTTFF-avgpool.yaml

   python train_classifier.py --gpu 7 --config ./configs/current_configs/featcluster/train_classifier_mini_conv4featcluster-lastlayer-default-poolingTTFF.yaml

   python train_classifier.py --gpu 0 --config ./configs/current_configs/featcluster/train_classifier_mini_conv4featcluster-lastlayer-default-poolingTTTF-avgpool.yaml

   python train_classifier.py --gpu 1 --config ./configs/current_configs/featcluster/train_classifier_mini_conv4featcluster-lastlayer-default-poolingTTTF.yaml

   python train_classifier.py --gpu 2 --config ./configs/current_configs/featcluster/train_classifier_mini_conv4featcluster-lastlayer-default.yaml

   python train_classifier.py --gpu 3 --config ./configs/current_configs/featcluster/train_classifier_mini_conv4featcluster-lastlayer-softmax-kmeans-10iter-2cluster-detach_coeff-eps0.001.yaml

   python train_classifier.py --gpu 4 --config ./configs/current_configs/featcluster/train_classifier_mini_conv4featcluster-lastlayer-softmax-kmeans-10iter-2cluster-detach_coeff-K100.yaml

   python train_classifier.py --gpu 5 --config ./configs/current_configs/featcluster/train_classifier_mini_conv4featcluster-lastlayer-softmax-kmeans-10iter-4cluster-detach_coeff.yaml

   python train_classifier.py --gpu 6 --config ./configs/current_configs/featcluster/train_classifier_mini_conv4featcluster-lastlayer-softmax-kmeans-40iter-2cluster-detach_coeff.yaml
   ```



   ```bash
   python test_few_shot.py --shot 1 --gpu 0 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4featcluster-lastlayer-default-avgpool/max-va.pth
   # test epoch 10: acc=51.34 +- 0.23 (%), loss=1.2200 (@9)

   python test_few_shot.py --shot 1 --gpu 1 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4featcluster-lastlayer-default-feat_concat-allcluster-fix_init/max-va.pth
   # test epoch 10: acc=43.98 +- 0.21 (%), loss=1.3951 (@9)

   python test_few_shot.py --shot 1 --gpu 2 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4featcluster-lastlayer-default-feat_concat-allcluster/max-va.pth
   # test epoch 10: acc=42.81 +- 0.20 (%), loss=1.4010 (@9)

   python test_few_shot.py --shot 1 --gpu 3 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4featcluster-lastlayer-default-feat_concat-avgcluster+avgpool-fix_init/max-va.pth
   # test epoch 10: acc=50.57 +- 0.22 (%), loss=1.2368 (@9)

   python test_few_shot.py --shot 1 --gpu 4 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4featcluster-lastlayer-default-feat_concat-avgcluster+avgpool/max-va.pth
   # test epoch 10: acc=50.78 +- 0.23 (%), loss=1.2283 (@9)

   python test_few_shot.py --shot 1 --gpu 5 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4featcluster-lastlayer-default-fix_init/max-va.pth
   # test epoch 10: acc=50.68 +- 0.23 (%), loss=1.2267 (@9)

   python test_few_shot.py --shot 1 --gpu 6 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4featcluster-lastlayer-default-poolingTTFF-avgpool/max-va.pth
   # test epoch 10: acc=49.45 +- 0.22 (%), loss=1.2638 (@9)

   python test_few_shot.py --shot 1 --gpu 7 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4featcluster-lastlayer-default-poolingTTFF/max-va.pth
   # test epoch 10: acc=49.99 +- 0.23 (%), loss=1.2522 (@9)

   python test_few_shot.py --shot 1 --gpu 0 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4featcluster-lastlayer-default-poolingTTTF-avgpool/max-va.pth
   # test epoch 10: acc=51.41 +- 0.22 (%), loss=1.2073 (@9)

   python test_few_shot.py --shot 1 --gpu 1 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4featcluster-lastlayer-default-poolingTTTF/max-va.pth
   # test epoch 10: acc=51.20 +- 0.23 (%), loss=1.2127 (@9)

   python test_few_shot.py --shot 1 --gpu 2 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4featcluster-lastlayer-default/max-va.pth
   # test epoch 10: acc=49.87 +- 0.22 (%), loss=1.2402 (@9)

   python test_few_shot.py --shot 1 --gpu 3 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4featcluster-lastlayer-softmax-kmeans-10iter-2cluster-detach_coeff-eps0.001/max-va.pth
   # test epoch 10: acc=49.87 +- 0.22 (%), loss=1.2425 (@9)

   python test_few_shot.py --shot 1 --gpu 4 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4featcluster-lastlayer-softmax-kmeans-10iter-2cluster-detach_coeff-K100/max-va.pth
   # test epoch 10: acc=50.59 +- 0.22 (%), loss=1.2227 (@9)

   python test_few_shot.py --shot 1 --gpu 5 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4featcluster-lastlayer-softmax-kmeans-10iter-4cluster-detach_coeff/max-va.pth
   # test epoch 10: acc=50.07 +- 0.22 (%), loss=1.2366 (@9)

   python test_few_shot.py --shot 1 --gpu 6 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4featcluster-lastlayer-softmax-kmeans-40iter-2cluster-detach_coeff/max-va.pth
   # test epoch 10: acc=50.14 +- 0.22 (%), loss=1.2334 (@9)
   ```


   ```bash
   python train_classifier.py --gpu 0 --config ./configs/current_configs/featcluster/train_classifier_mini_conv4featcluster-lastlayer-default-poolingTFFF-avgpool.yaml

   python train_classifier.py --gpu 1 --config ./configs/current_configs/featcluster/train_classifier_mini_conv4featcluster-lastlayer-default-poolingTFFF.yaml

   python train_classifier.py --gpu 2 --config ./configs/current_configs/featcluster/train_classifier_mini_conv4featcluster-lastlayer-default-poolingFFFF-avgpool.yaml

   python train_classifier.py --gpu 3 --config ./configs/current_configs/featcluster/train_classifier_mini_conv4featcluster-lastlayer-default-poolingFFFF.yaml

   python train_classifier.py --gpu 4 --config ./configs/current_configs/featcluster/train_classifier_mini_conv4featcluster-lastlayer-default-poolingTTFF-avgpool.yaml --tag=trial2

   python train_classifier.py --gpu 5 --config ./configs/current_configs/featcluster/train_classifier_mini_conv4featcluster-lastlayer-default-poolingTTFF.yaml --tag=trial2

   python train_classifier.py --gpu 6 --config ./configs/current_configs/featcluster/train_classifier_mini_conv4featcluster-lastlayer-default-poolingTTTF-avgpool.yaml --tag=trial2

   python train_classifier.py --gpu 7 --config ./configs/current_configs/featcluster/train_classifier_mini_conv4featcluster-lastlayer-default-poolingTTTF.yaml --tag=trial2
   ```


   ```bash
   python test_few_shot.py --shot 1 --gpu 0 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4featcluster-lastlayer-default-poolingTFFF-avgpool/max-va.pth
   # test epoch 10: acc=47.47 +- 0.22 (%), loss=1.3182 (@9)

   python test_few_shot.py --shot 1 --gpu 1 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4featcluster-lastlayer-default-poolingTFFF/max-va.pth
   # test epoch 10: acc=47.11 +- 0.22 (%), loss=1.3285 (@9)
   
   python test_few_shot.py --shot 1 --gpu 2 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4featcluster-lastlayer-default-poolingFFFF-avgpool/max-va.pth
   # test epoch 10: acc=45.18 +- 0.23 (%), loss=1.4037 (@9)

   python test_few_shot.py --shot 1 --gpu 3 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4featcluster-lastlayer-default-poolingFFFF/max-va.pth
   # test epoch 10: acc=45.12 +- 0.22 (%), loss=1.3760 (@9)

   python test_few_shot.py --shot 1 --gpu 4 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4featcluster-lastlayer-default-poolingTTFF-avgpool_trial2/max-va.pth
   # test epoch 10: acc=50.43 +- 0.23 (%), loss=1.2426 (@9)

   python test_few_shot.py --shot 1 --gpu 5 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4featcluster-lastlayer-default-poolingTTFF_trial2/max-va.pth
   # test epoch 10: acc=48.90 +- 0.22 (%), loss=1.2726 (@9)

   python test_few_shot.py --shot 1 --gpu 6 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4featcluster-lastlayer-default-poolingTTTF-avgpool_trial2/max-va.pth
   # test epoch 10: acc=51.38 +- 0.23 (%), loss=1.2148 (@9)

   python test_few_shot.py --shot 1 --gpu 7 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4featcluster-lastlayer-default-poolingTTTF_trial2/max-va.pth
   # test epoch 10: acc=50.54 +- 0.22 (%), loss=1.2255 (@9)
   ```

   Debug:

   ```bash
   python train_classifier.py --gpu 0 --config ./configs/current_configs/featcluster/train_classifier_mini_conv4featcluster-debug.yaml --num_workers=0 --vscode_debug
   ```


7. Y-branch - Conv-4.

   Debug:

   ```bash
   python train_classifier.py --gpu 0 --config ./configs/current_configs/ybranch/train_classifier_mini_conv4ybranch-debug.yaml --num_workers=0 --vscode_debug

   python train_meta.py --gpu 0 --config ./configs/current_configs/ybranch/train_meta_mini_res12ybranch-debug.yaml --num_workers=0 --vscode_debug
   ```

8. Feature Clustering - All layers - Conv-4.

   ```bash
   python train_classifier.py --gpu 0 --config ./configs/current_configs/featcluster_all_layers/train_classifier_mini_conv4featcluster-all-layers-FFFF-hdim60zdim60.yaml

   python train_classifier.py --gpu 1 --config ./configs/current_configs/featcluster_all_layers/train_classifier_mini_conv4featcluster-all-layers-FFFF.yaml

   python train_classifier.py --gpu 2 --config ./configs/current_configs/featcluster_all_layers/train_classifier_mini_conv4featcluster-all-layers-FFTT-input.yaml

   python train_classifier.py --gpu 3 --config ./configs/current_configs/featcluster_all_layers/train_classifier_mini_conv4featcluster-all-layers-FFTT-out_conv.yaml
   
   python train_classifier.py --gpu 4 --config ./configs/current_configs/featcluster_all_layers/train_classifier_mini_conv4featcluster-all-layers-TTFF-input.yaml
   
   python train_classifier.py --gpu 5 --config ./configs/current_configs/featcluster_all_layers/train_classifier_mini_conv4featcluster-all-layers-TTFF-out_conv.yaml
   
   python train_classifier.py --gpu 6 --config ./configs/current_configs/featcluster_all_layers/train_classifier_mini_conv4featcluster-all-layers-TTTT-input.yaml

   python train_classifier.py --gpu 0 --config ./configs/current_configs/featcluster_all_layers/train_classifier_mini_conv4featcluster-all-layers-FFTT-input3x3.yaml
   
   python train_classifier.py --gpu 1 --config ./configs/current_configs/featcluster_all_layers/train_classifier_mini_conv4featcluster-all-layers-TTFF-input3x3.yaml

   python train_classifier.py --gpu 2 --config ./configs/current_configs/featcluster_all_layers/train_classifier_mini_conv4featcluster-all-layers-TTTT-input3x3-2clusters.yaml
   
   python train_classifier.py --gpu 3 --config ./configs/current_configs/featcluster_all_layers/train_classifier_mini_conv4featcluster-all-layers-TTTT-input3x3-4clusters.yaml
   
   python train_classifier.py --gpu 4 --config ./configs/current_configs/featcluster_all_layers/train_classifier_mini_conv4featcluster-all-layers-TTTT-input3x3-8clusters.yaml
   
   python train_classifier.py --gpu 5 --config ./configs/current_configs/featcluster_all_layers/train_classifier_mini_conv4featcluster-all-layers-TTTT-input3x3-16clusters.yaml
   
   python train_classifier.py --gpu 6 --config ./configs/current_configs/featcluster_all_layers/train_classifier_mini_conv4featcluster-all-layers-TTTT-input3x3.yaml
   
   python train_classifier.py --gpu 7 --config ./configs/current_configs/featcluster_all_layers/train_classifier_mini_conv4featcluster-all-layers-TTTT-out_conv.yaml
   ```

   ```bash
   python test_few_shot.py --shot 1 --gpu 0 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4featcluster-all-layers-FFFF-hdim60zdim60/max-va.pth
   # (rerun) test epoch 10: acc=20.00 +- 0.00 (%), loss=1.6094 (@9)

   python test_few_shot.py --shot 1 --gpu 1 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4featcluster-all-layers-FFFF/max-va.pth
   # test epoch 10: acc=46.50 +- 0.21 (%), loss=1.3181 (@9)

   python test_few_shot.py --shot 1 --gpu 2 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4featcluster-all-layers-FFTT-input/max-va.pth
   # test epoch 10: acc=47.06 +- 0.21 (%), loss=1.3089 (@9)

   python test_few_shot.py --shot 1 --gpu 3 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4featcluster-all-layers-FFTT-out_conv/max-va.pth
   # (rerun) test epoch 10: acc=20.11 +- 0.02 (%), loss=1.6307 (@9)

   python test_few_shot.py --shot 1 --gpu 4 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4featcluster-all-layers-TTFF-input/max-va.pth
   # test epoch 10: acc=46.24 +- 0.21 (%), loss=1.3247 (@9)

   python test_few_shot.py --shot 1 --gpu 5 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4featcluster-all-layers-TTFF-out_conv/max-va.pth
   # test epoch 10: acc=46.28 +- 0.21 (%), loss=1.3081 (@9)

   python test_few_shot.py --shot 1 --gpu 6 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4featcluster-all-layers-TTTT-input/max-va.pth
   # test epoch 10: acc=46.69 +- 0.21 (%), loss=1.3049 (@9)

   python test_few_shot.py --shot 1 --gpu 0 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4featcluster-all-layers-FFTT-input3x3/max-va.pth
   # (rerun) test epoch 10: acc=20.31 +- 0.04 (%), loss=1.7291 (@9)

   python test_few_shot.py --shot 1 --gpu 1 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4featcluster-all-layers-TTFF-input3x3/max-va.pth
   # test epoch 10: acc=47.35 +- 0.21 (%), loss=1.2967 (@9)

   python test_few_shot.py --shot 1 --gpu 2 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4featcluster-all-layers-TTTT-input3x3-2clusters/max-va.pth
   # test epoch 10: acc=46.47 +- 0.21 (%), loss=1.3050 (@9)
   
   python test_few_shot.py --shot 1 --gpu 3 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4featcluster-all-layers-TTTT-input3x3-4clusters/max-va.pth
   # test epoch 10: acc=45.72 +- 0.21 (%), loss=1.3204 (@9)
   
   python test_few_shot.py --shot 1 --gpu 4 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4featcluster-all-layers-TTTT-input3x3-8clusters/max-va.pth
   # (rerun) (out of mem in eval)
   
   python test_few_shot.py --shot 1 --gpu 5 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4featcluster-all-layers-TTTT-input3x3-16clusters/max-va.pth
   # (rerun) (out of mem in training)
   
   python test_few_shot.py --shot 1 --gpu 6 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4featcluster-all-layers-TTTT-input3x3/max-va.pth
   # test epoch 10: acc=46.06 +- 0.21 (%), loss=1.3177 (@9)
   
   python test_few_shot.py --shot 1 --gpu 7 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4featcluster-all-layers-TTTT-out_conv/max-va.pth
   # test epoch 10: acc=47.17 +- 0.21 (%), loss=1.2951 (@9)
   
   ```

   Eval 1357 shots:
   ```bash
   ./scripts/test-shot-1357.sh 0 conv4featcluster-all-layers-FFFF-hdim60zdim60
   ./scripts/test-shot-1357.sh 1 conv4featcluster-all-layers-FFFF
   ./scripts/test-shot-1357.sh 2 conv4featcluster-all-layers-FFTT-input
   ./scripts/test-shot-1357.sh 3 conv4featcluster-all-layers-FFTT-out_conv
   ./scripts/test-shot-1357.sh 4 conv4featcluster-all-layers-TTFF-input
   ./scripts/test-shot-1357.sh 5 conv4featcluster-all-layers-TTFF-out_conv
   ./scripts/test-shot-1357.sh 6 conv4featcluster-all-layers-TTTT-input
   ./scripts/test-shot-1357.sh 0 conv4featcluster-all-layers-FFTT-input3x3
   ./scripts/test-shot-1357.sh 1 conv4featcluster-all-layers-TTFF-input3x3
   ./scripts/test-shot-1357.sh 2 conv4featcluster-all-layers-TTTT-input3x3-2clusters
   ./scripts/test-shot-1357.sh 3 conv4featcluster-all-layers-TTTT-input3x3-4clusters
   ./scripts/test-shot-1357.sh 4 conv4featcluster-all-layers-TTTT-input3x3-8clusters
   ./scripts/test-shot-1357.sh 5 conv4featcluster-all-layers-TTTT-input3x3-16clusters
   ./scripts/test-shot-1357.sh 6 conv4featcluster-all-layers-TTTT-input3x3
   ./scripts/test-shot-1357.sh 7 conv4featcluster-all-layers-TTTT-out_conv
   ./scripts/test-shot-1357.sh 0 conv4featcluster-all-layers-TTTT-input-fix_init
   ./scripts/test-shot-1357.sh 1 conv4featcluster-all-layers-TTTT-input3x3-fix_init
   ./scripts/test-shot-1357.sh 2 conv4featcluster-all-layers-TTTT-out_conv-fix_init
   ```

   Re-run Training:
   ```bash
   ./scripts/train-featcluster_all_layers.sh 8 FFFF-hdim60zdim60
   ./scripts/train-featcluster_all_layers.sh 9 FFTT-out_conv
   ./scripts/train-featcluster_all_layers.sh 0 TTTT-input3x3-16clusters
   ./scripts/train-featcluster_all_layers.sh 1 TTTT-input3x3-8clusters
   ./scripts/train-featcluster_all_layers.sh 6 FFTT-input3x3
   ```

   Eval 1357 shots:
   ```bash
   ./scripts/test-shot-1357.sh 7 conv4featcluster-all-layers-FFTT-input3x3
   ./scripts/test-shot-1357.sh 0 conv4featcluster-all-layers-FFFF-hdim60zdim60
   ./scripts/test-shot-1357.sh 1 conv4featcluster-all-layers-FFTT-out_conv
   ```

   Fix init / disable detach coeff:
   ```bash
   ./scripts/train-featcluster_all_layers.sh 1 TTTT-input-disable-detach
   ./scripts/train-featcluster_all_layers.sh 2 TTTT-input3x3-disable-detach
   ./scripts/train-featcluster_all_layers.sh 3 TTTT-out_conv-disable-detach    # disable-detach => causes NaN
   ./scripts/train-featcluster_all_layers.sh 4 TTTT-input-fix_init
   ./scripts/train-featcluster_all_layers.sh 5 TTTT-input3x3-fix_init
   ./scripts/train-featcluster_all_layers.sh 6 TTTT-out_conv-fix_init
   ```

   Additional training.
   ```bash
   ./scripts/traintest-2more-featcluster_all_layers.sh 0 FFTT-input3x3
   ./scripts/traintest-2more-featcluster_all_layers.sh 2 TTFF-input3x3
   ./scripts/traintest-2more-featcluster_all_layers.sh 3 TTTT-input3x3
   ./scripts/traintest-2more-featcluster_all_layers.sh 4 TTTT-input3x3-fix_init
   ./scripts/traintest-2more-featcluster_all_layers.sh 6 TTTT-input-fix_init
   ./scripts/traintest-2more-featcluster_all_layers.sh 7 TTTT-out_conv-fix_init
   ./scripts/traintest-2more-featcluster_all_layers.sh 0 FFFF-hdim60zdim60
   ./scripts/traintest-2more-featcluster_all_layers.sh 1 FFFF
   ./scripts/traintest-2more-featcluster_all_layers.sh 2 FFTT-input
   ./scripts/traintest-2more-featcluster_all_layers.sh 3 FFTT-out_conv
   ./scripts/traintest-2more-featcluster_all_layers.sh 4 TTFF-input
   ./scripts/traintest-2more-featcluster_all_layers.sh 5 TTFF-out_conv
   ./scripts/traintest-2more-featcluster_all_layers.sh 6 TTTT-input
   ./scripts/traintest-2more-featcluster_all_layers.sh 7 TTTT-out_conv
   ```
   
   ```bash
   ./scripts/traintest-2more-featcluster_all_layers.sh 0 TTTT-input3x3-2clusters
   ./scripts/traintest-2more-featcluster_all_layers.sh 1 TTTT-input3x3-4clusters
   ./scripts/traintest-2more-featcluster_all_layers.sh 2 TTTT-input3x3-8clusters
   ./scripts/traintest-2more-featcluster_all_layers.sh 3 TTTT-input3x3-16clusters
   ./scripts/traintest-2more-featcluster_all_layers.sh 4 TTFF-input
   ./scripts/traintest-2more-featcluster_all_layers.sh 5 TTTT-input-fix_init
   ```

   Debug:
   ```bash
   python train_classifier.py --gpu 0 --config ./configs/current_configs/featcluster_all_layers/train_classifier_mini_conv4featcluster-all-layers-debug.yaml --num_workers=0 --vscode_debug

   ./scripts/test-shot-1357.sh 0 conv4featcluster-all-layers-FFFF

   ./scripts/traintest-2more-featcluster_all_layers.sh 0 debug
   ```

9. Fix embeddings testing - Adjust the use of norms.
   ```bash
   ./scripts/test-dists.sh 0 conv4featcluster-all-layers-FFFF
   ./scripts/test-dists.sh 1 conv4featcluster-all-layers-FFFF_trial2
   ./scripts/test-dists.sh 2 conv4featcluster-all-layers-FFFF_trial3
   ./scripts/test-dists.sh 3 res12
   ./scripts/test-dists.sh 4 res12_1gpu

   ./scripts/test-dists-0.8to1.2.sh 0 conv4featcluster-all-layers-FFFF
   ./scripts/test-dists-0.8to1.2.sh 1 conv4featcluster-all-layers-FFFF_trial2
   ./scripts/test-dists-0.8to1.2.sh 2 conv4featcluster-all-layers-FFFF_trial3
   ./scripts/test-dists-0.8to1.2.sh 3 res12
   ./scripts/test-dists-0.8to1.2.sh 4 res12_1gpu

   ./scripts/test-dists-between-another.sh 0 conv4featcluster-all-layers-FFFF
   ./scripts/test-dists-between-another.sh 1 conv4featcluster-all-layers-FFFF_trial2
   ./scripts/test-dists-between-another.sh 2 conv4featcluster-all-layers-FFFF_trial3

   ./scripts/test-dists-only-norm.sh 5 conv4featcluster-all-layers-FFFF
   ./scripts/test-dists-only-norm.sh 6 conv4featcluster-all-layers-FFFF_trial2
   ./scripts/test-dists-only-norm.sh 7 conv4featcluster-all-layers-FFFF_trial3

   ./scripts/test-dists-single-feature-012.sh 0 conv4featcluster-all-layers-FFFF

   python test_few_shot.py --shot 1 --gpu 0 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4featcluster-all-layers-FFFF/max-va.pth --method='only-norm' --logits_coeff_list='1.0'  # ~23%
   python test_few_shot.py --shot 1 --gpu 0 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4featcluster-all-layers-FFFF/max-va.pth --method='cos,only-norm' --logits_coeff_list='1.0,1.0'  # ~25%
   python test_few_shot.py --shot 1 --gpu 1 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4featcluster-all-layers-FFFF/max-va.pth --method='cos,only-norm' --logits_coeff_list='1.0,0.1'  # ~31%
   python test_few_shot.py --shot 1 --gpu 2 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4featcluster-all-layers-FFFF/max-va.pth --method='cos,only-norm' --logits_coeff_list='1.0,0.01' # ~42.8%
   python test_few_shot.py --shot 1 --gpu 3 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4featcluster-all-layers-FFFF/max-va.pth --method='cos,only-norm' --logits_coeff_list='1.0,0.001' # ~46.62%
   python test_few_shot.py --shot 1 --gpu 4 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4featcluster-all-layers-FFFF/max-va.pth --method='cos,only-norm' --logits_coeff_list='1.0,0.0001' # ~46.70%
   python test_few_shot.py --shot 1 --gpu 5 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4featcluster-all-layers-FFFF/max-va.pth --method='cos,only-norm' --logits_coeff_list='1.0,0.0' # ~46.69%
   ```

   Debug:
   ```bash
   python test_few_shot.py --shot 1 --gpu 0 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4featcluster-all-layers-FFFF/max-va.pth --method='between' --norm_coeff=1.0 --vscode_debug --num_workers=0

   python test_few_shot.py --shot 1 --gpu 0 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4featcluster-all-layers-FFFF/max-va.pth --method='cos,cos' --logits_coeff_list='1.0,1.0'

   python test_few_shot.py --shot 1 --gpu 0 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4featcluster-all-layers-FFFF/max-va.pth --method='cos' --logits_coeff_list='1.0' --disable_out_feat_flatten  --vscode_debug --num_workers=0

   python test_few_shot.py --shot 1 --gpu 0 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4featcluster-all-layers-FFFF_trial2/max-va.pth --method='spatial-normalized' --dist_func_list='cos' --logits_coeff_list='1.0' --disable_out_feat_flatten  --vscode_debug --num_workers=0

   python test_few_shot.py --shot 1 --gpu 1 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4featcluster-all-layers-FFFF/max-va.pth --method='spatial-norm' --dist_func_list='sqr' --logits_coeff_list='1.0' --disable_out_feat_flatten  --vscode_debug --num_workers=0

   python test_few_shot.py --shot 1 --gpu 1 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=/home/shawn/cube/capsule/fewshot/save/3-class-64-linear-conv4-lee-84-long/max-va.pth --method='raw' --dist_func_list='cos' --logits_coeff_list='1.0' --disable_out_feat_flatten  --vscode_debug --num_workers=0

   python test_few_shot.py --shot 1 --gpu 0 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=/home/shawn/cube/capsule/fewshot/save/3-class-64-linear-conv4-lee-84-long/max-va.pth --method='spatial-norm' --dist_func_list='cos' --logits_coeff_list='1.0' --disable_out_feat_flatten  --vscode_debug --num_workers=0
   
   python test_few_shot.py --shot 1 --gpu 0 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=/home/shawn/cube/capsule/fewshot/save/3-class-64-linear-conv4-lee-84-long/max-va.pth --method='spatial-normalized' --dist_func_list='cos' --logits_coeff_list='1.0' --disable_out_feat_flatten  --vscode_debug --num_workers=0

   python test_few_shot.py --shot 1 --gpu 0 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=/home/shawn/cube/capsule/fewshot/save/3-class-64-linear-conv4-lee-84-long/max-va.pth --method='cos' --dist_func_list='none' --logits_coeff_list='1.0' 

   python test_few_shot.py --shot 1 --gpu 0 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=/home/shawn/cube/capsule/fewshot/save/3-class-64-linear-conv4-lee-84-long/max-va.pth --method='avgpool' --dist_func_list='cos' --logits_coeff_list='1.0' --disable_out_feat_flatten

   python test_few_shot.py --shot 1 --gpu 0 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=/home/shawn/cube/capsule/fewshot/save/3-class-64-linear-conv4-lee-84-long/max-va.pth --method='cos,spatial-norm' --dist_func_list='none,sqr' --logits_coeff_list='20,1/64' --disable_out_feat_flatten

   python test_few_shot.py --shot 1 --gpu 0 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=/home/shawn/cube/capsule/fewshot/save/3-class-64-linear-conv4-lee-84-long/max-va.pth --method='cos,spatial-norm,avgpool' --dist_func_list='none,sqr,cos' --logits_coeff_list='20,1/64,20' --disable_out_feat_flatten

   # res12
   python test_few_shot.py --shot 1 --gpu 0 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/res12_1gpu/max-va.pth --method='avgpool' --dist_func_list='cos' --logits_coeff_list='1.0' --disable_out_feat_flatten

   python test_few_shot.py --shot 1 --gpu 0 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/res12_1gpu/max-va.pth --method='avgpool,spatial-norm' --dist_func_list='cos,cos' --logits_coeff_list='20.0,20.0' --disable_out_feat_flatten

   python test_few_shot.py --shot 1 --gpu 0 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/res12_1gpu/max-va.pth --method='spatial-norm' --dist_func_list='sqr' --logits_coeff_list='1.0' --disable_out_feat_flatten

   python test_few_shot.py --shot 1 --gpu 1 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/res12_1gpu/max-va.pth --method='spatial-norm' --dist_func_list='cos' --logits_coeff_list='1.0' --disable_out_feat_flatten

   python test_few_shot.py --shot 1 --gpu 0 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/res12_1gpu/max-va.pth --method='avgpool,cos' --dist_func_list='cos,none' --logits_coeff_list='20.0,20.0' --disable_out_feat_flatten  # ~61.91%

   python test_few_shot.py --shot 1 --gpu 1 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/res12_1gpu/max-va.pth --method='cos' --dist_func_list='none' --logits_coeff_list='1.0' --disable_out_feat_flatten


   python test_few_shot.py --shot 1 --gpu 1 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=/home/shawn/cube/capsule/fewshot/save/3-class-64-linear-conv4-lee-84-long/max-va.pth --method='sqr' --dist_func_list='none' --logits_coeff_list='1.0' --disable_out_feat_flatten  --vscode_debug --num_workers=0

   python test_few_shot.py --shot 1 --gpu 1 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=/home/shawn/cube/capsule/fewshot/save/3-class-64-linear-conv4-lee-84-long/max-va.pth --method='raw' --dist_func_list='sqr-sub' --logits_coeff_list='1.0' --disable_out_feat_flatten  --vscode_debug --num_workers=0

   python test_few_shot.py --shot 1 --gpu 1 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=/home/shawn/cube/capsule/fewshot/save/3-class-64-linear-conv4-lee-84-long/max-va.pth --method='raw' --dist_func_list='sqr' --logits_coeff_list='1.0' --disable_out_feat_flatten  --vscode_debug --num_workers=0

   ```

10. Fix embeddings testing - Feature clustering.
   Extract conv-4 features:
   ```bash
   python feat_extract.py --gpu 0 --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/conv4featcluster-all-layers-FFFF/max-va.pth --split train

   python feat_extract.py --gpu 1 --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/conv4featcluster-all-layers-FFFF/max-va.pth --split test
   
   python feat_extract.py --gpu 2 --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/conv4featcluster-all-layers-FFFF/max-va.pth --split val
   ```

   1600-dim : Cosine similarity + Softmax 
   ```bash
   python test_few_shot.py --shot 1 --gpu 0 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4featcluster-all-layers-FFFF/max-va.pth --method='cos' --dist_func_list='none' --logits_coeff_list='1.0' --disable_out_feat_flatten
   # ~46.69% -> 46.50%

   python test_few_shot.py --shot 1 --gpu 1 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4featcluster-all-layers-FFFF/max-va.pth --method='avgpool' --dist_func_list='cos' --logits_coeff_list='1.0' --disable_out_feat_flatten
   # ~45.72%

   python test_few_shot.py --shot 1 --gpu 2 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4featcluster-all-layers-FFFF/max-va.pth --method='spatial-norm' --dist_func_list='sqr' --logits_coeff_list='1.0' --disable_out_feat_flatten
   # ~43.84%

   python test_few_shot.py --shot 1 --gpu 3 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4featcluster-all-layers-FFFF/max-va.pth --method='spatial-norm' --dist_func_list='cos' --logits_coeff_list='1.0' --disable_out_feat_flatten
   # ~44.91%

   python test_few_shot.py --shot 1 --gpu 4 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4featcluster-all-layers-FFFF/max-va.pth --method='cos,avgpool' --dist_func_list='none,cos' --logits_coeff_list='20,20' --disable_out_feat_flatten
   # ~47.86%

   python test_few_shot.py --shot 1 --gpu 5 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4featcluster-all-layers-FFFF/max-va.pth --method='cos,spatial-norm' --dist_func_list='none,cos' --logits_coeff_list='20,10' --disable_out_feat_flatten
   # ~47.88%

   python test_few_shot.py --shot 1 --gpu 5 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4featcluster-all-layers-FFFF/max-va.pth --method='cos,spatial-norm' --dist_func_list='none,cos' --logits_coeff_list='20,20' --disable_out_feat_flatten
   # ~47.92%

   python test_few_shot.py --shot 1 --gpu 5 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4featcluster-all-layers-FFFF/max-va.pth --method='cos,spatial-norm' --dist_func_list='none,cos' --logits_coeff_list='20,30' --disable_out_feat_flatten
   # ~47.68%

   python test_few_shot.py --shot 1 --gpu 5 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4featcluster-all-layers-FFFF/max-va.pth --method='cos,spatial-norm' --dist_func_list='none,cos' --logits_coeff_list='20,40' --disable_out_feat_flatten
   # ~47.40%

   python test_few_shot.py --shot 1 --gpu 6 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4featcluster-all-layers-FFFF/max-va.pth --method='cos,spatial-norm,avgpool' --dist_func_list='none,sqr,cos' --logits_coeff_list='20,1/64,20' --disable_out_feat_flatten
   # ~47.75%

   python test_few_shot.py --shot 1 --gpu 7 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4featcluster-all-layers-FFFF/max-va.pth --method='cos,spatial-norm,avgpool' --dist_func_list='none,cos,cos' --logits_coeff_list='20,20,20' --disable_out_feat_flatten
   # ~47.61%

   python test_few_shot.py --shot 1 --gpu 8 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4featcluster-all-layers-FFFF/max-va.pth --method='cos,cluster-cos' --dist_func_list='none,cos' --logits_coeff_list='20,30' --disable_out_feat_flatten --feat_clusters_filename='train_feat_clusters.pt'
   # ~46.99%


   python test_few_shot.py --shot 1 --gpu 8 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4featcluster-all-layers-FFFF/max-va.pth --method='cos,cluster-cos' --dist_func_list='none,cos' --logits_coeff_list='20,200' --disable_out_feat_flatten --feat_clusters_filename='train_feat_clusters.pt'
   # ~46.73%

   python test_few_shot.py --shot 1 --gpu 8 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4featcluster-all-layers-FFFF/max-va.pth --method='cos,cluster-cos' --dist_func_list='none,cos' --logits_coeff_list='20,150' --disable_out_feat_flatten --feat_clusters_filename='train_feat_clusters.pt'
   # ~46.94%

   python test_few_shot.py --shot 1 --gpu 8 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4featcluster-all-layers-FFFF/max-va.pth --method='cos,cluster-cos' --dist_func_list='none,cos' --logits_coeff_list='20,100' --disable_out_feat_flatten --feat_clusters_filename='train_feat_clusters.pt'
   # ~47.02%

   python test_few_shot.py --shot 1 --gpu 8 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4featcluster-all-layers-FFFF/max-va.pth --method='cos,cluster-cos' --dist_func_list='none,cos' --logits_coeff_list='20,90' --disable_out_feat_flatten --feat_clusters_filename='train_feat_clusters.pt'
   # ~47.04%

   python test_few_shot.py --shot 1 --gpu 8 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4featcluster-all-layers-FFFF/max-va.pth --method='cos,cluster-cos' --dist_func_list='none,cos' --logits_coeff_list='20,80' --disable_out_feat_flatten --feat_clusters_filename='train_feat_clusters.pt'
   # ~47.04% -> 46.75%

   python test_few_shot.py --shot 1 --gpu 8 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4featcluster-all-layers-FFFF/max-va.pth --method='cos,cluster-cos' --dist_func_list='none,cos' --logits_coeff_list='20,60' --disable_out_feat_flatten --feat_clusters_filename='train_feat_clusters.pt'
   # ~47.00%

   python test_few_shot.py --shot 1 --gpu 8 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4featcluster-all-layers-FFFF/max-va.pth --method='cos,cluster-cos' --dist_func_list='none,cos' --logits_coeff_list='20,50' --disable_out_feat_flatten --feat_clusters_filename='train_feat_clusters.pt'
   # ~46.99%

   python test_few_shot.py --shot 1 --gpu 8 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4featcluster-all-layers-FFFF/max-va.pth --method='cos,cluster-cos' --dist_func_list='none,cos' --logits_coeff_list='20,20' --disable_out_feat_flatten --feat_clusters_filename='train_feat_clusters.pt'
   # ~46.87%

   python test_few_shot.py --shot 1 --gpu 0 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4featcluster-all-layers-FFFF/max-va.pth --method='cos,cluster-cos' --dist_func_list='none,cos' --logits_coeff_list='20,10' --disable_out_feat_flatten --feat_clusters_filename='train_feat_clusters.pt'
   # ~46.75%

   python test_few_shot.py --shot 1 --gpu 0 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4featcluster-all-layers-FFFF/max-va.pth --method='cos,cluster-cos-softmax' --dist_func_list='none,cos' --logits_coeff_list='20,20' --disable_out_feat_flatten --feat_clusters_filename='train_feat_clusters.pt'
   # ~44.96%

   python test_few_shot.py --shot 1 --gpu 1 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4featcluster-all-layers-FFFF/max-va.pth --method='cos,spatial-norm,cluster-cos' --dist_func_list='none,cos,cos' --logits_coeff_list='20,20,20' --disable_out_feat_flatten --feat_clusters_filename='train_feat_clusters.pt'
   # ~47.90%

   python test_few_shot.py --shot 1 --gpu 1 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4featcluster-all-layers-FFFF/max-va.pth --method='cos,spatial-norm,cluster-cos' --dist_func_list='none,cos,cos' --logits_coeff_list='20,20,80' --disable_out_feat_flatten --feat_clusters_filename='train_feat_clusters.pt'
   # ~47.91%

   python test_few_shot.py --shot 1 --gpu 1 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4featcluster-all-layers-FFFF/max-va.pth --method='cos,spatial-norm,cluster-cos' --dist_func_list='none,cos,cos' --logits_coeff_list='20,20,160' --disable_out_feat_flatten --feat_clusters_filename='train_feat_clusters.pt'
   # ~47.71%

   python test_few_shot.py --shot 1 --gpu 1 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4featcluster-all-layers-FFFF/max-va.pth --method='cos,avgpool,cluster-cos' --dist_func_list='none,cos,cos' --logits_coeff_list='20,20,20' --disable_out_feat_flatten --feat_clusters_filename='train_feat_clusters.pt'
   # ~47.82%
   ```
   
   Debug:
   ```bash
   python test_few_shot.py --shot 1 --gpu 0 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4featcluster-all-layers-FFFF/max-va.pth --method='cluster-cos-softmax' --dist_func_list='cos' --logits_coeff_list='1.0' --disable_out_feat_flatten --feat_clusters_filename='train_feat_clusters.pt' --vscode_debug --num_workers=0
   # ~40.09%

   python test_few_shot.py --shot 1 --gpu 0 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4featcluster-all-layers-FFFF/max-va.pth --method='cluster-cos' --dist_func_list='cos' --logits_coeff_list='1.0' --disable_out_feat_flatten --feat_clusters_filename='train_feat_clusters.pt' --vscode_debug --num_workers=0
   # ~41.83%

   python test_few_shot.py --shot 1 --gpu 0 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4featcluster-all-layers-FFFF/max-va.pth --method='linear-transform' --dist_func_list='cos' --logits_coeff_list='1.0' --disable_out_feat_flatten --linear_weights_filename='linear_weights.pt' --vscode_debug --num_workers=0
   # ~51.76%

   python test_few_shot.py --shot 1 --gpu 0 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4featcluster-all-layers-FFFF/max-va.pth --method='linear-cos' --dist_func_list='cos' --logits_coeff_list='1.0' --disable_out_feat_flatten --linear_weights_filename='linear_weights.pt' --vscode_debug --num_workers=0
   # ~51.71%

   python test_few_shot.py --shot 1 --gpu 0 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/res12/max-va.pth --method='linear-transform' --dist_func_list='cos' --logits_coeff_list='1.0' --linear_weights_filename='linear_weights.pt' --vscode_debug --num_workers=0
   # ~60.71%

   python test_few_shot.py --shot 1 --gpu 0 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/res12/max-va.pth --method='linear-cos' --dist_func_list='cos' --logits_coeff_list='1.0' --linear_weights_filename='linear_weights.pt' --vscode_debug --num_workers=0
   # ~60.75%
   ```

10. Fix embeddings testing - Early layers.

   Side outputs.
   ```bash
   python train_classifier.py --gpu 0 --config ./configs/current_configs/sideout/train_classifier_mini_conv4sideout.yaml

   python train_classifier.py --gpu 1 --config ./configs/current_configs/sideout/train_classifier_mini_conv4sideout.yaml --tag trial2

   python train_classifier.py --gpu 2 --config ./configs/current_configs/sideout/train_classifier_mini_conv4sideout.yaml --tag trial3

   python test_few_shot.py --shot 1 --gpu 0 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout/max-va.pth --method='cos' --dist_func_list='none' --logits_coeff_list='1.0'
   # 46.06% -> 46.49%
   
   python test_few_shot.py --shot 1 --gpu 1 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout_trial2/max-va.pth --method='cos' --dist_func_list='none' --logits_coeff_list='1.0'
   # 47.23% -> 47.47%

   python test_few_shot.py --shot 1 --gpu 2 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout_trial3/max-va.pth --method='cos' --dist_func_list='none' --logits_coeff_list='1.0'
   # 46.69% -> 46.75%
   
   python test_few_shot.py --shot 1 --gpu 0 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout/max-va.pth --method='cos' --dist_func_list='none' --logits_coeff_list='1.0' --disable_out_feat_flatten --sideout --feat_source_list='0.maxpool' 
   # 28.33% -> 28.16%

   python test_few_shot.py --shot 1 --gpu 1 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout/max-va.pth --method='cos' --dist_func_list='none' --logits_coeff_list='1.0' --disable_out_feat_flatten --sideout --feat_source_list='1.maxpool' 
   # 31.17% -> 31.21%

   python test_few_shot.py --shot 1 --gpu 2 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout/max-va.pth --method='cos' --dist_func_list='none' --logits_coeff_list='1.0' --disable_out_feat_flatten --sideout --feat_source_list='2.maxpool' 
   # 35.19% -> 35.09%

   python test_few_shot.py --shot 1 --gpu 3 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout/max-va.pth --method='cos' --dist_func_list='none' --logits_coeff_list='1.0' --disable_out_feat_flatten --sideout --feat_source_list='3.maxpool' 
   # 46.06% -> 46.49%

   python test_few_shot.py --shot 1 --gpu 4 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout/max-va.pth --method='avgpool' --dist_func_list='cos' --logits_coeff_list='1.0' --disable_out_feat_flatten --sideout --feat_source_list='0.maxpool' 
   # 31.70% -> 31.55%

   python test_few_shot.py --shot 1 --gpu 5 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout/max-va.pth --method='avgpool' --dist_func_list='cos' --logits_coeff_list='1.0' --disable_out_feat_flatten --sideout --feat_source_list='1.maxpool' 
   # 36.32% -> 36.18% 

   python test_few_shot.py --shot 1 --gpu 6 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout/max-va.pth --method='avgpool' --dist_func_list='cos' --logits_coeff_list='1.0' --disable_out_feat_flatten --sideout --feat_source_list='2.maxpool' 
   # 40.28% -> 40.24%

   python test_few_shot.py --shot 1 --gpu 7 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout/max-va.pth --method='avgpool' --dist_func_list='cos' --logits_coeff_list='1.0' --disable_out_feat_flatten --sideout --feat_source_list='3.maxpool' 
   # 44.77% -> 45.37%

   python test_few_shot.py --shot 1 --gpu 0 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout/max-va.pth --method='raw' --dist_func_list='cos' --logits_coeff_list='1.0' --disable_out_feat_flatten --sideout --feat_source_list='3.conv' 
   # 40.96% -> 41.07%

   python test_few_shot.py --shot 1 --gpu 1 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout/max-va.pth --method='raw' --dist_func_list='cos' --logits_coeff_list='1.0' --disable_out_feat_flatten --sideout --feat_source_list='3.bn' 
   # 41.99% -> 42.20%

   python test_few_shot.py --shot 1 --gpu 2 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout/max-va.pth --method='raw' --dist_func_list='cos' --logits_coeff_list='1.0' --disable_out_feat_flatten --sideout --feat_source_list='3.relu' 
   # 45.52% -> 45.74%

   python test_few_shot.py --shot 1 --gpu 3 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout/max-va.pth --method='raw' --dist_func_list='cos' --logits_coeff_list='1.0' --disable_out_feat_flatten --sideout --feat_source_list='3.maxpool' 
   # 46.06% -> 46.49%

   python test_few_shot.py --shot 1 --gpu 4 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout/max-va.pth --method='avgpool' --dist_func_list='cos' --logits_coeff_list='1.0' --disable_out_feat_flatten --sideout --feat_source_list='3.conv' 
   # 42.84% -> 43.36%

   python test_few_shot.py --shot 1 --gpu 5 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout/max-va.pth --method='avgpool' --dist_func_list='cos' --logits_coeff_list='1.0' --disable_out_feat_flatten --sideout --feat_source_list='3.bn' 
   # 43.42% -> 43.92%

   python test_few_shot.py --shot 1 --gpu 6 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout/max-va.pth --method='avgpool' --dist_func_list='cos' --logits_coeff_list='1.0' --disable_out_feat_flatten --sideout --feat_source_list='3.relu' 
   # 45.35% -> 45.90%

   python test_few_shot.py --shot 1 --gpu 7 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout/max-va.pth --method='avgpool' --dist_func_list='cos' --logits_coeff_list='1.0' --disable_out_feat_flatten --sideout --feat_source_list='3.maxpool' 
   # 44.77% -> 45.37%
   ```

   Extract conv-4 features with side outputs.
   ```bash
   python feat_extract.py --gpu 0 --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/conv4sideout/max-va.pth --split train --sideout --feat_source="0.maxpool"

   python feat_extract.py --gpu 1 --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/conv4sideout/max-va.pth --split train --sideout --feat_source="1.maxpool"

   python feat_extract.py --gpu 2 --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/conv4sideout/max-va.pth --split train --sideout --feat_source="2.maxpool"

   python feat_extract.py --gpu 3 --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/conv4sideout/max-va.pth --split train --sideout --feat_source="3.maxpool"

   python feat_extract.py --gpu 0 --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/conv4sideout/max-va.pth --split train --sideout --feat_source="0.maxpool" --save_path=/home/shawn/weijian/data/conv4sideout

   python feat_extract.py --gpu 1 --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/conv4sideout/max-va.pth --split train --sideout --feat_source="1.maxpool" --save_path=/home/shawn/weijian/data/conv4sideout

   python feat_extract.py --gpu 2 --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/conv4sideout/max-va.pth --split train --sideout --feat_source="2.maxpool" --save_path=/home/shawn/weijian/data/conv4sideout

   python feat_extract.py --gpu 3 --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/conv4sideout/max-va.pth --split train --sideout --feat_source="3.maxpool" --save_path=/home/shawn/weijian/data/conv4sideout

   python test_few_shot.py --shot 1 --gpu 8 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout/max-va.pth --method='cos,cluster-cos' --dist_func_list='none,cos' --logits_coeff_list='20,80' --disable_out_feat_flatten --feat_clusters_filename='train_3.maxpool_flatten_feat_clusters.pt' --vscode_debug --num_workers=0 
   # 46.25% -> 46.68%


   python test_few_shot.py --shot 1 --gpu 0 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout/max-va.pth --method='cluster-cos-avgpool' --dist_func_list='cos' --logits_coeff_list='1.0' --disable_out_feat_flatten --feat_clusters_filename='train_0.maxpool_avgpool_feat_clusters.pt' --sideout --feat_source_list='0.maxpool'
   # 27.75% ->

   python test_few_shot.py --shot 1 --gpu 1 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout/max-va.pth --method='cluster-cos-avgpool' --dist_func_list='cos' --logits_coeff_list='1.0' --disable_out_feat_flatten --feat_clusters_filename='train_1.maxpool_avgpool_feat_clusters.pt' --sideout --feat_source_list='1.maxpool'
   # 31.17% ->

   python test_few_shot.py --shot 1 --gpu 2 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout/max-va.pth --method='cluster-cos-avgpool' --dist_func_list='cos' --logits_coeff_list='1.0' --disable_out_feat_flatten --feat_clusters_filename='train_2.maxpool_avgpool_feat_clusters.pt' --sideout --feat_source_list='2.maxpool'
   # 33.60% ->

   python test_few_shot.py --shot 1 --gpu 3 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout/max-va.pth --method='cluster-cos-avgpool' --dist_func_list='cos' --logits_coeff_list='1.0' --disable_out_feat_flatten --feat_clusters_filename='train_3.maxpool_avgpool_feat_clusters.pt' --sideout --feat_source_list='3.maxpool'
   # 41.42% ->



   python test_few_shot.py --shot 1 --gpu 0 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout/max-va.pth --method='cos,cluster-cos-avgpool' --dist_func_list='none,cos' --logits_coeff_list='20,80' --disable_out_feat_flatten --feat_clusters_filename='train_0.maxpool_avgpool_feat_clusters.pt' --sideout --feat_source_list='final,0.maxpool'
   # 46.09% -> 46.50%

   python test_few_shot.py --shot 1 --gpu 1 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout/max-va.pth --method='cos,cluster-cos-avgpool' --dist_func_list='none,cos' --logits_coeff_list='20,80' --disable_out_feat_flatten --feat_clusters_filename='train_1.maxpool_avgpool_feat_clusters.pt' --sideout --feat_source_list='final,1.maxpool'
   # 46.15% -> 46.46%

   python test_few_shot.py --shot 1 --gpu 2 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout/max-va.pth --method='cos,cluster-cos-avgpool' --dist_func_list='none,cos' --logits_coeff_list='20,80' --disable_out_feat_flatten --feat_clusters_filename='train_2.maxpool_avgpool_feat_clusters.pt' --sideout --feat_source_list='final,2.maxpool'
   # 46.28% -> 46.47%

   python test_few_shot.py --shot 1 --gpu 3 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout/max-va.pth --method='cos,cluster-cos-avgpool' --dist_func_list='none,cos' --logits_coeff_list='20,80' --disable_out_feat_flatten --feat_clusters_filename='train_3.maxpool_avgpool_feat_clusters.pt' --sideout --feat_source_list='final,3.maxpool'
   # 46.39% -> 46.75%



   python test_few_shot.py --shot 1 --gpu 0 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout/max-va.pth --method='cos,cluster-cos-avgpool' --dist_func_list='none,cos' --logits_coeff_list='20,20' --disable_out_feat_flatten --feat_clusters_filename='train_0.maxpool_avgpool_feat_clusters.pt' --sideout --feat_source_list='final,0.maxpool'
   # 46.10% -> 46.51% 

   python test_few_shot.py --shot 1 --gpu 1 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout/max-va.pth --method='cos,cluster-cos-avgpool' --dist_func_list='none,cos' --logits_coeff_list='20,20' --disable_out_feat_flatten --feat_clusters_filename='train_1.maxpool_avgpool_feat_clusters.pt' --sideout --feat_source_list='final,1.maxpool'
   # 46.11% -> 46.52%

   python test_few_shot.py --shot 1 --gpu 2 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout/max-va.pth --method='cos,cluster-cos-avgpool' --dist_func_list='none,cos' --logits_coeff_list='20,20' --disable_out_feat_flatten --feat_clusters_filename='train_2.maxpool_avgpool_feat_clusters.pt' --sideout --feat_source_list='final,2.maxpool'
   # 46.13% -> 46.53%

   python test_few_shot.py --shot 1 --gpu 3 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout/max-va.pth --method='cos,cluster-cos-avgpool' --dist_func_list='none,cos' --logits_coeff_list='20,20' --disable_out_feat_flatten --feat_clusters_filename='train_3.maxpool_avgpool_feat_clusters.pt' --sideout --feat_source_list='final,3.maxpool'
   # 46.12% -> 46.58%








   python test_few_shot.py --shot 1 --gpu 0 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout/max-va.pth --method='cos,avgpool' --dist_func_list='none,cos' --logits_coeff_list='20,80' --disable_out_feat_flatten --sideout --feat_source_list='final,0.maxpool'
   # 41.07% -> 40.90%

   python test_few_shot.py --shot 1 --gpu 1 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout/max-va.pth --method='cos,avgpool' --dist_func_list='none,cos' --logits_coeff_list='20,80' --disable_out_feat_flatten --sideout --feat_source_list='final,1.maxpool'
   # 41.25% -> 41.22%

   python test_few_shot.py --shot 1 --gpu 2 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout/max-va.pth --method='cos,avgpool' --dist_func_list='none,cos' --logits_coeff_list='20,80' --disable_out_feat_flatten --sideout --feat_source_list='final,2.maxpool'
   # 43.27% -> 43.29%

   python test_few_shot.py --shot 1 --gpu 3 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout/max-va.pth --method='cos,avgpool' --dist_func_list='none,cos' --logits_coeff_list='20,80' --disable_out_feat_flatten --sideout --feat_source_list='final,3.maxpool'
   # 46.23% -> 46.76%



   python test_few_shot.py --shot 1 --gpu 0 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout/max-va.pth --method='cos,avgpool' --dist_func_list='none,cos' --logits_coeff_list='20,20' --disable_out_feat_flatten --sideout --feat_source_list='final,0.maxpool'
   # 45.23% -> 45.38%

   python test_few_shot.py --shot 1 --gpu 1 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout/max-va.pth --method='cos,avgpool' --dist_func_list='none,cos' --logits_coeff_list='20,20' --disable_out_feat_flatten --sideout --feat_source_list='final,1.maxpool'
   # 45.42% -> 45.45%

   python test_few_shot.py --shot 1 --gpu 2 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout/max-va.pth --method='cos,avgpool' --dist_func_list='none,cos' --logits_coeff_list='20,20' --disable_out_feat_flatten --sideout --feat_source_list='final,2.maxpool'
   # 46.13% -> 46.11% 

   python test_few_shot.py --shot 1 --gpu 3 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout/max-va.pth --method='cos,avgpool' --dist_func_list='none,cos' --logits_coeff_list='20,20' --disable_out_feat_flatten --sideout --feat_source_list='final,3.maxpool'
   # 47.36% -> 47.78%



   python test_few_shot.py --shot 1 --gpu 0 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout/max-va.pth --method='cos,avgpool' --dist_func_list='none,cos' --logits_coeff_list='20,10' --disable_out_feat_flatten --sideout --feat_source_list='final,0.maxpool'
   # 45.95% -> 46.23%

   python test_few_shot.py --shot 1 --gpu 1 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout/max-va.pth --method='cos,avgpool' --dist_func_list='none,cos' --logits_coeff_list='20,10' --disable_out_feat_flatten --sideout --feat_source_list='final,1.maxpool'
   # 46.27% -> 46.38%

   python test_few_shot.py --shot 1 --gpu 2 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout/max-va.pth --method='cos,avgpool' --dist_func_list='none,cos' --logits_coeff_list='20,10' --disable_out_feat_flatten --sideout --feat_source_list='final,2.maxpool'
   # 46.64% -> 46.75%

   python test_few_shot.py --shot 1 --gpu 3 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout/max-va.pth --method='cos,avgpool' --dist_func_list='none,cos' --logits_coeff_list='20,10' --disable_out_feat_flatten --sideout --feat_source_list='final,3.maxpool'
   # 47.19% -> 47.71%
   ```


   Debug:
   ```bash
   python test_few_shot.py --shot 1 --gpu 0 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout/max-va.pth --method='cos' --dist_func_list='none' --logits_coeff_list='1.0' --disable_out_feat_flatten --sideout --feat_source_list='0.maxpool' --vscode_debug --num_workers=0 

   python feat_extract.py --gpu 0 --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/conv4sideout/max-va.pth --split train --sideout --feat_source="3.maxpool" --vscode_debug --num_workers=0 
   ```

11. Fix embeddings testing - New clustering script.

   Tune iterations and number of clusters.
   ```bash
   # Note that ./save/conv4sideout/max-va.pth is accidentally removed by me!

   python feat_extract.py --gpu 0 --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/conv4sideout_trial2/max-va.pth --split train --sideout --feat_source="0.maxpool" --save_path=/home/shawn/weijian/data/conv4sideout_trial2

   python feat_extract.py --gpu 1 --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/conv4sideout_trial2/max-va.pth --split train --sideout --feat_source="1.maxpool" --save_path=/home/shawn/weijian/data/conv4sideout_trial2

   python feat_extract.py --gpu 2 --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/conv4sideout_trial2/max-va.pth --split train --sideout --feat_source="2.maxpool" --save_path=/home/shawn/weijian/data/conv4sideout_trial2

   python feat_extract.py --gpu 3 --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/conv4sideout_trial2/max-va.pth --split train --sideout --feat_source="3.maxpool" --save_path=/home/shawn/weijian/data/conv4sideout_trial2

   python calc_clusters.py --feat_path="/home/shawn/weijian/data/conv4sideout_trial2/train_3.maxpool_all_feat.pt" --feat_process="flatten" --save_name "train_3.maxpool_flatten"

   python test_few_shot.py --shot 1 --gpu 7 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout_trial2/max-va.pth --method='cos' --dist_func_list='none' --logits_coeff_list='1.0' --disable_out_feat_flatten
   # 47.23% -> 47.47% 

   python test_few_shot.py --shot 1 --gpu 8 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout_trial2/max-va.pth --method='cos,cluster-cos' --dist_func_list='none,cos' --logits_coeff_list='20,80' --disable_out_feat_flatten --feat_clusters_filename='/home/shawn/weijian/data/conv4sideout_trial2/train_3.maxpool_flatten/feat_clusters.pt'
   # 47.57% -> 47.69%

   python test_few_shot.py --shot 1 --gpu 8 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout_trial2/max-va.pth --method='cos,cluster-cos' --dist_func_list='none,cos' --logits_coeff_list='20,20' --disable_out_feat_flatten --feat_clusters_filename='/home/shawn/weijian/data/conv4sideout_trial2/train_3.maxpool_flatten/feat_clusters.pt'
   # 47.40% -> 47.62%

   python calc_clusters.py --gpu 0 --feat_path="/home/shawn/weijian/data/conv4sideout_trial2/train_3.maxpool_all_feat.pt" --feat_process="flatten" --iterations=10 --save_name "train_3.maxpool_flatten_iter10_gpu"

   python calc_clusters.py --gpu 1 --feat_path="/home/shawn/weijian/data/conv4sideout_trial2/train_3.maxpool_all_feat.pt" --feat_process="flatten" --iterations=50 --save_name "train_3.maxpool_flatten_iter50_gpu"

   python calc_clusters.py --gpu 0 --feat_path="/home/shawn/weijian/data/conv4sideout_trial2/train_3.maxpool_all_feat.pt" --feat_process="flatten" --iterations=200 --save_name "train_3.maxpool_flatten_iter200_gpu"
   
   python calc_clusters.py --gpu 1 --feat_path="/home/shawn/weijian/data/conv4sideout_trial2/train_3.maxpool_all_feat.pt" --feat_process="flatten" --iterations=500 --save_name "train_3.maxpool_flatten_iter500_gpu"
   
   python calc_clusters.py --gpu 2 --feat_path="/home/shawn/weijian/data/conv4sideout_trial2/train_3.maxpool_all_feat.pt" --feat_process="flatten" --iterations=1000 --save_name "train_3.maxpool_flatten_iter1000_gpu"
   
   python test_few_shot.py --shot 1 --gpu 8 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout_trial2/max-va.pth --method='cos,cluster-cos' --dist_func_list='none,cos' --logits_coeff_list='20,80' --disable_out_feat_flatten --feat_clusters_filename='/home/shawn/weijian/data/conv4sideout_trial2/train_3.maxpool_flatten_iter10_gpu/feat_clusters.pt'
   # 47.54% -> 47.68%

   python test_few_shot.py --shot 1 --gpu 7 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout_trial2/max-va.pth --method='cos,cluster-cos' --dist_func_list='none,cos' --logits_coeff_list='20,80' --disable_out_feat_flatten --feat_clusters_filename='/home/shawn/weijian/data/conv4sideout_trial2/train_3.maxpool_flatten_iter50_gpu/feat_clusters.pt'
   # 47.57% -> 47.70%

   python test_few_shot.py --shot 1 --gpu 8 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout_trial2/max-va.pth --method='cos,cluster-cos' --dist_func_list='none,cos' --logits_coeff_list='20,80' --disable_out_feat_flatten --feat_clusters_filename='/home/shawn/weijian/data/conv4sideout_trial2/train_3.maxpool_flatten_iter200_gpu/feat_clusters.pt'
   # 47.59% -> 47.69%

   python test_few_shot.py --shot 1 --gpu 7 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout_trial2/max-va.pth --method='cos,cluster-cos' --dist_func_list='none,cos' --logits_coeff_list='20,80' --disable_out_feat_flatten --feat_clusters_filename='/home/shawn/weijian/data/conv4sideout_trial2/train_3.maxpool_flatten_iter500_gpu/feat_clusters.pt'
   # 47.59% -> 47.71%

   python test_few_shot.py --shot 1 --gpu 6 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout_trial2/max-va.pth --method='cos,cluster-cos' --dist_func_list='none,cos' --logits_coeff_list='20,80' --disable_out_feat_flatten --feat_clusters_filename='/home/shawn/weijian/data/conv4sideout_trial2/train_3.maxpool_flatten_iter1000_gpu/feat_clusters.pt'
   # 47.59% -> 47.71%

   python calc_clusters.py --gpu 0 --feat_path="/home/shawn/weijian/data/conv4sideout_trial2/train_3.maxpool_all_feat.pt" --feat_process="flatten" --num_clusters=50 --save_name "train_3.maxpool_flatten_50clusters_gpu"

   python calc_clusters.py --gpu 1 --feat_path="/home/shawn/weijian/data/conv4sideout_trial2/train_3.maxpool_all_feat.pt" --feat_process="flatten" --num_clusters=100 --save_name "train_3.maxpool_flatten_100clusters_gpu"

   python calc_clusters.py --gpu 2 --feat_path="/home/shawn/weijian/data/conv4sideout_trial2/train_3.maxpool_all_feat.pt" --feat_process="flatten" --num_clusters=200 --save_name "train_3.maxpool_flatten_200clusters_gpu"

   python calc_clusters.py --gpu 3 --feat_path="/home/shawn/weijian/data/conv4sideout_trial2/train_3.maxpool_all_feat.pt" --feat_process="flatten" --num_clusters=400 --save_name "train_3.maxpool_flatten_400clusters_gpu"


   python test_few_shot.py --shot 1 --gpu 0 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout_trial2/max-va.pth --method='cos,cluster-cos' --dist_func_list='none,cos' --logits_coeff_list='20,80' --disable_out_feat_flatten --feat_clusters_filename='/home/shawn/weijian/data/conv4sideout_trial2/train_3.maxpool_flatten_50clusters_gpu/feat_clusters.pt'
   # 47.54% -> 47.67%

   python test_few_shot.py --shot 1 --gpu 1 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout_trial2/max-va.pth --method='cos,cluster-cos' --dist_func_list='none,cos' --logits_coeff_list='20,80' --disable_out_feat_flatten --feat_clusters_filename='/home/shawn/weijian/data/conv4sideout_trial2/train_3.maxpool_flatten_100clusters_gpu/feat_clusters.pt'
   # 47.68% -> 47.78%

   python test_few_shot.py --shot 1 --gpu 2 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout_trial2/max-va.pth --method='cos,cluster-cos' --dist_func_list='none,cos' --logits_coeff_list='20,80' --disable_out_feat_flatten --feat_clusters_filename='/home/shawn/weijian/data/conv4sideout_trial2/train_3.maxpool_flatten_200clusters_gpu/feat_clusters.pt'
   # 47.62% -> 47.76%

   python test_few_shot.py --shot 1 --gpu 3 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout_trial2/max-va.pth --method='cos,cluster-cos' --dist_func_list='none,cos' --logits_coeff_list='20,80' --disable_out_feat_flatten --feat_clusters_filename='/home/shawn/weijian/data/conv4sideout_trial2/train_3.maxpool_flatten_400clusters_gpu/feat_clusters.pt'
   # 47.67% -> 47.78%



   python calc_clusters.py --gpu 1 --feat_path="/home/shawn/weijian/data/conv4sideout_trial2/train_3.maxpool_all_feat.pt" --feat_process="flatten" --num_clusters=100 --iterations=500 --save_name "train_3.maxpool_flatten_100clusters_iter500_gpu"

   python test_few_shot.py --shot 1 --gpu 3 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout_trial2/max-va.pth --method='cos,cluster-cos' --dist_func_list='none,cos' --logits_coeff_list='20,80' --disable_out_feat_flatten --feat_clusters_filename='/home/shawn/weijian/data/conv4sideout_trial2/train_3.maxpool_flatten_100clusters_iter500_gpu/feat_clusters.pt'
   # 47.65% -> 47.77%
   ```



   Flatten vs. avgpool:

   ```bash
   python calc_clusters.py --gpu 0 --feat_path="/home/shawn/weijian/data/conv4sideout_trial2/train_3.maxpool_all_feat.pt" --feat_process="avgpool" --save_name "train_3.maxpool_avgpool"

   python test_few_shot.py --shot 1 --gpu 0 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout_trial2/max-va.pth --method='cos,cluster-cos-avgpool' --dist_func_list='none,cos' --logits_coeff_list='20,80' --disable_out_feat_flatten --feat_clusters_filename='/home/shawn/weijian/data/conv4sideout_trial2/train_3.maxpool_avgpool/feat_clusters.pt'
   # 47.73% -> 47.87%

   python calc_clusters.py --gpu 1 --feat_path="/home/shawn/weijian/data/conv4sideout_trial2/train_3.maxpool_all_feat.pt" --feat_process="avgpool" --num_clusters=100 --save_name "train_3.maxpool_avgpool_100clusters"

   python calc_clusters.py --gpu 2 --feat_path="/home/shawn/weijian/data/conv4sideout_trial2/train_3.maxpool_all_feat.pt" --feat_process="avgpool" --num_clusters=200 --save_name "train_3.maxpool_avgpool_200clusters"

   python test_few_shot.py --shot 1 --gpu 1 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout_trial2/max-va.pth --method='cos,cluster-cos-avgpool' --dist_func_list='none,cos' --logits_coeff_list='20,80' --disable_out_feat_flatten --feat_clusters_filename='/home/shawn/weijian/data/conv4sideout_trial2/train_3.maxpool_avgpool_100clusters/feat_clusters.pt'
   # 47.73% -> 47.87%

   python test_few_shot.py --shot 1 --gpu 2 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout_trial2/max-va.pth --method='cos,cluster-cos-avgpool' --dist_func_list='none,cos' --logits_coeff_list='20,80' --disable_out_feat_flatten --feat_clusters_filename='/home/shawn/weijian/data/conv4sideout_trial2/train_3.maxpool_avgpool_200clusters/feat_clusters.pt'
   # 47.76% -> 47.89%

   python test_few_shot.py --shot 1 --gpu 3 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout_trial2/max-va.pth --method='cos,avgpool' --dist_func_list='none,cos' --logits_coeff_list='20,80' --disable_out_feat_flatten 
   # 48.00% -> 48.02%

   python test_few_shot.py --shot 1 --gpu 4 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout_trial2/max-va.pth --method='cos,avgpool' --dist_func_list='none,cos' --logits_coeff_list='20,20' --disable_out_feat_flatten 
   # 48.87% -> 48.93%
   ```
   

   Early layers.

   ```bash
   python calc_clusters.py --feat_path="/home/shawn/weijian/data/conv4sideout_trial2/train_0.maxpool_all_feat.pt" --feat_process="flatten" --save_name "train_0.maxpool_flatten"

   python calc_clusters.py --gpu 0 --feat_path="/home/shawn/weijian/data/conv4sideout_trial2/train_1.maxpool_all_feat.pt" --feat_process="flatten" --save_name "train_1.maxpool_flatten"

   python calc_clusters.py --gpu 1 --feat_path="/home/shawn/weijian/data/conv4sideout_trial2/train_2.maxpool_all_feat.pt" --feat_process="flatten" --save_name "train_2.maxpool_flatten"
   

   python calc_clusters.py --gpu 0 --feat_path="/home/shawn/weijian/data/conv4sideout_trial2/train_0.maxpool_all_feat.pt" --feat_process="avgpool" --save_name "train_0.maxpool_avgpool"

   python calc_clusters.py --gpu 1 --feat_path="/home/shawn/weijian/data/conv4sideout_trial2/train_1.maxpool_all_feat.pt" --feat_process="avgpool" --save_name "train_1.maxpool_avgpool"

   python calc_clusters.py --gpu 2 --feat_path="/home/shawn/weijian/data/conv4sideout_trial2/train_2.maxpool_all_feat.pt" --feat_process="avgpool" --save_name "train_2.maxpool_avgpool"

   python test_few_shot.py --shot 1 --gpu 3 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout_trial2/max-va.pth --method='cos,cluster-cos' --dist_func_list='none,cos' --logits_coeff_list='20,80' --disable_out_feat_flatten --feat_clusters_filename='/home/shawn/weijian/data/conv4sideout_trial2/train_0.maxpool_flatten/feat_clusters.pt' --sideout --feat_source_list='final,0.maxpool' 
   # 46.89% -> 47.09%

   python test_few_shot.py --shot 1 --gpu 4 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout_trial2/max-va.pth --method='cos,cluster-cos' --dist_func_list='none,cos' --logits_coeff_list='20,80' --disable_out_feat_flatten --feat_clusters_filename='/home/shawn/weijian/data/conv4sideout_trial2/train_1.maxpool_flatten/feat_clusters.pt' --sideout --feat_source_list='final,1.maxpool' 
   # 46.90% -> 47.12%

   python test_few_shot.py --shot 1 --gpu 5 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout_trial2/max-va.pth --method='cos,cluster-cos' --dist_func_list='none,cos' --logits_coeff_list='20,80' --disable_out_feat_flatten --feat_clusters_filename='/home/shawn/weijian/data/conv4sideout_trial2/train_2.maxpool_flatten/feat_clusters.pt' --sideout --feat_source_list='final,2.maxpool' 
   # 46.92% -> 47.06%

   python test_few_shot.py --shot 1 --gpu 8 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout_trial2/max-va.pth --method='cos,cluster-cos' --dist_func_list='none,cos' --logits_coeff_list='20,80' --disable_out_feat_flatten --feat_clusters_filename='/home/shawn/weijian/data/conv4sideout_trial2/train_3.maxpool_flatten/feat_clusters.pt' --sideout --feat_source_list='final,3.maxpool' 
   # 47.57% -> 47.69%



   python test_few_shot.py --shot 1 --gpu 4 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout_trial2/max-va.pth --method='cos,cluster-cos-avgpool' --dist_func_list='none,cos' --logits_coeff_list='20,80' --disable_out_feat_flatten --feat_clusters_filename='/home/shawn/weijian/data/conv4sideout_trial2/train_0.maxpool_avgpool/feat_clusters.pt' --sideout --feat_source_list='final,0.maxpool' 
   # 47.26% -> 47.46%

   python test_few_shot.py --shot 1 --gpu 5 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout_trial2/max-va.pth --method='cos,cluster-cos-avgpool' --dist_func_list='none,cos' --logits_coeff_list='20,80' --disable_out_feat_flatten --feat_clusters_filename='/home/shawn/weijian/data/conv4sideout_trial2/train_1.maxpool_avgpool/feat_clusters.pt' --sideout --feat_source_list='final,1.maxpool' 
   # 47.14% -> 47.43%

   python test_few_shot.py --shot 1 --gpu 6 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout_trial2/max-va.pth --method='cos,cluster-cos-avgpool' --dist_func_list='none,cos' --logits_coeff_list='20,80' --disable_out_feat_flatten --feat_clusters_filename='/home/shawn/weijian/data/conv4sideout_trial2/train_2.maxpool_avgpool/feat_clusters.pt' --sideout --feat_source_list='final,2.maxpool' 
   # 47.24% -> 47.47%

   python test_few_shot.py --shot 1 --gpu 7 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout_trial2/max-va.pth --method='cos,cluster-cos-avgpool' --dist_func_list='none,cos' --logits_coeff_list='20,80' --disable_out_feat_flatten --feat_clusters_filename='/home/shawn/weijian/data/conv4sideout_trial2/train_3.maxpool_avgpool/feat_clusters.pt' --sideout --feat_source_list='final,3.maxpool' 
   # 47.73% -> 47.87%




   python test_few_shot.py --shot 1 --gpu 0 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout_trial2/max-va.pth --method='cos,cluster-cos-avgpool' --dist_func_list='none,cos' --logits_coeff_list='20,20' --disable_out_feat_flatten --feat_clusters_filename='/home/shawn/weijian/data/conv4sideout_trial2/train_0.maxpool_avgpool/feat_clusters.pt' --sideout --feat_source_list='final,0.maxpool' 
   # 47.34% -> 47.51%

   python test_few_shot.py --shot 1 --gpu 1 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout_trial2/max-va.pth --method='cos,cluster-cos-avgpool' --dist_func_list='none,cos' --logits_coeff_list='20,20' --disable_out_feat_flatten --feat_clusters_filename='/home/shawn/weijian/data/conv4sideout_trial2/train_1.maxpool_avgpool/feat_clusters.pt' --sideout --feat_source_list='final,1.maxpool' 
   # 47.26% -> 47.47%

   python test_few_shot.py --shot 1 --gpu 2 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout_trial2/max-va.pth --method='cos,cluster-cos-avgpool' --dist_func_list='none,cos' --logits_coeff_list='20,20' --disable_out_feat_flatten --feat_clusters_filename='/home/shawn/weijian/data/conv4sideout_trial2/train_2.maxpool_avgpool/feat_clusters.pt' --sideout --feat_source_list='final,2.maxpool' 
   # 47.24% -> 47.48%

   python test_few_shot.py --shot 1 --gpu 7 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout_trial2/max-va.pth --method='cos,cluster-cos-avgpool' --dist_func_list='none,cos' --logits_coeff_list='20,20' --disable_out_feat_flatten --feat_clusters_filename='/home/shawn/weijian/data/conv4sideout_trial2/train_3.maxpool_avgpool/feat_clusters.pt' --sideout --feat_source_list='final,3.maxpool'
   # 47.45% -> 47.66%
   ```

   Spatial 1x1:
   ```bash
   python calc_clusters.py --gpu 2 --feat_path="/home/shawn/weijian/data/conv4sideout_trial2/train_3.maxpool_all_feat.pt" --feat_process="flatten" --save_name "train_3.maxpool_flatten_trial2" 
   
   python calc_clusters.py --gpu 3 --feat_path="/home/shawn/weijian/data/conv4sideout_trial2/train_2.maxpool_all_feat.pt" --feat_process="spatial1x1" --save_name "train_2.maxpool_spatial1x1" --stage=2

   python calc_clusters.py --gpu 3 --feat_path="/home/shawn/weijian/data/conv4sideout_trial2/train_3.maxpool_all_feat.pt" --feat_process="spatial1x1" --save_name "train_3.maxpool_spatial1x1" --stage=3

   python calc_clusters.py --gpu 3 --feat_path="/home/shawn/weijian/data/conv4sideout_trial2/train_3.maxpool_all_feat.pt" --feat_process="spatial1x1" --save_name "train_3.maxpool_spatial1x1_200clusters" --stage=3 --num_clusters=200


   python test_few_shot.py --shot 1 --gpu 0 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout_trial2/max-va.pth --method='cos,cluster-cos-avgpool' --dist_func_list='none,cos' --logits_coeff_list='20,80' --disable_out_feat_flatten --feat_clusters_filename='/home/shawn/weijian/data/conv4sideout_trial2/train_3.maxpool_spatial1x1/feat_clusters.pt' --sideout --feat_source_list='final,3.maxpool'
   # 47.68% -> 47.77%
   
   python test_few_shot.py --shot 1 --gpu 1 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout_trial2/max-va.pth --method='cos,cluster-cos-avgpool' --dist_func_list='none,cos' --logits_coeff_list='20,20' --disable_out_feat_flatten --feat_clusters_filename='/home/shawn/weijian/data/conv4sideout_trial2/train_3.maxpool_spatial1x1/feat_clusters.pt' --sideout --feat_source_list='final,3.maxpool'
   # 47.76% -> 47.88%  (high)
   
   python test_few_shot.py --shot 1 --gpu 2 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout_trial2/max-va.pth --method='cos,cluster-cos-avgpool' --dist_func_list='none,cos' --logits_coeff_list='20,80' --disable_out_feat_flatten --feat_clusters_filename='/home/shawn/weijian/data/conv4sideout_trial2/train_3.maxpool_spatial1x1_200clusters/feat_clusters.pt' --sideout --feat_source_list='final,3.maxpool'
   # 47.80% -> 47.93%

   python test_few_shot.py --shot 1 --gpu 3 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout_trial2/max-va.pth --method='cos,cluster-cos-avgpool' --dist_func_list='none,cos' --logits_coeff_list='20,20' --disable_out_feat_flatten --feat_clusters_filename='/home/shawn/weijian/data/conv4sideout_trial2/train_3.maxpool_spatial1x1_200clusters/feat_clusters.pt' --sideout --feat_source_list='final,3.maxpool'
   # 47.85% -> 47.97%  (high)




   python test_few_shot.py --shot 1 --gpu 0 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout_trial2/max-va.pth --method='cos,cluster-cos-spatial1x1' --dist_func_list='none,cos' --logits_coeff_list='20,80' --disable_out_feat_flatten --feat_clusters_filename='/home/shawn/weijian/data/conv4sideout_trial2/train_3.maxpool_spatial1x1/feat_clusters.pt' --sideout --feat_source_list='final,3.maxpool'
   # 44.75% -> 44.78%
   
   python test_few_shot.py --shot 1 --gpu 1 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout_trial2/max-va.pth --method='cos,cluster-cos-spatial1x1' --dist_func_list='none,cos' --logits_coeff_list='20,20' --disable_out_feat_flatten --feat_clusters_filename='/home/shawn/weijian/data/conv4sideout_trial2/train_3.maxpool_spatial1x1/feat_clusters.pt' --sideout --feat_source_list='final,3.maxpool'
   # 46.53% -> 46.68%
   
   python test_few_shot.py --shot 1 --gpu 2 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout_trial2/max-va.pth --method='cos,cluster-cos-spatial1x1' --dist_func_list='none,cos' --logits_coeff_list='20,80' --disable_out_feat_flatten --feat_clusters_filename='/home/shawn/weijian/data/conv4sideout_trial2/train_3.maxpool_spatial1x1_200clusters/feat_clusters.pt' --sideout --feat_source_list='final,3.maxpool'
   # 44.83% -> 44.97%

   python test_few_shot.py --shot 1 --gpu 3 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout_trial2/max-va.pth --method='cos,cluster-cos-spatial1x1' --dist_func_list='none,cos' --logits_coeff_list='20,20' --disable_out_feat_flatten --feat_clusters_filename='/home/shawn/weijian/data/conv4sideout_trial2/train_3.maxpool_spatial1x1_200clusters/feat_clusters.pt' --sideout --feat_source_list='final,3.maxpool'
   # 46.56% -> 46.73%
   
   


   
   python test_few_shot.py --shot 1 --gpu 4 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout_trial2/max-va.pth --method='cos,cluster-cos-spatial1x1-avgpool' --dist_func_list='none,cos' --logits_coeff_list='20,80' --disable_out_feat_flatten --feat_clusters_filename='/home/shawn/weijian/data/conv4sideout_trial2/train_3.maxpool_spatial1x1/feat_clusters.pt' --sideout --feat_source_list='final,3.maxpool'
   # 46.67% -> 46.83%
   
   python test_few_shot.py --shot 1 --gpu 5 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout_trial2/max-va.pth --method='cos,cluster-cos-spatial1x1-avgpool' --dist_func_list='none,cos' --logits_coeff_list='20,20' --disable_out_feat_flatten --feat_clusters_filename='/home/shawn/weijian/data/conv4sideout_trial2/train_3.maxpool_spatial1x1/feat_clusters.pt' --sideout --feat_source_list='final,3.maxpool'
   # 47.43% -> 47.60%
   
   python test_few_shot.py --shot 1 --gpu 6 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout_trial2/max-va.pth --method='cos,cluster-cos-spatial1x1-avgpool' --dist_func_list='none,cos' --logits_coeff_list='20,80' --disable_out_feat_flatten --feat_clusters_filename='/home/shawn/weijian/data/conv4sideout_trial2/train_3.maxpool_spatial1x1_200clusters/feat_clusters.pt' --sideout --feat_source_list='final,3.maxpool'
   # 46.84% -> 46.99%

   python test_few_shot.py --shot 1 --gpu 7 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout_trial2/max-va.pth --method='cos,cluster-cos-spatial1x1-avgpool' --dist_func_list='none,cos' --logits_coeff_list='20,20' --disable_out_feat_flatten --feat_clusters_filename='/home/shawn/weijian/data/conv4sideout_trial2/train_3.maxpool_spatial1x1_200clusters/feat_clusters.pt' --sideout --feat_source_list='final,3.maxpool'
   # 47.56% -> 47.69%
   ```



   Spatial - anysize:
   ```bash
   python calc_clusters.py --gpu 0 --feat_path="/home/shawn/weijian/data/conv4sideout_trial2/train_3.maxpool_all_feat.pt" --feat_process="spatial" --save_name "train_3.maxpool_spatial_1x1_k1s1p0" --stage=3 --kernel_size=1 --stride=1 --padding=0 --vscode_debug

   python calc_clusters.py --gpu 0 --feat_path="/home/shawn/weijian/data/conv4sideout_trial2/train_3.maxpool_all_feat.pt" --feat_process="spatial" --save_name "train_3.maxpool_spatial_2x2_k2s1p0" --stage=3 --kernel_size=2 --stride=1 --padding=0

   python calc_clusters.py --gpu 0 --feat_path="/home/shawn/weijian/data/conv4sideout_trial2/train_3.maxpool_all_feat.pt" --feat_process="spatial" --save_name "train_3.maxpool_spatial_5x5_k5s1p0" --stage=3 --kernel_size=5 --stride=1 --padding=0



   python test_few_shot.py --shot 1 --gpu 0 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout_trial2/max-va.pth --method='cos,cluster-cos-spatial' --dist_func_list='none,cos' --logits_coeff_list='20,20' --disable_out_feat_flatten --feat_clusters_filename='/home/shawn/weijian/data/conv4sideout_trial2/train_3.maxpool_spatial_1x1_k1s1p0/feat_clusters.pt' --sideout --feat_source_list='final,3.maxpool'
   # 46.53% -> 46.68%

   python test_few_shot.py --shot 1 --gpu 1 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout_trial2/max-va.pth --method='cos,cluster-cos-spatial' --dist_func_list='none,cos' --logits_coeff_list='20,20' --disable_out_feat_flatten --feat_clusters_filename='/home/shawn/weijian/data/conv4sideout_trial2/train_3.maxpool_spatial_2x2_k2s1p0/feat_clusters.pt' --sideout --feat_source_list='final,3.maxpool' --kernel_size=2 --stride=1 --padding=0
   # 47.36% -> 47.58%

   python test_few_shot.py --shot 1 --gpu 2 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout_trial2/max-va.pth --method='cos,cluster-cos-spatial' --dist_func_list='none,cos' --logits_coeff_list='20,20' --disable_out_feat_flatten --feat_clusters_filename='/home/shawn/weijian/data/conv4sideout_trial2/train_3.maxpool_spatial_5x5_k5s1p0/feat_clusters.pt' --sideout --feat_source_list='final,3.maxpool' --kernel_size=5 --stride=1 --padding=0
   # 47.37% -> 47.60%



   python test_few_shot.py --shot 1 --gpu 3 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout_trial2/max-va.pth --method='cos,cluster-cos-spatial-avgpool' --dist_func_list='none,cos' --logits_coeff_list='20,20' --disable_out_feat_flatten --feat_clusters_filename='/home/shawn/weijian/data/conv4sideout_trial2/train_3.maxpool_spatial_1x1_k1s1p0/feat_clusters.pt' --sideout --feat_source_list='final,3.maxpool'
   # 47.43% -> 47.60%

   python test_few_shot.py --shot 1 --gpu 4 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout_trial2/max-va.pth --method='cos,cluster-cos-spatial-avgpool' --dist_func_list='none,cos' --logits_coeff_list='20,20' --disable_out_feat_flatten --feat_clusters_filename='/home/shawn/weijian/data/conv4sideout_trial2/train_3.maxpool_spatial_2x2_k2s1p0/feat_clusters.pt' --sideout --feat_source_list='final,3.maxpool' --kernel_size=2 --stride=1 --padding=0
   # 47.77% -> 47.96%  (interesting)

   python test_few_shot.py --shot 1 --gpu 5 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout_trial2/max-va.pth --method='cos,cluster-cos-spatial-avgpool' --dist_func_list='none,cos' --logits_coeff_list='20,20' --disable_out_feat_flatten --feat_clusters_filename='/home/shawn/weijian/data/conv4sideout_trial2/train_3.maxpool_spatial_5x5_k5s1p0/feat_clusters.pt' --sideout --feat_source_list='final,3.maxpool' --kernel_size=5 --stride=1 --padding=0
   # 47.37% -> 47.60%
   ```




   Spatial - anysize - early layer:
   ```bash
   python calc_clusters.py --gpu 0 --feat_path="/home/shawn/weijian/data/conv4sideout_trial2/train_2.maxpool_all_feat.pt" --feat_process="spatial" --save_name "train_2.maxpool_spatial_1x1_k1s1p0" --stage=2 --kernel_size=1 --stride=1 --padding=0 --vscode_debug

   python calc_clusters.py --gpu 1 --feat_path="/home/shawn/weijian/data/conv4sideout_trial2/train_2.maxpool_all_feat.pt" --feat_process="spatial" --save_name "train_2.maxpool_spatial_2x2_k2s1p0" --stage=2 --kernel_size=2 --stride=1 --padding=0

   python calc_clusters.py --feat_path="/home/shawn/weijian/data/conv4sideout_trial2/train_2.maxpool_all_feat.pt" --feat_process="spatial" --save_name "train_2.maxpool_spatial_5x5_k5s1p0" --stage=2 --kernel_size=5 --stride=1 --padding=0



   python test_few_shot.py --shot 1 --gpu 0 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout_trial2/max-va.pth --method='cos,cluster-cos-spatial' --dist_func_list='none,cos' --logits_coeff_list='20,20' --disable_out_feat_flatten --feat_clusters_filename='/home/shawn/weijian/data/conv4sideout_trial2/train_2.maxpool_spatial_1x1_k1s1p0/feat_clusters.pt' --sideout --feat_source_list='final,2.maxpool'
   # 46.72% -> 46.89%

   python test_few_shot.py --shot 1 --gpu 1 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout_trial2/max-va.pth --method='cos,cluster-cos-spatial' --dist_func_list='none,cos' --logits_coeff_list='20,20' --disable_out_feat_flatten --feat_clusters_filename='/home/shawn/weijian/data/conv4sideout_trial2/train_2.maxpool_spatial_2x2_k2s1p0/feat_clusters.pt' --sideout --feat_source_list='final,2.maxpool' --kernel_size=2 --stride=1 --padding=0
   # 46.92% -> 47.12%

   python test_few_shot.py --shot 1 --gpu 2 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout_trial2/max-va.pth --method='cos,cluster-cos-spatial' --dist_func_list='none,cos' --logits_coeff_list='20,20' --disable_out_feat_flatten --feat_clusters_filename='/home/shawn/weijian/data/conv4sideout_trial2/train_2.maxpool_spatial_5x5_k5s1p0/feat_clusters.pt' --sideout --feat_source_list='final,2.maxpool' --kernel_size=5 --stride=1 --padding=0
   # 47.26% -> 47.51%




   python test_few_shot.py --shot 1 --gpu 3 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout_trial2/max-va.pth --method='cos,cluster-cos-spatial-avgpool' --dist_func_list='none,cos' --logits_coeff_list='20,20' --disable_out_feat_flatten --feat_clusters_filename='/home/shawn/weijian/data/conv4sideout_trial2/train_2.maxpool_spatial_1x1_k1s1p0/feat_clusters.pt' --sideout --feat_source_list='final,2.maxpool'
   # 47.20% -> 47.48%

   python test_few_shot.py --shot 1 --gpu 4 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout_trial2/max-va.pth --method='cos,cluster-cos-spatial-avgpool' --dist_func_list='none,cos' --logits_coeff_list='20,20' --disable_out_feat_flatten --feat_clusters_filename='/home/shawn/weijian/data/conv4sideout_trial2/train_2.maxpool_spatial_2x2_k2s1p0/feat_clusters.pt' --sideout --feat_source_list='final,2.maxpool' --kernel_size=2 --stride=1 --padding=0
   # 47.25% -> 47.52%

   python test_few_shot.py --shot 1 --gpu 5 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout_trial2/max-va.pth --method='cos,cluster-cos-spatial-avgpool' --dist_func_list='none,cos' --logits_coeff_list='20,20' --disable_out_feat_flatten --feat_clusters_filename='/home/shawn/weijian/data/conv4sideout_trial2/train_2.maxpool_spatial_5x5_k5s1p0/feat_clusters.pt' --sideout --feat_source_list='final,2.maxpool' --kernel_size=5 --stride=1 --padding=0
   # 47.34% -> 47.58%
   ```

   Per-image visualization:
   ```bash
   python calc_clusters.py --gpu 0 --feat_path="/home/shawn/weijian/data/conv4sideout_trial2/train_3.maxpool_all_feat.pt" --feat_process="spatial" --save_name "train_3.maxpool_spatial_1x1_k1s1p0" --stage=3 --kernel_size=1 --stride=1 --padding=0

   python calc_clusters.py --gpu 1 --feat_path="/home/shawn/weijian/data/conv4sideout_trial2/train_3.maxpool_all_feat.pt" --feat_process="spatial" --save_name "train_3.maxpool_spatial_2x2_k2s2p0" --stage=3 --kernel_size=2 --stride=2 --padding=0

   python calc_clusters.py --gpu 2 --feat_path="/home/shawn/weijian/data/conv4sideout_trial2/train_2.maxpool_all_feat.pt" --feat_process="spatial" --save_name "train_2.maxpool_spatial_1x1_k1s1p0" --stage=2 --kernel_size=1 --stride=1 --padding=0

   python calc_clusters.py --gpu 3 --feat_path="/home/shawn/weijian/data/conv4sideout_trial2/train_2.maxpool_all_feat.pt" --feat_process="spatial" --save_name "train_2.maxpool_spatial_2x2_k2s2p0" --stage=2 --kernel_size=2 --stride=2 --padding=0

   python calc_clusters.py --show_cur_iter --feat_path="/home/shawn/weijian/data/conv4sideout_trial2/train_1.maxpool_all_feat.pt" --feat_process="spatial" --save_name "train_1.maxpool_spatial_1x1_k1s1p0" --stage=1 --kernel_size=1 --stride=1 --padding=0

   python calc_clusters.py --show_cur_iter --feat_path="/home/shawn/weijian/data/conv4sideout_trial2/train_1.maxpool_all_feat.pt" --feat_process="spatial" --save_name "train_1.maxpool_spatial_2x2_k2s2p0" --stage=1 --kernel_size=2 --stride=2 --padding=0
   ```



11. Fix embeddings testing - Early layers (trained w/ avgpool).

   Side outputs: Training and evaluation.
   ```bash
   python train_classifier.py --gpu 0 --config ./configs/current_configs/sideout/train_classifier_mini_conv4sideout_avgpool.yaml

   python train_classifier.py --gpu 1 --config ./configs/current_configs/sideout/train_classifier_mini_conv4sideout_avgpool.yaml --tag trial2

   python train_classifier.py --gpu 2 --config ./configs/current_configs/sideout/train_classifier_mini_conv4sideout_avgpool.yaml --tag trial3

   python test_few_shot.py --shot 1 --gpu 0 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout-avgpool/max-va.pth --method='cos' --dist_func_list='none' --logits_coeff_list='1.0'
   # 51.65% -> 51.52%
 
   python test_few_shot.py --shot 1 --gpu 1 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout-avgpool_trial2/max-va.pth --method='cos' --dist_func_list='none' --logits_coeff_list='1.0'
   # 50.75% -> 50.90%

   python test_few_shot.py --shot 1 --gpu 2 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout-avgpool_trial3/max-va.pth --method='cos' --dist_func_list='none' --logits_coeff_list='1.0'
   # 51.25% -> 51.13%
   ```
   
   Feature extraction.
   ```bash
   python feat_extract.py --gpu 0 --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/conv4sideout-avgpool/max-va.pth --split train --sideout --feat_source="0.maxpool" --save_path=/home/shawn/weijian/data/conv4sideout_avgpool

   python feat_extract.py --gpu 1 --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/conv4sideout-avgpool/max-va.pth --split train --sideout --feat_source="1.maxpool" --save_path=/home/shawn/weijian/data/conv4sideout_avgpool

   python feat_extract.py --gpu 2 --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/conv4sideout-avgpool/max-va.pth --split train --sideout --feat_source="2.maxpool" --save_path=/home/shawn/weijian/data/conv4sideout_avgpool

   python feat_extract.py --gpu 3 --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/conv4sideout-avgpool/max-va.pth --split train --sideout --feat_source="3.maxpool" --save_path=/home/shawn/weijian/data/conv4sideout_avgpool
   ```

   





   Test with other features (flatten features / spatial-norm).
   ```bash
   # Only avgpool.
   python test_few_shot.py --shot 1 --gpu 5 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout-avgpool/max-va.pth --method='avgpool' --dist_func_list='cos' --logits_coeff_list='20' --disable_out_feat_flatten --sideout --feat_source_list='0.maxpool'
   # 31.81% -> 31.61%

   python test_few_shot.py --shot 1 --gpu 6 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout-avgpool/max-va.pth --method='avgpool' --dist_func_list='cos' --logits_coeff_list='20' --disable_out_feat_flatten --sideout --feat_source_list='1.maxpool'
   # 37.01% -> 36.83%

   python test_few_shot.py --shot 1 --gpu 7 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout-avgpool/max-va.pth --method='avgpool' --dist_func_list='cos' --logits_coeff_list='20' --disable_out_feat_flatten --sideout --feat_source_list='2.maxpool'
   # 43.82% -> 43.85%

   python test_few_shot.py --shot 1 --gpu 8 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout-avgpool/max-va.pth --method='avgpool' --dist_func_list='cos' --logits_coeff_list='20' --disable_out_feat_flatten --sideout --feat_source_list='3.maxpool'
   # 51.65% -> 51.52%





   # Final + avgpool.
   python test_few_shot.py --shot 1 --gpu 4 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout-avgpool/max-va.pth --method='cos,avgpool' --dist_func_list='none,cos' --logits_coeff_list='20,20' --disable_out_feat_flatten --sideout --feat_source_list='final,0.maxpool'
   # 49.94% -> 49.64%

   python test_few_shot.py --shot 1 --gpu 1 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout-avgpool/max-va.pth --method='cos,avgpool' --dist_func_list='none,cos' --logits_coeff_list='20,20' --disable_out_feat_flatten --sideout --feat_source_list='final,1.maxpool'
   # 49.19% -> 49.13%

   python test_few_shot.py --shot 1 --gpu 2 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout-avgpool/max-va.pth --method='cos,avgpool' --dist_func_list='none,cos' --logits_coeff_list='20,20' --disable_out_feat_flatten --sideout --feat_source_list='final,2.maxpool'
   # 49.62% -> 49.59%

   python test_few_shot.py --shot 1 --gpu 3 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout-avgpool/max-va.pth --method='cos,avgpool' --dist_func_list='none,cos' --logits_coeff_list='20,20' --disable_out_feat_flatten --sideout --feat_source_list='final,3.maxpool'
   # 51.65% -> 51.52%




   # Only flatten.
   python test_few_shot.py --shot 1 --gpu 5 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout-avgpool/max-va.pth --method='cos' --dist_func_list='none' --logits_coeff_list='20' --disable_out_feat_flatten --sideout --feat_source_list='0.maxpool'
   # 28.08% -> 27.91%

   python test_few_shot.py --shot 1 --gpu 6 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout-avgpool/max-va.pth --method='cos' --dist_func_list='none' --logits_coeff_list='20' --disable_out_feat_flatten --sideout --feat_source_list='1.maxpool'
   # 30.97% -> 30.78%

   python test_few_shot.py --shot 1 --gpu 7 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout-avgpool/max-va.pth --method='cos' --dist_func_list='none' --logits_coeff_list='20' --disable_out_feat_flatten --sideout --feat_source_list='2.maxpool'
   # 38.90% -> 38.68%

   python test_few_shot.py --shot 1 --gpu 8 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout-avgpool/max-va.pth --method='cos' --dist_func_list='none' --logits_coeff_list='20' --disable_out_feat_flatten --sideout --feat_source_list='3.maxpool'
   # 49.72% -> 49.54%



   
    # Final + flatten.
   python test_few_shot.py --shot 1 --gpu 1 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout-avgpool/max-va.pth --method='cos,cos' --dist_func_list='none,none' --logits_coeff_list='20,20' --disable_out_feat_flatten --sideout --feat_source_list='final,0.maxpool'
   # 50.02% -> 49.78%

   python test_few_shot.py --shot 1 --gpu 2 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout-avgpool/max-va.pth --method='cos,cos' --dist_func_list='none,none' --logits_coeff_list='20,20' --disable_out_feat_flatten --sideout --feat_source_list='final,1.maxpool'
   # 50.62% -> 50.43%

   python test_few_shot.py --shot 1 --gpu 3 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout-avgpool/max-va.pth --method='cos,cos' --dist_func_list='none,none' --logits_coeff_list='20,20' --disable_out_feat_flatten --sideout --feat_source_list='final,2.maxpool'
   # 51.40% -> 51.07%

   python test_few_shot.py --shot 1 --gpu 4 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout-avgpool/max-va.pth --method='cos,cos' --dist_func_list='none,none' --logits_coeff_list='20,20' --disable_out_feat_flatten --sideout --feat_source_list='final,3.maxpool'
   # 52.27% -> 52.17%




   # Only spatial norm.
   python test_few_shot.py --shot 1 --gpu 0 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout-avgpool/max-va.pth --method='spatial-norm' --dist_func_list='cos' --logits_coeff_list='20' --disable_out_feat_flatten --sideout --feat_source_list='0.maxpool'
   # 32.64% -> 32.56%

   python test_few_shot.py --shot 1 --gpu 1 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout-avgpool/max-va.pth --method='spatial-norm' --dist_func_list='cos' --logits_coeff_list='20' --disable_out_feat_flatten --sideout --feat_source_list='1.maxpool'
   # 38.35% -> 38.19%

   python test_few_shot.py --shot 1 --gpu 2 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout-avgpool/max-va.pth --method='spatial-norm' --dist_func_list='cos' --logits_coeff_list='20' --disable_out_feat_flatten --sideout --feat_source_list='2.maxpool'
   # 44.26% -> 44.36%

   python test_few_shot.py --shot 1 --gpu 3 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout-avgpool/max-va.pth --method='spatial-norm' --dist_func_list='cos' --logits_coeff_list='20' --disable_out_feat_flatten --sideout --feat_source_list='3.maxpool'
   # 51.36% -> 51.35%





   # Final + spatial-norm.
   python test_few_shot.py --shot 1 --gpu 0 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout-avgpool/max-va.pth --method='cos,spatial-norm' --dist_func_list='none,cos' --logits_coeff_list='20,20' --disable_out_feat_flatten --sideout --feat_source_list='final,0.maxpool'
   # 50.70% -> 50.42%

   python test_few_shot.py --shot 1 --gpu 1 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout-avgpool/max-va.pth --method='cos,spatial-norm' --dist_func_list='none,cos' --logits_coeff_list='20,20' --disable_out_feat_flatten --sideout --feat_source_list='final,1.maxpool'
   # 50.78% -> 50.62%

   python test_few_shot.py --shot 1 --gpu 2 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout-avgpool/max-va.pth --method='cos,spatial-norm' --dist_func_list='none,cos' --logits_coeff_list='20,20' --disable_out_feat_flatten --sideout --feat_source_list='final,2.maxpool'
   # 50.46% -> 50.23%

   python test_few_shot.py --shot 1 --gpu 3 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout-avgpool/max-va.pth --method='cos,spatial-norm' --dist_func_list='none,cos' --logits_coeff_list='20,20' --disable_out_feat_flatten --sideout --feat_source_list='final,3.maxpool'
   # 51.69% -> 51.66%
   ```




   Clustering: Last layer 1x1.
   ```bash
   python calc_clusters.py --show_cur_iter --feat_path="/home/shawn/weijian/data/conv4sideout_avgpool/train_0.maxpool_all_feat.pt" --feat_process="spatial" --save_name "train_0.maxpool_spatial_1x1_k1s1p0" --stage=0 --kernel_size=1 --stride=1 --padding=0

   python calc_clusters.py --show_cur_iter --feat_path="/home/shawn/weijian/data/conv4sideout_avgpool/train_1.maxpool_all_feat.pt" --feat_process="spatial" --save_name "train_1.maxpool_spatial_1x1_k1s1p0" --stage=1 --kernel_size=1 --stride=1 --padding=0

   python calc_clusters.py --gpu 2 --feat_path="/home/shawn/weijian/data/conv4sideout_avgpool/train_2.maxpool_all_feat.pt" --feat_process="spatial" --save_name "train_2.maxpool_spatial_1x1_k1s1p0" --stage=2 --kernel_size=1 --stride=1 --padding=0

   python calc_clusters.py --gpu 3 --feat_path="/home/shawn/weijian/data/conv4sideout_avgpool/train_3.maxpool_all_feat.pt" --feat_process="spatial" --save_name "train_3.maxpool_spatial_1x1_k1s1p0" --stage=3 --kernel_size=1 --stride=1 --padding=0
   ```

   Test with clustering features.
   ```bash
   # Only cluster-cos-spatial.
   python test_few_shot.py --shot 1 --gpu 4 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout-avgpool/max-va.pth --method='cluster-cos-spatial' --dist_func_list='cos' --logits_coeff_list='20' --disable_out_feat_flatten --feat_clusters_filename='/home/shawn/weijian/data/conv4sideout_avgpool/train_0.maxpool_spatial_1x1_k1s1p0/feat_clusters.pt' --sideout --feat_source_list='0.maxpool'
   # 25.63% -> 25.65%

   python test_few_shot.py --shot 1 --gpu 1 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout-avgpool/max-va.pth --method='cluster-cos-spatial' --dist_func_list='cos' --logits_coeff_list='20' --disable_out_feat_flatten --feat_clusters_filename='/home/shawn/weijian/data/conv4sideout_avgpool/train_1.maxpool_spatial_1x1_k1s1p0/feat_clusters.pt' --sideout --feat_source_list='1.maxpool'
   # 29.04% -> 29.23% (perhaps it is too sparse / uniform and there is not enough information)
   #                   try to (1) print the values (2) use softmax + different K.

   python test_few_shot.py --shot 1 --gpu 2 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout-avgpool/max-va.pth --method='cluster-cos-spatial' --dist_func_list='cos' --logits_coeff_list='20' --disable_out_feat_flatten --feat_clusters_filename='/home/shawn/weijian/data/conv4sideout_avgpool/train_2.maxpool_spatial_1x1_k1s1p0/feat_clusters.pt' --sideout --feat_source_list='2.maxpool'
   # 31.34% -> 31.19%

   python test_few_shot.py --shot 1 --gpu 3 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout-avgpool/max-va.pth --method='cluster-cos-spatial' --dist_func_list='cos' --logits_coeff_list='20' --disable_out_feat_flatten --feat_clusters_filename='/home/shawn/weijian/data/conv4sideout_avgpool/train_3.maxpool_spatial_1x1_k1s1p0/feat_clusters.pt' --sideout --feat_source_list='3.maxpool'
   # 40.88% -> 41.05%







   # Final + cluster-cos-spatial.
   python test_few_shot.py --shot 1 --gpu 1 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout-avgpool/max-va.pth --method='cos,cluster-cos-spatial' --dist_func_list='none,cos' --logits_coeff_list='20,20' --disable_out_feat_flatten --feat_clusters_filename='/home/shawn/weijian/data/conv4sideout_avgpool/train_0.maxpool_spatial_1x1_k1s1p0/feat_clusters.pt' --sideout --feat_source_list='final,0.maxpool'
   # 51.72% -> 51.50%

   python test_few_shot.py --shot 1 --gpu 2 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout-avgpool/max-va.pth --method='cos,cluster-cos-spatial' --dist_func_list='none,cos' --logits_coeff_list='20,20' --disable_out_feat_flatten --feat_clusters_filename='/home/shawn/weijian/data/conv4sideout_avgpool/train_1.maxpool_spatial_1x1_k1s1p0/feat_clusters.pt' --sideout --feat_source_list='final,1.maxpool'
   # 51.77% -> 51.59% (interesting)

   python test_few_shot.py --shot 1 --gpu 3 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout-avgpool/max-va.pth --method='cos,cluster-cos-spatial' --dist_func_list='none,cos' --logits_coeff_list='20,20' --disable_out_feat_flatten --feat_clusters_filename='/home/shawn/weijian/data/conv4sideout_avgpool/train_2.maxpool_spatial_1x1_k1s1p0/feat_clusters.pt' --sideout --feat_source_list='final,2.maxpool'
   # 51.52% -> 51.29%

   python test_few_shot.py --shot 1 --gpu 4 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout-avgpool/max-va.pth --method='cos,cluster-cos-spatial' --dist_func_list='none,cos' --logits_coeff_list='20,20' --disable_out_feat_flatten --feat_clusters_filename='/home/shawn/weijian/data/conv4sideout_avgpool/train_3.maxpool_spatial_1x1_k1s1p0/feat_clusters.pt' --sideout --feat_source_list='final,3.maxpool'
   # 51.47% -> 51.34%






   
   # Only cluster-cos-spatial-avgpool.
   python test_few_shot.py --shot 1 --gpu 1 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout-avgpool/max-va.pth --method='cluster-cos-spatial-avgpool' --dist_func_list='cos' --logits_coeff_list='20' --disable_out_feat_flatten --feat_clusters_filename='/home/shawn/weijian/data/conv4sideout_avgpool/train_0.maxpool_spatial_1x1_k1s1p0/feat_clusters.pt' --sideout --feat_source_list='0.maxpool'
   # 29.46% -> 29.22%

   python test_few_shot.py --shot 1 --gpu 2 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout-avgpool/max-va.pth --method='cluster-cos-spatial-avgpool' --dist_func_list='cos' --logits_coeff_list='20' --disable_out_feat_flatten --feat_clusters_filename='/home/shawn/weijian/data/conv4sideout_avgpool/train_1.maxpool_spatial_1x1_k1s1p0/feat_clusters.pt' --sideout --feat_source_list='1.maxpool'
   # 33.15% -> 33.05%

   python test_few_shot.py --shot 1 --gpu 3 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout-avgpool/max-va.pth --method='cluster-cos-spatial-avgpool' --dist_func_list='cos' --logits_coeff_list='20' --disable_out_feat_flatten --feat_clusters_filename='/home/shawn/weijian/data/conv4sideout_avgpool/train_2.maxpool_spatial_1x1_k1s1p0/feat_clusters.pt' --sideout --feat_source_list='2.maxpool'
   # 36.19% -> 36.39%

   python test_few_shot.py --shot 1 --gpu 4 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout-avgpool/max-va.pth --method='cluster-cos-spatial-avgpool' --dist_func_list='cos' --logits_coeff_list='20' --disable_out_feat_flatten --feat_clusters_filename='/home/shawn/weijian/data/conv4sideout_avgpool/train_3.maxpool_spatial_1x1_k1s1p0/feat_clusters.pt' --sideout --feat_source_list='3.maxpool'
   # 43.33% -> 43.39%






   # Final + cluster-cos-spatial-avgpool.
   python test_few_shot.py --shot 1 --gpu 5 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout-avgpool/max-va.pth --method='cos,cluster-cos-spatial-avgpool' --dist_func_list='none,cos' --logits_coeff_list='20,20' --disable_out_feat_flatten --feat_clusters_filename='/home/shawn/weijian/data/conv4sideout_avgpool/train_0.maxpool_spatial_1x1_k1s1p0/feat_clusters.pt' --sideout --feat_source_list='final,0.maxpool'
   # 51.57% -> 51.46%

   python test_few_shot.py --shot 1 --gpu 6 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout-avgpool/max-va.pth --method='cos,cluster-cos-spatial-avgpool' --dist_func_list='none,cos' --logits_coeff_list='20,20' --disable_out_feat_flatten --feat_clusters_filename='/home/shawn/weijian/data/conv4sideout_avgpool/train_1.maxpool_spatial_1x1_k1s1p0/feat_clusters.pt' --sideout --feat_source_list='final,1.maxpool'
   # 51.54% -> 51.39%

   python test_few_shot.py --shot 1 --gpu 7 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout-avgpool/max-va.pth --method='cos,cluster-cos-spatial-avgpool' --dist_func_list='none,cos' --logits_coeff_list='20,20' --disable_out_feat_flatten --feat_clusters_filename='/home/shawn/weijian/data/conv4sideout_avgpool/train_2.maxpool_spatial_1x1_k1s1p0/feat_clusters.pt' --sideout --feat_source_list='final,2.maxpool'
   # 51.38% -> 51.24%

   python test_few_shot.py --shot 1 --gpu 8 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout-avgpool/max-va.pth --method='cos,cluster-cos-spatial-avgpool' --dist_func_list='none,cos' --logits_coeff_list='20,20' --disable_out_feat_flatten --feat_clusters_filename='/home/shawn/weijian/data/conv4sideout_avgpool/train_3.maxpool_spatial_1x1_k1s1p0/feat_clusters.pt' --sideout --feat_source_list='final,3.maxpool'
   # 51.36% -> 51.27%   









   # Only cluster-cos-avgpool.
   python test_few_shot.py --shot 1 --gpu 1 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout-avgpool/max-va.pth --method='cluster-cos-avgpool' --dist_func_list='cos' --logits_coeff_list='20' --disable_out_feat_flatten --feat_clusters_filename='/home/shawn/weijian/data/conv4sideout_avgpool/train_0.maxpool_spatial_1x1_k1s1p0/feat_clusters.pt' --sideout --feat_source_list='0.maxpool'
   # 29.85% -> 29.76%

   python test_few_shot.py --shot 1 --gpu 2 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout-avgpool/max-va.pth --method='cluster-cos-avgpool' --dist_func_list='cos' --logits_coeff_list='20' --disable_out_feat_flatten --feat_clusters_filename='/home/shawn/weijian/data/conv4sideout_avgpool/train_1.maxpool_spatial_1x1_k1s1p0/feat_clusters.pt' --sideout --feat_source_list='1.maxpool'
   # 34.13% -> 33.97%

   python test_few_shot.py --shot 1 --gpu 3 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout-avgpool/max-va.pth --method='cluster-cos-avgpool' --dist_func_list='cos' --logits_coeff_list='20' --disable_out_feat_flatten --feat_clusters_filename='/home/shawn/weijian/data/conv4sideout_avgpool/train_2.maxpool_spatial_1x1_k1s1p0/feat_clusters.pt' --sideout --feat_source_list='2.maxpool'
   # 38.41% -> 38.63%

   python test_few_shot.py --shot 1 --gpu 4 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout-avgpool/max-va.pth --method='cluster-cos-avgpool' --dist_func_list='cos' --logits_coeff_list='20' --disable_out_feat_flatten --feat_clusters_filename='/home/shawn/weijian/data/conv4sideout_avgpool/train_3.maxpool_spatial_1x1_k1s1p0/feat_clusters.pt' --sideout --feat_source_list='3.maxpool'
   # 45.95% -> 46.06%






   # Final + cluster-cos-avgpool.
   python test_few_shot.py --shot 1 --gpu 5 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout-avgpool/max-va.pth --method='cos,cluster-cos-avgpool' --dist_func_list='none,cos' --logits_coeff_list='20,20' --disable_out_feat_flatten --feat_clusters_filename='/home/shawn/weijian/data/conv4sideout_avgpool/train_0.maxpool_spatial_1x1_k1s1p0/feat_clusters.pt' --sideout --feat_source_list='final,0.maxpool'
   # 51.61% -> 51.49%

   python test_few_shot.py --shot 1 --gpu 6 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout-avgpool/max-va.pth --method='cos,cluster-cos-avgpool' --dist_func_list='none,cos' --logits_coeff_list='20,20' --disable_out_feat_flatten --feat_clusters_filename='/home/shawn/weijian/data/conv4sideout_avgpool/train_1.maxpool_spatial_1x1_k1s1p0/feat_clusters.pt' --sideout --feat_source_list='final,1.maxpool'
   # 51.58% -> 51.43%

   python test_few_shot.py --shot 1 --gpu 7 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout-avgpool/max-va.pth --method='cos,cluster-cos-avgpool' --dist_func_list='none,cos' --logits_coeff_list='20,20' --disable_out_feat_flatten --feat_clusters_filename='/home/shawn/weijian/data/conv4sideout_avgpool/train_2.maxpool_spatial_1x1_k1s1p0/feat_clusters.pt' --sideout --feat_source_list='final,2.maxpool'
   # 51.45% -> 51.31%

   python test_few_shot.py --shot 1 --gpu 8 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout-avgpool/max-va.pth --method='cos,cluster-cos-avgpool' --dist_func_list='none,cos' --logits_coeff_list='20,20' --disable_out_feat_flatten --feat_clusters_filename='/home/shawn/weijian/data/conv4sideout_avgpool/train_3.maxpool_spatial_1x1_k1s1p0/feat_clusters.pt' --sideout --feat_source_list='final,3.maxpool'
   # 51.45% -> 51.35%





    # Only cluster-cos-spatial-softmax.
   python test_few_shot.py --shot 1 --gpu 1 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout-avgpool/max-va.pth --method='cluster-cos-spatial-softmax' --dist_func_list='cos' --logits_coeff_list='20' --disable_out_feat_flatten --feat_clusters_filename='/home/shawn/weijian/data/conv4sideout_avgpool/train_0.maxpool_spatial_1x1_k1s1p0/feat_clusters.pt' --sideout --feat_source_list='0.maxpool'
   # 29.07% -> 29.02%

   python test_few_shot.py --shot 1 --gpu 2 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout-avgpool/max-va.pth --method='cluster-cos-spatial-softmax' --dist_func_list='cos' --logits_coeff_list='20' --disable_out_feat_flatten --feat_clusters_filename='/home/shawn/weijian/data/conv4sideout_avgpool/train_1.maxpool_spatial_1x1_k1s1p0/feat_clusters.pt' --sideout --feat_source_list='1.maxpool'
   # 29.98% -> 29.63%

   python test_few_shot.py --shot 1 --gpu 3 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout-avgpool/max-va.pth --method='cluster-cos-spatial-softmax' --dist_func_list='cos' --logits_coeff_list='20' --disable_out_feat_flatten --feat_clusters_filename='/home/shawn/weijian/data/conv4sideout_avgpool/train_2.maxpool_spatial_1x1_k1s1p0/feat_clusters.pt' --sideout --feat_source_list='2.maxpool'
   # 34.21% -> 34.25%

   python test_few_shot.py --shot 1 --gpu 4 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout-avgpool/max-va.pth --method='cluster-cos-spatial-softmax' --dist_func_list='cos' --logits_coeff_list='20' --disable_out_feat_flatten --feat_clusters_filename='/home/shawn/weijian/data/conv4sideout_avgpool/train_3.maxpool_spatial_1x1_k1s1p0/feat_clusters.pt' --sideout --feat_source_list='3.maxpool'
   # 41.02% -> 41.36%





   # Final + cluster-cos-spatial-softmax.
   python test_few_shot.py --shot 1 --gpu 5 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout-avgpool/max-va.pth --method='cos,cluster-cos-spatial-softmax' --dist_func_list='none,cos' --logits_coeff_list='20,20' --disable_out_feat_flatten --feat_clusters_filename='/home/shawn/weijian/data/conv4sideout_avgpool/train_0.maxpool_spatial_1x1_k1s1p0/feat_clusters.pt' --sideout --feat_source_list='final,0.maxpool'
   # 50.20% -> 50.09%

   python test_few_shot.py --shot 1 --gpu 6 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout-avgpool/max-va.pth --method='cos,cluster-cos-spatial-softmax' --dist_func_list='none,cos' --logits_coeff_list='20,20' --disable_out_feat_flatten --feat_clusters_filename='/home/shawn/weijian/data/conv4sideout_avgpool/train_1.maxpool_spatial_1x1_k1s1p0/feat_clusters.pt' --sideout --feat_source_list='final,1.maxpool'
   # 49.41% -> 49.36%

   python test_few_shot.py --shot 1 --gpu 7 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout-avgpool/max-va.pth --method='cos,cluster-cos-spatial-softmax' --dist_func_list='none,cos' --logits_coeff_list='20,20' --disable_out_feat_flatten --feat_clusters_filename='/home/shawn/weijian/data/conv4sideout_avgpool/train_2.maxpool_spatial_1x1_k1s1p0/feat_clusters.pt' --sideout --feat_source_list='final,2.maxpool'
   # 50.50% -> 50.27%

   python test_few_shot.py --shot 1 --gpu 8 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout-avgpool/max-va.pth --method='cos,cluster-cos-spatial-softmax' --dist_func_list='none,cos' --logits_coeff_list='20,20' --disable_out_feat_flatten --feat_clusters_filename='/home/shawn/weijian/data/conv4sideout_avgpool/train_3.maxpool_spatial_1x1_k1s1p0/feat_clusters.pt' --sideout --feat_source_list='final,3.maxpool'
   # 50.34% -> 50.45%



   ```


   Clustering: Last layer 1x1 - More clusters.
   ```bash
   # (out of mem) python calc_clusters.py --show_cur_iter --feat_path="/home/shawn/weijian/data/conv4sideout_avgpool/train_0.maxpool_all_feat.pt" --feat_process="spatial" --save_name "train_0.maxpool_spatial_1x1_k1s1p0_200clusters" --stage=0 --kernel_size=1 --stride=1 --padding=0 --num_clusters=200

   python calc_clusters.py --show_cur_iter --feat_path="/home/shawn/weijian/data/conv4sideout_avgpool/train_1.maxpool_all_feat.pt" --feat_process="spatial" --save_name "train_1.maxpool_spatial_1x1_k1s1p0_200clusters" --stage=1 --kernel_size=1 --stride=1 --padding=0 --num_clusters=200

   python calc_clusters.py --show_cur_iter --feat_path="/home/shawn/weijian/data/conv4sideout_avgpool/train_2.maxpool_all_feat.pt" --feat_process="spatial" --save_name "train_2.maxpool_spatial_1x1_k1s1p0_200clusters" --stage=2 --kernel_size=1 --stride=1 --padding=0 --num_clusters=200

   python calc_clusters.py --gpu 9 --feat_path="/home/shawn/weijian/data/conv4sideout_avgpool/train_3.maxpool_all_feat.pt" --feat_process="spatial" --save_name "train_3.maxpool_spatial_1x1_k1s1p0_200clusters" --stage=3 --kernel_size=1 --stride=1 --padding=0 --num_clusters=200



   # (out of mem) python calc_clusters.py --show_cur_iter --feat_path="/home/shawn/weijian/data/conv4sideout_avgpool/train_0.maxpool_all_feat.pt" --feat_process="spatial" --save_name "train_0.maxpool_spatial_1x1_k1s1p0_1000clusters" --stage=0 --kernel_size=1 --stride=1 --padding=0 --num_clusters=1000

   # (out of mem) python calc_clusters.py --show_cur_iter --feat_path="/home/shawn/weijian/data/conv4sideout_avgpool/train_1.maxpool_all_feat.pt" --feat_process="spatial" --save_name "train_1.maxpool_spatial_1x1_k1s1p0_1000clusters" --stage=1 --kernel_size=1 --stride=1 --padding=0 --num_clusters=1000

   python calc_clusters.py --show_cur_iter --feat_path="/home/shawn/weijian/data/conv4sideout_avgpool/train_2.maxpool_all_feat.pt" --feat_process="spatial" --save_name "train_2.maxpool_spatial_1x1_k1s1p0_1000clusters" --stage=2 --kernel_size=1 --stride=1 --padding=0 --num_clusters=1000

   python calc_clusters.py --gpu 4 --feat_path="/home/shawn/weijian/data/conv4sideout_avgpool/train_3.maxpool_all_feat.pt" --feat_process="spatial" --save_name "train_3.maxpool_spatial_1x1_k1s1p0_1000clusters" --stage=3 --kernel_size=1 --stride=1 --padding=0 --num_clusters=1000
   
   
   
   
   # Final + cluster-cos-spatial.
   # python test_few_shot.py --shot 1 --gpu 1 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout-avgpool/max-va.pth --method='cos,cluster-cos-spatial' --dist_func_list='none,cos' --logits_coeff_list='20,20' --disable_out_feat_flatten --feat_clusters_filename='/home/shawn/weijian/data/conv4sideout_avgpool/train_0.maxpool_spatial_1x1_k1s1p0_200clusters/feat_clusters.pt' --sideout --feat_source_list='final,0.maxpool'
   # 

   python test_few_shot.py --shot 1 --gpu 2 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout-avgpool/max-va.pth --method='cos,cluster-cos-spatial' --dist_func_list='none,cos' --logits_coeff_list='20,20' --disable_out_feat_flatten --feat_clusters_filename='/home/shawn/weijian/data/conv4sideout_avgpool/train_1.maxpool_spatial_1x1_k1s1p0_200clusters/feat_clusters.pt' --sideout --feat_source_list='final,1.maxpool'
   # 51.73% -> 51.57%

   python test_few_shot.py --shot 1 --gpu 3 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout-avgpool/max-va.pth --method='cos,cluster-cos-spatial' --dist_func_list='none,cos' --logits_coeff_list='20,20' --disable_out_feat_flatten --feat_clusters_filename='/home/shawn/weijian/data/conv4sideout_avgpool/train_2.maxpool_spatial_1x1_k1s1p0_200clusters/feat_clusters.pt' --sideout --feat_source_list='final,2.maxpool'
   # 51.52% -> 51.27%

   python test_few_shot.py --shot 1 --gpu 4 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout-avgpool/max-va.pth --method='cos,cluster-cos-spatial' --dist_func_list='none,cos' --logits_coeff_list='20,20' --disable_out_feat_flatten --feat_clusters_filename='/home/shawn/weijian/data/conv4sideout_avgpool/train_3.maxpool_spatial_1x1_k1s1p0_200clusters/feat_clusters.pt' --sideout --feat_source_list='final,3.maxpool'
   # 51.48% -> 51.35%





   # Final + cluster-cos-spatial.
   # python test_few_shot.py --shot 1 --gpu 1 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout-avgpool/max-va.pth --method='cos,cluster-cos-spatial' --dist_func_list='none,cos' --logits_coeff_list='20,20' --disable_out_feat_flatten --feat_clusters_filename='/home/shawn/weijian/data/conv4sideout_avgpool/train_0.maxpool_spatial_1x1_k1s1p0_200clusters/feat_clusters.pt' --sideout --feat_source_list='final,0.maxpool'
   # 

   # python test_few_shot.py --shot 1 --gpu 2 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout-avgpool/max-va.pth --method='cos,cluster-cos-spatial' --dist_func_list='none,cos' --logits_coeff_list='20,20' --disable_out_feat_flatten --feat_clusters_filename='/home/shawn/weijian/data/conv4sideout_avgpool/train_1.maxpool_spatial_1x1_k1s1p0_200clusters/feat_clusters.pt' --sideout --feat_source_list='final,1.maxpool'
   # 

   python test_few_shot.py --shot 1 --gpu 3 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout-avgpool/max-va.pth --method='cos,cluster-cos-spatial' --dist_func_list='none,cos' --logits_coeff_list='20,20' --disable_out_feat_flatten --feat_clusters_filename='/home/shawn/weijian/data/conv4sideout_avgpool/train_2.maxpool_spatial_1x1_k1s1p0_1000clusters/feat_clusters.pt' --sideout --feat_source_list='final,2.maxpool'
   # 51.53% -> 51.29%

   python test_few_shot.py --shot 1 --gpu 4 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout-avgpool/max-va.pth --method='cos,cluster-cos-spatial' --dist_func_list='none,cos' --logits_coeff_list='20,20' --disable_out_feat_flatten --feat_clusters_filename='/home/shawn/weijian/data/conv4sideout_avgpool/train_3.maxpool_spatial_1x1_k1s1p0_1000clusters/feat_clusters.pt' --sideout --feat_source_list='final,3.maxpool'
   # 51.49% -> 51.37%
   ```





12. Re-train res12 with side outputs.

   Train:
   ```bash
   python train_classifier.py --gpu 0 --config ./configs/current_configs/sideout/train_classifier_mini_res12sideout.yaml

   python train_classifier.py --gpu 1 --config ./configs/current_configs/sideout/train_classifier_mini_res12sideout.yaml --tag=trial2

   python train_classifier.py --gpu 2 --config ./configs/current_configs/sideout/train_classifier_mini_res12sideout.yaml --tag=trial3
   ```

   Evaluation:
   ```bash
   python test_few_shot.py --shot 1 --gpu 0 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/res12sideout/max-va.pth --method='cos' --dist_func_list='none' --logits_coeff_list='1.0'
   # 61.27% -> 61.10%
   
   python test_few_shot.py --shot 1 --gpu 1 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/res12sideout_trial2/max-va.pth --method='cos' --dist_func_list='none' --logits_coeff_list='1.0'
   # 60.60% -> 60.78%

   python test_few_shot.py --shot 1 --gpu 2 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/res12sideout_trial3/max-va.pth --method='cos' --dist_func_list='none' --logits_coeff_list='1.0'
   # 61.24% -> 61.28%
   ```


   Feature extraction.
   ```bash
   python feat_extract.py --gpu 3 --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/res12sideout/max-va.pth --split train --sideout --feat_source="0.maxpool" --save_path=/home/shawn/weijian/data/res12sideout

   python feat_extract.py --gpu 4 --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/res12sideout/max-va.pth --split train --sideout --feat_source="1.maxpool" --save_path=/home/shawn/weijian/data/res12sideout

   python feat_extract.py --gpu 5 --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/res12sideout/max-va.pth --split train --sideout --feat_source="2.maxpool" --save_path=/home/shawn/weijian/data/res12sideout

   python feat_extract.py --gpu 6 --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/res12sideout/max-va.pth --split train --sideout --feat_source="3.maxpool" --save_path=/home/shawn/weijian/data/res12sideout
   ```



   Test with other features (avgpool / flatten features).
   ```bash
   # Only avgpool.
   python test_few_shot.py --shot 1 --gpu 0 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/res12sideout/max-va.pth --method='avgpool' --dist_func_list='cos' --logits_coeff_list='20' --disable_out_feat_flatten --sideout --feat_source_list='0.maxpool'
   # 32.92% -> 32.75%

   python test_few_shot.py --shot 1 --gpu 1 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/res12sideout/max-va.pth --method='avgpool' --dist_func_list='cos' --logits_coeff_list='20' --disable_out_feat_flatten --sideout --feat_source_list='1.maxpool'
   # 38.75% -> 38.78%

   python test_few_shot.py --shot 1 --gpu 2 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/res12sideout/max-va.pth --method='avgpool' --dist_func_list='cos' --logits_coeff_list='20' --disable_out_feat_flatten --sideout --feat_source_list='2.maxpool'
   # 47.03% -> 46.82%

   python test_few_shot.py --shot 1 --gpu 3 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/res12sideout/max-va.pth --method='avgpool' --dist_func_list='cos' --logits_coeff_list='20' --disable_out_feat_flatten --sideout --feat_source_list='3.maxpool'
   # 61.27% -> 61.10%





   # Final + avgpool.
   python test_few_shot.py --shot 1 --gpu 4 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/res12sideout/max-va.pth --method='cos,avgpool' --dist_func_list='none,cos' --logits_coeff_list='20,20' --disable_out_feat_flatten --sideout --feat_source_list='final,0.maxpool'
   # 57.38% -> 57.24%

   python test_few_shot.py --shot 1 --gpu 5 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/res12sideout/max-va.pth --method='cos,avgpool' --dist_func_list='none,cos' --logits_coeff_list='20,20' --disable_out_feat_flatten --sideout --feat_source_list='final,1.maxpool'
   # 58.54% -> 58.48%

   python test_few_shot.py --shot 1 --gpu 6 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/res12sideout/max-va.pth --method='cos,avgpool' --dist_func_list='none,cos' --logits_coeff_list='20,20' --disable_out_feat_flatten --sideout --feat_source_list='final,2.maxpool'
   # 59.68% -> 59.51%

   python test_few_shot.py --shot 1 --gpu 7 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/res12sideout/max-va.pth --method='cos,avgpool' --dist_func_list='none,cos' --logits_coeff_list='20,20' --disable_out_feat_flatten --sideout --feat_source_list='final,3.maxpool'
   # 61.27% -> 61.10%







   # Only flatten.
   python test_few_shot.py --shot 1 --gpu 0 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/res12sideout/max-va.pth --method='cos' --dist_func_list='none' --logits_coeff_list='20' --disable_out_feat_flatten --sideout --feat_source_list='0.maxpool'
   # 31.73% -> 31.13%

   python test_few_shot.py --shot 1 --gpu 1 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/res12sideout/max-va.pth --method='cos' --dist_func_list='none' --logits_coeff_list='20' --disable_out_feat_flatten --sideout --feat_source_list='1.maxpool'
   # 35.51% -> 35.29%

   python test_few_shot.py --shot 1 --gpu 2 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/res12sideout/max-va.pth --method='cos' --dist_func_list='none' --logits_coeff_list='20' --disable_out_feat_flatten --sideout --feat_source_list='2.maxpool'
   # 39.88% -> 39.90%

   python test_few_shot.py --shot 1 --gpu 3 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/res12sideout/max-va.pth --method='cos' --dist_func_list='none' --logits_coeff_list='20' --disable_out_feat_flatten --sideout --feat_source_list='3.maxpool'
   # 59.45% -> 59.12%



   
    # Final + flatten.
   python test_few_shot.py --shot 1 --gpu 1 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/res12sideout/max-va.pth --method='cos,cos' --dist_func_list='none,none' --logits_coeff_list='20,20' --disable_out_feat_flatten --sideout --feat_source_list='final,0.maxpool'
   # 59.62% -> 59.37%

   python test_few_shot.py --shot 1 --gpu 2 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/res12sideout/max-va.pth --method='cos,cos' --dist_func_list='none,none' --logits_coeff_list='20,20' --disable_out_feat_flatten --sideout --feat_source_list='final,1.maxpool'
   # 60.88% -> 60.65%

   python test_few_shot.py --shot 1 --gpu 3 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/res12sideout/max-va.pth --method='cos,cos' --dist_func_list='none,none' --logits_coeff_list='20,20' --disable_out_feat_flatten --sideout --feat_source_list='final,2.maxpool'
   # 61.76% -> 61.45%

   python test_few_shot.py --shot 1 --gpu 4 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/res12sideout/max-va.pth --method='cos,cos' --dist_func_list='none,none' --logits_coeff_list='20,20' --disable_out_feat_flatten --sideout --feat_source_list='final,3.maxpool'
   # 61.70% -> 61.52%
   ```



   Clustering: Each layer 1x1.
   ```bash
   python calc_clusters.py --show_cur_iter --feat_path="/home/shawn/weijian/data/res12sideout/train_0.maxpool_all_feat.pt" --feat_process="spatial" --save_name "train_0.maxpool_spatial_1x1_k1s1p0" --stage=0 --kernel_size=1 --stride=1 --padding=0

   python calc_clusters.py --show_cur_iter --feat_path="/home/shawn/weijian/data/res12sideout/train_1.maxpool_all_feat.pt" --feat_process="spatial" --save_name "train_1.maxpool_spatial_1x1_k1s1p0" --stage=1 --kernel_size=1 --stride=1 --padding=0

   python calc_clusters.py --show_cur_iter --feat_path="/home/shawn/weijian/data/res12sideout/train_2.maxpool_all_feat.pt" --feat_process="spatial" --save_name "train_2.maxpool_spatial_1x1_k1s1p0" --stage=2 --kernel_size=1 --stride=1 --padding=0

   python calc_clusters.py --gpu 3 --feat_path="/home/shawn/weijian/data/res12sideout/train_3.maxpool_all_feat.pt" --feat_process="spatial" --save_name "train_3.maxpool_spatial_1x1_k1s1p0" --stage=3 --kernel_size=1 --stride=1 --padding=0

   python calc_clusters.py --gpu 3 --feat_path="/home/shawn/weijian/data/res12sideout/train_3.maxpool_all_feat.pt" --feat_process="spatial" --save_name "train_3.maxpool_spatial_1x1_k1s1p0_cos-shift" --stage=3 --kernel_size=1 --stride=1 --padding=0 --heatmap_mode='cos-shift'
   ```


   Test clustering.
   ```bash
   # Only cluster-cos-spatial.
   python test_few_shot.py --shot 1 --gpu 0 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/res12sideout/max-va.pth --method='cluster-cos-spatial' --dist_func_list='cos' --logits_coeff_list='20' --disable_out_feat_flatten --feat_clusters_filename='/home/shawn/weijian/data/res12sideout/train_0.maxpool_spatial_1x1_k1s1p0/feat_clusters.pt' --sideout --feat_source_list='0.maxpool'
   # -

   python test_few_shot.py --shot 1 --gpu 1 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/res12sideout/max-va.pth --method='cluster-cos-spatial' --dist_func_list='cos' --logits_coeff_list='20' --disable_out_feat_flatten --feat_clusters_filename='/home/shawn/weijian/data/res12sideout/train_1.maxpool_spatial_1x1_k1s1p0/feat_clusters.pt' --sideout --feat_source_list='1.maxpool'
   # 28.67% -> 28.44%

   python test_few_shot.py --shot 1 --gpu 2 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/res12sideout/max-va.pth --method='cluster-cos-spatial' --dist_func_list='cos' --logits_coeff_list='20' --disable_out_feat_flatten --feat_clusters_filename='/home/shawn/weijian/data/res12sideout/train_2.maxpool_spatial_1x1_k1s1p0/feat_clusters.pt' --sideout --feat_source_list='2.maxpool'
   # 30.21% -> 30.56%

   python test_few_shot.py --shot 1 --gpu 3 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/res12sideout/max-va.pth --method='cluster-cos-spatial' --dist_func_list='cos' --logits_coeff_list='20' --disable_out_feat_flatten --feat_clusters_filename='/home/shawn/weijian/data/res12sideout/train_3.maxpool_spatial_1x1_k1s1p0/feat_clusters.pt' --sideout --feat_source_list='3.maxpool'
   # 45.39% -> 45.24%








   # Final + cluster-cos-spatial.
   python test_few_shot.py --shot 1 --gpu 4 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/res12sideout/max-va.pth --method='cos,cluster-cos-spatial' --dist_func_list='none,cos' --logits_coeff_list='20,20' --disable_out_feat_flatten --feat_clusters_filename='/home/shawn/weijian/data/res12sideout/train_0.maxpool_spatial_1x1_k1s1p0/feat_clusters.pt' --sideout --feat_source_list='final,0.maxpool'
   # -

   python test_few_shot.py --shot 1 --gpu 5 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/res12sideout/max-va.pth --method='cos,cluster-cos-spatial' --dist_func_list='none,cos' --logits_coeff_list='20,20' --disable_out_feat_flatten --feat_clusters_filename='/home/shawn/weijian/data/res12sideout/train_1.maxpool_spatial_1x1_k1s1p0/feat_clusters.pt' --sideout --feat_source_list='final,1.maxpool'
   # 60.92% -> 60.76%

   python test_few_shot.py --shot 1 --gpu 6 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/res12sideout/max-va.pth --method='cos,cluster-cos-spatial' --dist_func_list='none,cos' --logits_coeff_list='20,20' --disable_out_feat_flatten --feat_clusters_filename='/home/shawn/weijian/data/res12sideout/train_2.maxpool_spatial_1x1_k1s1p0/feat_clusters.pt' --sideout --feat_source_list='final,2.maxpool'
   # 61.46% -> 61.24%

   python test_few_shot.py --shot 1 --gpu 7 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/res12sideout/max-va.pth --method='cos,cluster-cos-spatial' --dist_func_list='none,cos' --logits_coeff_list='20,20' --disable_out_feat_flatten --feat_clusters_filename='/home/shawn/weijian/data/res12sideout/train_3.maxpool_spatial_1x1_k1s1p0/feat_clusters.pt' --sideout --feat_source_list='final,3.maxpool'
   # 60.35% -> 60.23%
   ```






13. Mini-batch clustering.

   ```bash
   python calc_clusters.py --gpu 3 --feat_path="/home/shawn/weijian/data/conv4sideout_avgpool/train_3.maxpool_all_feat.pt" --feat_process="spatial" --save_name "train_3.maxpool_spatial_1x1_k1s1p0_minibatch_batchsize960000" --stage=3 --kernel_size=1 --stride=1 --padding=0 --cluster_mode="minibatch" --batch_size=960000

   python calc_clusters.py --gpu 3 --feat_path="/home/shawn/weijian/data/conv4sideout_avgpool/train_3.maxpool_all_feat.pt" --feat_process="spatial" --save_name "train_3.maxpool_spatial_1x1_k1s1p0_minibatch_batchsize38400" --stage=3 --kernel_size=1 --stride=1 --padding=0 --cluster_mode="minibatch" --batch_size=38400

   python calc_clusters.py --gpu 3 --feat_path="/home/shawn/weijian/data/conv4sideout_avgpool/train_3.maxpool_all_feat.pt" --feat_process="spatial" --save_name "train_3.maxpool_spatial_1x1_k1s1p0_minibatch_batchsize128" --stage=3 --kernel_size=1 --stride=1 --padding=0 --cluster_mode="minibatch" --batch_size=128

   python calc_clusters.py --gpu 0 --feat_path="/home/shawn/weijian/data/conv4sideout_avgpool/train_3.maxpool_all_feat.pt" --feat_process="spatial" --save_name "train_3.maxpool_spatial_1x1_k1s1p0_minibatch_batchsize128_alpha1.0" --stage=3 --kernel_size=1 --stride=1 --padding=0 --cluster_mode="minibatch" --batch_size=128 --alpha=1.0

   python calc_clusters.py --gpu 1 --feat_path="/home/shawn/weijian/data/conv4sideout_avgpool/train_3.maxpool_all_feat.pt" --feat_process="spatial" --save_name "train_3.maxpool_spatial_1x1_k1s1p0_minibatch_batchsize128_alpha0.5" --stage=3 --kernel_size=1 --stride=1 --padding=0 --cluster_mode="minibatch" --batch_size=128 --alpha=0.5

   python calc_clusters.py --gpu 2 --feat_path="/home/shawn/weijian/data/conv4sideout_avgpool/train_3.maxpool_all_feat.pt" --feat_process="spatial" --save_name "train_3.maxpool_spatial_1x1_k1s1p0_minibatch_batchsize128_alpha0.1" --stage=3 --kernel_size=1 --stride=1 --padding=0 --cluster_mode="minibatch" --batch_size=128 --alpha=0.1

   python calc_clusters.py --gpu 2 --feat_path="/home/shawn/weijian/data/conv4sideout_avgpool/train_3.maxpool_all_feat.pt" --feat_process="spatial" --save_name "train_3.maxpool_spatial_1x1_k1s1p0_minibatch_batchsize128_alpha0.01" --stage=3 --kernel_size=1 --stride=1 --padding=0 --cluster_mode="minibatch" --batch_size=128 --alpha=0.01

   python calc_clusters.py --gpu 2 --feat_path="/home/shawn/weijian/data/conv4sideout_avgpool/train_3.maxpool_all_feat.pt" --feat_process="spatial" --save_name "train_3.maxpool_spatial_1x1_k1s1p0_minibatch_batchsize128_alpha0.001" --stage=3 --kernel_size=1 --stride=1 --padding=0 --cluster_mode="minibatch" --batch_size=128 --alpha=0.001

   python calc_clusters.py --gpu 2 --feat_path="/home/shawn/weijian/data/conv4sideout_avgpool/train_3.maxpool_all_feat.pt" --feat_process="spatial" --save_name "train_3.maxpool_spatial_1x1_k1s1p0_minibatch_batchsize128_alpha0.01_shuffle" --stage=3 --kernel_size=1 --stride=1 --padding=0 --cluster_mode="minibatch" --batch_size=128 --alpha=0.01 --shuffle

   python calc_clusters.py --gpu 2 --feat_path="/home/shawn/weijian/data/conv4sideout_avgpool/train_3.maxpool_all_feat.pt" --feat_process="spatial" --save_name "train_3.maxpool_spatial_1x1_k1s1p0_minibatch_batchsize128_shuffle_sculley-like" --stage=3 --kernel_size=1 --stride=1 --padding=0 --cluster_mode="minibatch" --batch_size=128 --shuffle --minibatch="sculley-like"  # It seems to work. It has similar histograms as sklearn minibatch k-means (batch size = 128).

   python calc_clusters.py --gpu 2 --feat_path="/home/shawn/weijian/data/conv4sideout_avgpool/train_3.maxpool_all_feat.pt" --feat_process="spatial" --save_name "train_3.maxpool_spatial_1x1_k1s1p0_minibatch_batchsize128_shuffle_sculley-like_iter1" --stage=3 --kernel_size=1 --stride=1 --padding=0 --cluster_mode="minibatch" --batch_size=128 --shuffle --minibatch="sculley-like" --iterations=1

   python calc_clusters.py --gpu 2 --feat_path="/home/shawn/weijian/data/conv4sideout_avgpool/train_3.maxpool_all_feat.pt" --feat_process="spatial" --save_name "train_3.maxpool_spatial_1x1_k1s1p0_minibatch_batchsize128_shuffle_sculley-like_iter1_eps0.0001" --stage=3 --kernel_size=1 --stride=1 --padding=0 --cluster_mode="minibatch" --batch_size=128 --shuffle --minibatch="sculley-like" --iterations=1 --eps=0.0001

   python calc_clusters.py --gpu 2 --feat_path="/home/shawn/weijian/data/conv4sideout_avgpool/train_3.maxpool_all_feat.pt" --feat_process="spatial" --save_name "train_3.maxpool_spatial_1x1_k1s1p0_minibatch_batchsize128_shuffle_sculley-like_iter1_eps0.0001_trial2" --stage=3 --kernel_size=1 --stride=1 --padding=0 --cluster_mode="minibatch" --batch_size=128 --shuffle --minibatch="sculley-like" --iterations=1 --eps=0.0001

   python calc_clusters.py --gpu 2 --feat_path="/home/shawn/weijian/data/conv4sideout_avgpool/train_3.maxpool_all_feat.pt" --feat_process="spatial" --save_name "train_3.maxpool_spatial_1x1_k1s1p0_minibatch_batchsize128_shuffle_sculley-like_iter1_200clusters" --stage=3 --kernel_size=1 --stride=1 --padding=0 --cluster_mode="minibatch" --batch_size=128 --shuffle --minibatch="sculley-like" --iterations=1 --num_clusters=200

   python calc_clusters.py --gpu 2 --feat_path="/home/shawn/weijian/data/conv4sideout_avgpool/train_3.maxpool_all_feat.pt" --feat_process="spatial" --save_name "train_3.maxpool_spatial_1x1_k1s1p0_minibatch_batchsize128_shuffle_sculley-like_iter10" --stage=3 --kernel_size=1 --stride=1 --padding=0 --cluster_mode="minibatch" --batch_size=128 --shuffle --minibatch="sculley-like" --iterations=10

   python calc_clusters.py --gpu 2 --feat_path="/home/shawn/weijian/data/conv4sideout_avgpool/train_3.maxpool_all_feat.pt" --feat_process="spatial" --save_name "train_3.maxpool_spatial_1x1_k1s1p0_minibatch_batchsize128_shuffle_sculley-like_kmeans" --stage=3 --kernel_size=1 --stride=1 --padding=0 --cluster_mode="minibatch" --batch_size=128 --shuffle --minibatch="sculley-like" --cluster_type='k-means'  # Worse.

   python calc_clusters.py --gpu 2 --feat_path="/home/shawn/weijian/data/conv4sideout_avgpool/train_3.maxpool_all_feat.pt" --feat_process="spatial" --save_name "train_3.maxpool_spatial_1x1_k1s1p0_minibatch_batchsize128_shuffle_sklearn" --stage=3 --kernel_size=1 --stride=1 --padding=0 --cluster_mode="minibatch" --batch_size=128 --shuffle --minibatch="sklearn" # It is worse... interesting.

   python calc_clusters.py --gpu 2 --feat_path="/home/shawn/weijian/data/conv4sideout_avgpool/train_3.maxpool_all_feat.pt" --feat_process="spatial" --save_name "train_3.maxpool_spatial_1x1_k1s1p0_minibatch_batchsize128_shuffle_sklearn_iter1" --stage=3 --kernel_size=1 --stride=1 --padding=0 --cluster_mode="minibatch" --batch_size=128 --shuffle --minibatch="sklearn" --iterations=1 

   python calc_clusters.py --gpu 3 --feat_path="/home/shawn/weijian/data/conv4sideout_avgpool/train_3.maxpool_all_feat.pt" --feat_process="spatial" --save_name "train_3.maxpool_spatial_1x1_k1s1p0_minibatch_batchsize128_shuffle_sklearn_kmeans" --stage=3 --kernel_size=1 --stride=1 --padding=0 --cluster_mode="minibatch" --batch_size=128 --shuffle --minibatch="sklearn" --cluster_type='k-means' # Worse.

   python calc_clusters.py --gpu 3 --feat_path="/home/shawn/weijian/data/conv4sideout_avgpool/train_3.maxpool_all_feat.pt" --feat_process="spatial" --save_name "train_3.maxpool_spatial_1x1_k1s1p0_minibatch_batchsize128_shuffle_sklearn_kmeans_iter1_eps0.0001" --stage=3 --kernel_size=1 --stride=1 --padding=0 --cluster_mode="minibatch" --batch_size=128 --shuffle --minibatch="sklearn" --cluster_type='k-means' --iterations=1 --eps=0.0001

   python calc_clusters.py --gpu 3 --feat_path="/home/shawn/weijian/data/conv4sideout_avgpool/train_3.maxpool_all_feat.pt" --feat_process="spatial" --save_name "train_3.maxpool_spatial_1x1_k1s1p0_minibatch_batchsize128_shuffle_sklearn_kmeans_32clusters" --stage=3 --kernel_size=1 --stride=1 --padding=0 --cluster_mode="minibatch" --batch_size=128 --shuffle --minibatch="sklearn" --cluster_type='k-means' --num_clusters=32 

   python calc_clusters.py --gpu 3 --feat_path="/home/shawn/weijian/data/conv4sideout_avgpool/train_3.maxpool_all_feat.pt" --feat_process="spatial" --save_name "train_3.maxpool_spatial_1x1_k1s1p0_minibatch_batchsize12800_shuffle_sklearn_kmeans" --stage=3 --kernel_size=1 --stride=1 --padding=0 --cluster_mode="minibatch" --batch_size=128 --shuffle --minibatch="sklearn" --cluster_type='k-means' --batch_size=12800  # Pretty ok. Large batch size seems good.

   python calc_clusters.py --feat_path="/home/shawn/weijian/data/conv4sideout_avgpool/train_3.maxpool_all_feat.pt" --feat_process="spatial" --save_name "train_3.maxpool_spatial_1x1_k1s1p0_sklearn-minibatch-kmeans-cos-batchsize128_trial2" --stage=3 --kernel_size=1 --stride=1 --padding=0 --cluster_mode='sklearn-minibatch-kmeans-cos' --batch_size=128
   
   python calc_clusters.py --feat_path="/home/shawn/weijian/data/conv4sideout_avgpool/train_3.maxpool_all_feat.pt" --feat_process="spatial" --save_name "train_3.maxpool_spatial_1x1_k1s1p0_sklearn-minibatch-kmeans-cos-batchsize128-cluster_init-custom-reassign0.0" --stage=3 --kernel_size=1 --stride=1 --padding=0 --cluster_mode='sklearn-minibatch-kmeans-cos' --batch_size=128 --cluster_init='custom' --reassignment_ratio=0.0






   python calc_clusters.py --gpu 0 --feat_path="/home/shawn/weijian/data/conv4sideout_avgpool/train_3.maxpool_all_feat.pt" --feat_process="spatial" --save_name "train_3.maxpool_spatial_5x5_k5s5p0_minibatch_batchsize128_shuffle_sculley-like" --stage=3 --kernel_size=5 --stride=5 --padding=0 --cluster_mode="minibatch" --batch_size=128 --shuffle --minibatch="sculley-like" 

   python calc_clusters.py --gpu 1 --feat_path="/home/shawn/weijian/data/conv4sideout_avgpool/train_3.maxpool_all_feat.pt" --feat_process="spatial" --save_name "train_3.maxpool_spatial_5x5_k5s5p0_minibatch_batchsize128_shuffle_sculley-like_K1" --stage=3 --kernel_size=5 --stride=5 --padding=0 --cluster_mode="minibatch" --batch_size=128 --shuffle --minibatch="sculley-like" --K=1.0

   python calc_clusters.py --gpu 2 --feat_path="/home/shawn/weijian/data/conv4sideout_avgpool/train_3.maxpool_all_feat.pt" --feat_process="spatial" --save_name "train_3.maxpool_spatial_5x5_k5s5p0_minibatch_batchsize128_shuffle_sculley-like_kmeans" --stage=3 --kernel_size=5 --stride=5 --padding=0 --cluster_mode="minibatch" --batch_size=128 --shuffle --minibatch="sculley-like" --cluster_type='k-means'

   python calc_clusters.py --gpu 3 --feat_path="/home/shawn/weijian/data/conv4sideout_avgpool/train_3.maxpool_all_feat.pt" --feat_process="spatial" --save_name "train_3.maxpool_spatial_5x5_k5s5p0_minibatch_batchsize128_shuffle_sculley-like_kmeans_argmax" --stage=3 --kernel_size=5 --stride=5 --padding=0 --cluster_mode="minibatch" --batch_size=128 --shuffle --minibatch="sculley-like" --cluster_type='k-means' --max_type='argmax'
   # For 5x5 patches, the sculley-like performance is much worse than sklearn minibatch k-means, but both are worse than normal batch kmeans.
   # Perhaps it is because 5x5 patches has more dimensions, which is harder. Our "sculley-like" cannot deal with that hard task.
   # 1. Need to check the algorithm difference.
   # 2. Another possible reason can be the initialization - Partially matters!
   # 3. Another possible reason can be the total iterations - Seems nope.
   # 4. The reassignment strategy in sklearn - Partially yes! Reduce reassignment ratio to 0.0 make the sklearn worse.
   # 5. The update rule strategy in sklearn - Partially yes!.
   # 6. Wait... I still use iterations=100 for previous experiments, which is wrong! It should be 1.

   python calc_clusters.py --gpu 3 --feat_path="/home/shawn/weijian/data/conv4sideout_avgpool/train_3.maxpool_all_feat.pt" --feat_process="spatial" --save_name "train_3.maxpool_spatial_5x5_k5s5p0_minibatch_batchsize128_shuffle_sculley-like_kmeans_argmax_epochs2" --stage=3 --kernel_size=5 --stride=5 --padding=0 --cluster_mode="minibatch" --batch_size=128 --shuffle --minibatch="sculley-like" --cluster_type='k-means' --max_type='argmax' --epochs=2

   python calc_clusters.py --gpu 3 --feat_path="/home/shawn/weijian/data/conv4sideout_avgpool/train_3.maxpool_all_feat.pt" --feat_process="spatial" --save_name "train_3.maxpool_spatial_5x5_k5s5p0_minibatch_batchsize128_shuffle_sklearn_kmeans_argmax_epochs2" --stage=3 --kernel_size=5 --stride=5 --padding=0 --cluster_mode="minibatch" --batch_size=128 --shuffle --minibatch="sklearn" --cluster_type='k-means' --max_type='argmax' --epochs=2
   # Can reproduce some of sklearn's results!

   python calc_clusters.py --gpu 3 --feat_path="/home/shawn/weijian/data/conv4sideout_avgpool/train_3.maxpool_all_feat.pt" --feat_process="spatial" --save_name "train_3.maxpool_spatial_5x5_k5s5p0_minibatch_batchsize128_shuffle_sklearn_kmeans_argmax_epochs2_V_count_init-0.01" --stage=3 --kernel_size=5 --stride=5 --padding=0 --cluster_mode="minibatch" --batch_size=128 --shuffle --minibatch="sklearn" --cluster_type='k-means' --max_type='argmax' --epochs=2 --V_count_init=0.01
   # Can reproduce some of sklearn's results!

   python calc_clusters.py --gpu 3 --feat_path="/home/shawn/weijian/data/conv4sideout_avgpool/train_3.maxpool_all_feat.pt" --feat_process="spatial" --save_name "train_3.maxpool_spatial_5x5_k5s5p0_minibatch_batchsize1280_shuffle_sculley-like_kmeans_argmax" --stage=3 --kernel_size=5 --stride=5 --padding=0 --cluster_mode="minibatch" --batch_size=1280 --shuffle --minibatch="sculley-like" --cluster_type='k-means' --max_type='argmax'
   # Pretty bad.

   python calc_clusters.py --gpu 3 --feat_path="/home/shawn/weijian/data/conv4sideout_avgpool/train_3.maxpool_all_feat.pt" --feat_process="spatial" --save_name "train_3.maxpool_spatial_5x5_k5s5p0_minibatch_batchsize12800_shuffle_sculley-like_kmeans_argmax" --stage=3 --kernel_size=5 --stride=5 --padding=0 --cluster_mode="minibatch" --batch_size=12800 --shuffle --minibatch="sculley-like" --cluster_type='k-means' --max_type='argmax'
   # Pretty bad.

   python calc_clusters.py --gpu 3 --feat_path="/home/shawn/weijian/data/conv4sideout_avgpool/train_3.maxpool_all_feat.pt" --feat_process="spatial" --save_name "train_3.maxpool_spatial_5x5_k5s5p0_minibatch_batchsize12800_shuffle_sculley-like" --stage=3 --kernel_size=5 --stride=5 --padding=0 --cluster_mode="minibatch" --batch_size=12800 --shuffle --minibatch="sculley-like"
   # Pretty bad.

   python calc_clusters.py --gpu 3 --feat_path="/home/shawn/weijian/data/conv4sideout_avgpool/train_3.maxpool_all_feat.pt" --feat_process="spatial" --save_name "train_3.maxpool_spatial_5x5_k5s5p0_minibatch_batchsize38400_shuffle_sculley-like" --stage=3 --kernel_size=5 --stride=5 --padding=0 --cluster_mode="minibatch" --batch_size=38400 --shuffle --minibatch="sculley-like"
   # Pretty bad.

   python calc_clusters.py --feat_path="/home/shawn/weijian/data/conv4sideout_avgpool/train_3.maxpool_all_feat.pt" --feat_process="spatial" --save_name "train_3.maxpool_spatial_5x5_k5s5p0_sklearn-minibatch-kmeans-cos-batchsize128" --stage=3 --kernel_size=5 --stride=5 --padding=0 --cluster_mode='sklearn-minibatch-kmeans-cos' --batch_size=128

   python calc_clusters.py --feat_path="/home/shawn/weijian/data/conv4sideout_avgpool/train_3.maxpool_all_feat.pt" --feat_process="spatial" --save_name "train_3.maxpool_spatial_5x5_k5s5p0_sklearn-minibatch-kmeans-cos-batchsize128-cluster_init-custom" --stage=3 --kernel_size=5 --stride=5 --padding=0 --cluster_mode='sklearn-minibatch-kmeans-cos' --batch_size=128 --cluster_init='custom'

   python calc_clusters.py --feat_path="/home/shawn/weijian/data/conv4sideout_avgpool/train_3.maxpool_all_feat.pt" --feat_process="spatial" --save_name "train_3.maxpool_spatial_5x5_k5s5p0_sklearn-minibatch-kmeans-cos-batchsize128-cluster_init-custom-reassign0.0" --stage=3 --kernel_size=5 --stride=5 --padding=0 --cluster_mode='sklearn-minibatch-kmeans-cos' --batch_size=128 --cluster_init='custom' --reassignment_ratio=0.0

   python calc_clusters.py --feat_path="/home/shawn/weijian/data/conv4sideout_avgpool/train_3.maxpool_all_feat.pt" --feat_process="spatial" --save_name "train_3.maxpool_spatial_5x5_k5s5p0_sklearn-minibatch-kmeans-cos-batchsize128-cluster_init-custom-reassign0.01" --stage=3 --kernel_size=5 --stride=5 --padding=0 --cluster_mode='sklearn-minibatch-kmeans-cos' --batch_size=128 --cluster_init='custom' --reassignment_ratio=0.01 

   python calc_clusters.py --feat_path="/home/shawn/weijian/data/conv4sideout_avgpool/train_3.maxpool_all_feat.pt" --feat_process="spatial" --save_name "train_3.maxpool_spatial_5x5_k5s5p0_sklearn-minibatch-kmeans-cos-batchsize128-cluster_init-custom-reassign0.1" --stage=3 --kernel_size=5 --stride=5 --padding=0 --cluster_mode='sklearn-minibatch-kmeans-cos' --batch_size=128 --cluster_init='custom' --reassignment_ratio=0.1

   python calc_clusters.py --feat_path="/home/shawn/weijian/data/conv4sideout_avgpool/train_3.maxpool_all_feat.pt" --feat_process="spatial" --save_name "train_3.maxpool_spatial_5x5_k5s5p0_sklearn-minibatch-kmeans-cos-batchsize1280" --stage=3 --kernel_size=5 --stride=5 --padding=0 --cluster_mode='sklearn-minibatch-kmeans-cos' --batch_size=1280
   # sklearn is much better.

   python calc_clusters.py --feat_path="/home/shawn/weijian/data/conv4sideout_avgpool/train_3.maxpool_all_feat.pt" --feat_process="spatial" --save_name "train_3.maxpool_spatial_5x5_k5s5p0_sklearn-minibatch-kmeans-batchsize128" --stage=3 --kernel_size=5 --stride=5 --padding=0 --cluster_mode='sklearn-minibatch-kmeans' --batch_size=128
   ```



14. Sanity check by sklearn.
   ```bash
   python calc_clusters.py --gpu 0 --feat_path="/home/shawn/weijian/data/conv4sideout_avgpool/train_3.maxpool_all_feat.pt" --feat_process="spatial" --save_name "train_3.maxpool_spatial_5x5_k5s5p0" --stage=3 --kernel_size=5 --stride=5 --padding=0

   python calc_clusters.py --feat_path="/home/shawn/weijian/data/conv4sideout_avgpool/train_3.maxpool_all_feat.pt" --feat_process="spatial" --save_name "train_3.maxpool_spatial_5x5_k5s5p0_sklearn-minibatch-kmeans-cos" --stage=3 --kernel_size=5 --stride=5 --padding=0 --cluster_mode='sklearn-minibatch-kmeans-cos'

   python calc_clusters.py --feat_path="/home/shawn/weijian/data/conv4sideout_avgpool/train_3.maxpool_all_feat.pt" --feat_process="spatial" --save_name "train_3.maxpool_spatial_5x5_k5s5p0_sklearn-minibatch-kmeans" --stage=3 --kernel_size=5 --stride=5 --padding=0 --cluster_mode='sklearn-minibatch-kmeans'

   python calc_clusters.py --gpu 0 --feat_path="/home/shawn/weijian/data/conv4sideout_avgpool/train_3.maxpool_all_feat.pt" --feat_process="spatial" --save_name "train_3.maxpool_spatial_1x1_k1s1p0" --stage=3 --kernel_size=1 --stride=1 --padding=0

   python calc_clusters.py --feat_path="/home/shawn/weijian/data/conv4sideout_avgpool/train_3.maxpool_all_feat.pt" --feat_process="spatial" --save_name "train_3.maxpool_spatial_1x1_k1s1p0_sklearn-minibatch-kmeans-cos" --stage=3 --kernel_size=1 --stride=1 --padding=0 --cluster_mode='sklearn-minibatch-kmeans-cos'

   python calc_clusters.py --feat_path="/home/shawn/weijian/data/conv4sideout_avgpool/train_3.maxpool_all_feat.pt" --feat_process="spatial" --save_name "train_3.maxpool_spatial_1x1_k1s1p0_sklearn-minibatch-kmeans" --stage=3 --kernel_size=1 --stride=1 --padding=0 --cluster_mode='sklearn-minibatch-kmeans'

   python calc_clusters.py --feat_path="/home/shawn/weijian/data/conv4sideout_avgpool/train_3.maxpool_all_feat.pt" --feat_process="spatial" --save_name "train_3.maxpool_spatial_1x1_k1s1p0_sklearn-minibatch-kmeans-cos-batchsize128" --stage=3 --kernel_size=1 --stride=1 --padding=0 --cluster_mode='sklearn-minibatch-kmeans-cos' --batch_size=128

   python calc_clusters.py --feat_path="/home/shawn/weijian/data/conv4sideout_avgpool/train_3.maxpool_all_feat.pt" --feat_process="spatial" --save_name "train_3.maxpool_spatial_1x1_k1s1p0_sklearn-minibatch-kmeans-cos-batchsize12800" --stage=3 --kernel_size=1 --stride=1 --padding=0 --cluster_mode='sklearn-minibatch-kmeans-cos' --batch_size=12800
   ```

14. Early layer clustering before re-train.
   ```bash
   python calc_clusters.py --gpu 0 --feat_path="/home/shawn/weijian/data/conv4sideout_avgpool/train_0.maxpool_all_feat.pt" --feat_process="spatial" --save_name "train_0.maxpool_spatial_1x1_k1s1p0_minibatch_batchsize128_shuffle_sculley-like" --stage=2 --kernel_size=1 --stride=1 --padding=0 --cluster_mode="minibatch" --batch_size=128 --shuffle --minibatch="sculley-like"

   python calc_clusters.py --gpu 1 --feat_path="/home/shawn/weijian/data/conv4sideout_avgpool/train_1.maxpool_all_feat.pt" --feat_process="spatial" --save_name "train_1.maxpool_spatial_1x1_k1s1p0_minibatch_batchsize128_shuffle_sculley-like" --stage=2 --kernel_size=1 --stride=1 --padding=0 --cluster_mode="minibatch" --batch_size=128 --shuffle --minibatch="sculley-like"

   python calc_clusters.py --gpu 2 --feat_path="/home/shawn/weijian/data/conv4sideout_avgpool/train_2.maxpool_all_feat.pt" --feat_process="spatial" --save_name "train_2.maxpool_spatial_1x1_k1s1p0_minibatch_batchsize128_shuffle_sculley-like" --stage=2 --kernel_size=1 --stride=1 --padding=0 --cluster_mode="minibatch" --batch_size=128 --shuffle --minibatch="sculley-like"

   # python calc_clusters.py --gpu 3 --feat_path="/home/shawn/weijian/data/conv4sideout_avgpool/train_3.maxpool_all_feat.pt" --feat_process="spatial" --save_name "train_3.maxpool_spatial_1x1_k1s1p0_minibatch_batchsize128_shuffle_sculley-like" --stage=3 --kernel_size=1 --stride=1 --padding=0 --cluster_mode="minibatch" --batch_size=128 --shuffle --minibatch="sculley-like"

   

   python calc_clusters.py --gpu 4 --feat_path="/home/shawn/weijian/data/conv4sideout_avgpool/train_0.maxpool_all_feat.pt" --feat_process="spatial" --save_name "train_0.maxpool_spatial_1x1_k1s1p0_minibatch_batchsize128x42x42_shuffle_sculley-like" --stage=2 --kernel_size=1 --stride=1 --padding=0 --cluster_mode="minibatch" --batch_size=225792 --shuffle --minibatch="sculley-like" --num_workers=8  # Multiple workers can greatly accelerate evaluation speed. If still slow, we can try sampling later.

   python calc_clusters.py --gpu 5 --feat_path="/home/shawn/weijian/data/conv4sideout_avgpool/train_1.maxpool_all_feat.pt" --feat_process="spatial" --save_name "train_1.maxpool_spatial_1x1_k1s1p0_minibatch_batchsize128x21x21_shuffle_sculley-like" --stage=2 --kernel_size=1 --stride=1 --padding=0 --cluster_mode="minibatch" --batch_size=56448 --shuffle --minibatch="sculley-like"

   python calc_clusters.py --gpu 6 --feat_path="/home/shawn/weijian/data/conv4sideout_avgpool/train_2.maxpool_all_feat.pt" --feat_process="spatial" --save_name "train_2.maxpool_spatial_1x1_k1s1p0_minibatch_batchsize128x10x10_shuffle_sculley-like" --stage=2 --kernel_size=1 --stride=1 --padding=0 --cluster_mode="minibatch" --batch_size=12800 --shuffle --minibatch="sculley-like"

   python calc_clusters.py --gpu 7 --feat_path="/home/shawn/weijian/data/conv4sideout_avgpool/train_3.maxpool_all_feat.pt" --feat_process="spatial" --save_name "train_3.maxpool_spatial_1x1_k1s1p0_minibatch_batchsize128x5x5_shuffle_sculley-like" --stage=3 --kernel_size=1 --stride=1 --padding=0 --cluster_mode="minibatch" --batch_size=3200 --shuffle --minibatch="sculley-like"

   ```

15. (temporary) Check the effectiveness of Sculley-like trick.
   ```bash
   python test_few_shot.py --shot 1 --gpu 4 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout-avgpool/max-va.pth --method='cos,cluster-cos-spatial' --dist_func_list='none,cos' --logits_coeff_list='20,20' --disable_out_feat_flatten --feat_clusters_filename='/home/shawn/weijian/data/conv4sideout_avgpool/train_3.maxpool_spatial_1x1_k1s1p0/feat_clusters.pt' --sideout --feat_source_list='final,3.maxpool'
   # 51.47% -> 51.34%

   python test_few_shot.py --shot 1 --gpu 5 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout-avgpool/max-va.pth --method='cos,cluster-cos-spatial' --dist_func_list='none,cos' --logits_coeff_list='20,20' --disable_out_feat_flatten --feat_clusters_filename='/home/shawn/weijian/data/conv4sideout_avgpool/train_3.maxpool_spatial_1x1_k1s1p0_minibatch_batchsize128_shuffle_sculley-like/feat_clusters.pt' --sideout --feat_source_list='final,3.maxpool'
   # 51.50% -> 51.36%

   python test_few_shot.py --shot 1 --gpu 5 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout-avgpool/max-va.pth --method='cos,cluster-cos-spatial' --dist_func_list='none,cos' --logits_coeff_list='20,20' --disable_out_feat_flatten --feat_clusters_filename='/home/shawn/weijian/data/conv4sideout_avgpool/train_3.maxpool_spatial_1x1_k1s1p0_minibatch_batchsize128_shuffle_sculley-like_iter10/feat_clusters.pt' --sideout --feat_source_list='final,3.maxpool'
   # 51.45% -> 51.35%

   python test_few_shot.py --shot 1 --gpu 5 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout-avgpool/max-va.pth --method='cos,cluster-cos-spatial' --dist_func_list='none,cos' --logits_coeff_list='20,20' --disable_out_feat_flatten --feat_clusters_filename='/home/shawn/weijian/data/conv4sideout_avgpool/train_3.maxpool_spatial_1x1_k1s1p0_minibatch_batchsize128_shuffle_sculley-like_iter1/feat_clusters.pt' --sideout --feat_source_list='final,3.maxpool'
   # 51.47% -> 51.35%

   python test_few_shot.py --shot 1 --gpu 5 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout-avgpool/max-va.pth --method='cos,cluster-cos-spatial' --dist_func_list='none,cos' --logits_coeff_list='20,20' --disable_out_feat_flatten --feat_clusters_filename='/home/shawn/weijian/data/conv4sideout_avgpool/train_3.maxpool_spatial_1x1_k1s1p0_minibatch_batchsize128_shuffle_sculley-like_iter1_200clusters/feat_clusters.pt' --sideout --feat_source_list='final,3.maxpool'
   # 51.45% -> 51.36%

   python test_few_shot.py --shot 1 --gpu 5 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout-avgpool/max-va.pth --method='cos,cluster-cos-spatial' --dist_func_list='none,cos' --logits_coeff_list='20,20' --disable_out_feat_flatten --feat_clusters_filename='/home/shawn/weijian/data/conv4sideout_avgpool/train_3.maxpool_spatial_1x1_k1s1p0_minibatch_batchsize128_shuffle_sculley-like_iter1_eps0.0001/feat_clusters.pt' --sideout --feat_source_list='final,3.maxpool'
   # 51.48% -> 51.37% .... Wait: eps should not affect dynamic-routing.

   python test_few_shot.py --shot 1 --gpu 5 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout-avgpool/max-va.pth --method='cos,cluster-cos-spatial' --dist_func_list='none,cos' --logits_coeff_list='20,20' --disable_out_feat_flatten --feat_clusters_filename='/home/shawn/weijian/data/conv4sideout_avgpool/train_3.maxpool_spatial_1x1_k1s1p0_minibatch_batchsize128_shuffle_sculley-like_iter1_eps0.0001_trial2/feat_clusters.pt' --sideout --feat_source_list='final,3.maxpool'
   # 51.46% -> 51.36%

   python test_few_shot.py --shot 1 --gpu 6 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout-avgpool/max-va.pth --method='cos,cluster-cos-spatial' --dist_func_list='none,cos' --logits_coeff_list='20,20' --disable_out_feat_flatten --feat_clusters_filename='/home/shawn/weijian/data/conv4sideout_avgpool/train_3.maxpool_spatial_1x1_k1s1p0_minibatch_batchsize128_shuffle_sklearn_kmeans/feat_clusters.pt' --sideout --feat_source_list='final,3.maxpool'
   # 51.47% -> 51.33%

   python test_few_shot.py --shot 1 --gpu 6 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout-avgpool/max-va.pth --method='cos,cluster-cos-spatial' --dist_func_list='none,cos' --logits_coeff_list='20,20' --disable_out_feat_flatten --feat_clusters_filename='/home/shawn/weijian/data/conv4sideout_avgpool/train_3.maxpool_spatial_1x1_k1s1p0_minibatch_batchsize128_shuffle_sklearn_kmeans_iter1_eps0.0001/feat_clusters.pt' --sideout --feat_source_list='final,3.maxpool'
   # 51.41% -> 51.33%

   python test_few_shot.py --shot 1 --gpu 7 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4sideout-avgpool/max-va.pth --method='cos,cluster-cos-spatial' --dist_func_list='none,cos' --logits_coeff_list='20,20' --disable_out_feat_flatten --feat_clusters_filename='/home/shawn/weijian/data/conv4sideout_avgpool/train_3.maxpool_spatial_1x1_k1s1p0_minibatch_batchsize128_shuffle_sklearn_kmeans_32clusters/feat_clusters.pt' --sideout --feat_source_list='final,3.maxpool'
   # 51.42% -> 51.33%
   ```

16. Conv-4 + Feature clustering.
   Debug:
   ```bash
   python train_classifier.py --gpu 0 --config ./configs/current_configs/featcluster_minibatch/train_classifier_mini_conv4featcluster-minibatch-debug.yaml

   python train_classifier.py --gpu 1 --config ./configs/current_configs/featcluster_minibatch/FFFF.yaml
   python train_classifier.py --gpu 1 --config ./configs/current_configs/featcluster_minibatch/FFFT.yaml
   python train_classifier.py --gpu 1 --config ./configs/current_configs/featcluster_minibatch/FFTT.yaml
   python train_classifier.py --gpu 1 --config ./configs/current_configs/featcluster_minibatch/FTTT.yaml
   python train_classifier.py --gpu 1 --config ./configs/current_configs/featcluster_minibatch/TTTT.yaml


   python test_few_shot.py --shot 1 --gpu 1 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4featcluster-minibatch-FFFF/max-va.pth
   # 50.89% -> 51.00%

   python test_few_shot.py --shot 1 --gpu 1 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4featcluster-minibatch-FFFT/max-va.pth
   # 50.71% -> 50.94%

   python test_few_shot.py --shot 1 --gpu 1 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4featcluster-minibatch-FFTT/max-va.pth
   # 50.28% -> 50.40%

   python test_few_shot.py --shot 1 --gpu 1 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4featcluster-minibatch-FTTT/max-va.pth
   # 50.54% -> 50.47%

   python test_few_shot.py --shot 1 --gpu 1 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4featcluster-minibatch-TTTT/max-va.pth
   # 50.29% -> 50.44%

   ./scripts/traintest-featcluster_minibatch.sh 1 FFFT-input3x3
   ./scripts/traintest-featcluster_minibatch.sh 1 FFTT-input3x3
   ./scripts/traintest-featcluster_minibatch.sh 1 FTTT-input3x3
   ./scripts/traintest-featcluster_minibatch.sh 1 TTTT-input3x3
   ./scripts/traintest-featcluster_minibatch.sh 2 FFFT-out_conv
   ./scripts/traintest-featcluster_minibatch.sh 2 FFTT-out_conv
   ./scripts/traintest-featcluster_minibatch.sh 2 FTTT-out_conv
   ./scripts/traintest-featcluster_minibatch.sh 2 TTTT-out_conv
   ./scripts/traintest-featcluster_minibatch.sh 3 FFFT
   ./scripts/traintest-featcluster_minibatch.sh 3 FFTT
   ./scripts/traintest-featcluster_minibatch.sh 3 FTTT
   ./scripts/traintest-featcluster_minibatch.sh 3 TTTT

   python test_few_shot.py --shot 5 --gpu 0 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/conv4featcluster-minibatch-FFFF/max-va.pth

   ./scripts/traintest-featcluster_minibatch_trial.sh 1 FFFF 2
   ./scripts/traintest-featcluster_minibatch_trial.sh 1 FFFF 3
   ./scripts/traintest-featcluster_minibatch_trial.sh 1 FFFF 4
   ./scripts/traintest-featcluster_minibatch_trial.sh 1 FTTT-out_conv 3-retrain

   ./scripts/traintest-featcluster_minibatch_trial.sh 2 TFFF 2
   ./scripts/traintest-featcluster_minibatch_trial.sh 2 TTFF 2
   ./scripts/traintest-featcluster_minibatch_trial.sh 2 TTTF 2

   ./scripts/traintest-featcluster_minibatch_trial.sh 2 TFFF 3
   ./scripts/traintest-featcluster_minibatch_trial.sh 3 TTFF 3
   ./scripts/traintest-featcluster_minibatch_trial.sh 3 TTTF 3

   ./scripts/traintest-featcluster_minibatch_trial.sh 3 TFFF 4
   ./scripts/traintest-featcluster_minibatch_trial.sh 3 TTFF 4
   ./scripts/traintest-featcluster_minibatch_trial.sh 4 TTTF 4



   ./scripts/traintest-featcluster_minibatch_trial.sh 1 TFFF-input3x3 2
   ./scripts/traintest-featcluster_minibatch_trial.sh 1 TTFF-input3x3 2
   ./scripts/traintest-featcluster_minibatch_trial.sh 1 TTTF-input3x3 2

   ./scripts/traintest-featcluster_minibatch_trial.sh 1 TFFF-input3x3 3
   ./scripts/traintest-featcluster_minibatch_trial.sh 2 TTFF-input3x3 3
   ./scripts/traintest-featcluster_minibatch_trial.sh 2 TTTF-input3x3 3

   ./scripts/traintest-featcluster_minibatch_trial.sh 2 TFFF-input3x3 4
   ./scripts/traintest-featcluster_minibatch_trial.sh 2 TTFF-input3x3 4
   ./scripts/traintest-featcluster_minibatch_trial.sh 3 TTTF-input3x3 4

   ./scripts/traintest-featcluster_minibatch_trial.sh 3 TFFF-out_conv 2
   ./scripts/traintest-featcluster_minibatch_trial.sh 3 TTFF-out_conv 2
   ./scripts/traintest-featcluster_minibatch_trial.sh 3 TTTF-out_conv 2

   ./scripts/traintest-featcluster_minibatch_trial.sh 4 TFFF-out_conv 3
   ./scripts/traintest-featcluster_minibatch_trial.sh 4 TTFF-out_conv 3
   ./scripts/traintest-featcluster_minibatch_trial.sh 4 TTTF-out_conv 3

   ./scripts/traintest-featcluster_minibatch_trial.sh 4 TFFF-out_conv 4
   ./scripts/traintest-featcluster_minibatch_trial.sh 0 TTFF-out_conv 4
   ./scripts/traintest-featcluster_minibatch_trial.sh 1 TTTF-out_conv 4
   ```

17. Visualization for clusters in minibatch training.

   Extract cluster centers.
   ```bash
   python weight_extract.py --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/conv4featcluster-minibatch-TTTT-out_conv_trial2/max-va.pth --layer_str="blocks.0.feat_cluster.V_buffer" --vscode_debug

   python weight_extract.py --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/conv4featcluster-minibatch-TTTT-out_conv_trial2/max-va.pth --layer_str="blocks.1.feat_cluster.V_buffer" 

   python weight_extract.py --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/conv4featcluster-minibatch-TTTT-out_conv_trial2/max-va.pth --layer_str="blocks.2.feat_cluster.V_buffer" 

   python weight_extract.py --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/conv4featcluster-minibatch-TTTT-out_conv_trial2/max-va.pth --layer_str="blocks.3.feat_cluster.V_buffer" 
   ```

   Extract features.
   ```bash
   python feat_extract.py --gpu 0 --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/conv4featcluster-minibatch-TTTT-out_conv_trial2/max-va.pth --split train --sideout --feat_source="0.conv" --save_path=/home/shawn/weijian/data/conv4featcluster-minibatch-TTTT-out_conv_trial2

   python feat_extract.py --gpu 1 --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/conv4featcluster-minibatch-TTTT-out_conv_trial2/max-va.pth --split train --sideout --feat_source="1.conv" --save_path=/home/shawn/weijian/data/conv4featcluster-minibatch-TTTT-out_conv_trial2
   
   python feat_extract.py --gpu 2 --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/conv4featcluster-minibatch-TTTT-out_conv_trial2/max-va.pth --split train --sideout --feat_source="2.conv" --save_path=/home/shawn/weijian/data/conv4featcluster-minibatch-TTTT-out_conv_trial2
   
   python feat_extract.py --gpu 3 --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/conv4featcluster-minibatch-TTTT-out_conv_trial2/max-va.pth --split train --sideout --feat_source="3.conv" --save_path=/home/shawn/weijian/data/conv4featcluster-minibatch-TTTT-out_conv_trial2
   ```

   Visualize clusters.
   ```bash
   python calc_clusters.py --feat_path="/home/shawn/weijian/data/conv4featcluster-minibatch-TTTT-out_conv_trial2/train_0.conv_all_feat.pt" --feat_process="spatial" --save_name "load-clusters-0" --stage=-1 --kernel_size=1 --stride=1 --padding=0 --cluster_mode="load-clusters" --clusters_path="./save/conv4featcluster-minibatch-TTTT-out_conv_trial2/blocks.0.feat_cluster.V_buffer_weights.pt"

   python calc_clusters.py --feat_path="/home/shawn/weijian/data/conv4featcluster-minibatch-TTTT-out_conv_trial2/train_1.conv_all_feat.pt" --feat_process="spatial" --save_name "load-clusters-1" --stage=0 --kernel_size=1 --stride=1 --padding=0 --cluster_mode="load-clusters" --clusters_path="./save/conv4featcluster-minibatch-TTTT-out_conv_trial2/blocks.1.feat_cluster.V_buffer_weights.pt"

   python calc_clusters.py --feat_path="/home/shawn/weijian/data/conv4featcluster-minibatch-TTTT-out_conv_trial2/train_2.conv_all_feat.pt" --feat_process="spatial" --save_name "load-clusters-2" --stage=1 --kernel_size=1 --stride=1 --padding=0 --cluster_mode="load-clusters" --clusters_path="./save/conv4featcluster-minibatch-TTTT-out_conv_trial2/blocks.2.feat_cluster.V_buffer_weights.pt"
   
   python calc_clusters.py --feat_path="/home/shawn/weijian/data/conv4featcluster-minibatch-TTTT-out_conv_trial2/train_3.conv_all_feat.pt" --feat_process="spatial" --save_name "load-clusters-3" --stage=2 --kernel_size=1 --stride=1 --padding=0 --cluster_mode="load-clusters" --clusters_path="./save/conv4featcluster-minibatch-TTTT-out_conv_trial2/blocks.3.feat_cluster.V_buffer_weights.pt"
   ```

18. Test classification.
   ```bash
   python test_classifier.py --gpu 2 --config ./configs/current_configs/featcluster_minibatch/test_classifier_mini.yaml --load=./save/conv4featcluster-minibatch-TTTT_trial2/max-va.pth

   python test_classifier.py --gpu 3 --config ./configs/current_configs/featcluster_minibatch/test_classifier_mini.yaml --load=./save/conv4featcluster-minibatch-TTTT-out_conv_trial2/max-va.pth

   python test_classifier.py --gpu 4 --config ./configs/current_configs/featcluster_minibatch/test_classifier_mini.yaml --load=./save/conv4featcluster-minibatch-FFFF/max-va.pth
   
   ./scripts/test-featcluster_minibatch.sh 1 FFFF
   ./scripts/test-featcluster_minibatch_all_input_types.sh 2 FFFT
   ./scripts/test-featcluster_minibatch_all_input_types.sh 1 FFTT
   ./scripts/test-featcluster_minibatch_all_input_types.sh 2 FTTT
   ./scripts/test-featcluster_minibatch_all_input_types.sh 1 TTTT
   ./scripts/test-featcluster_minibatch_all_input_types.sh 2 TTTF
   ./scripts/test-featcluster_minibatch_all_input_types.sh 1 TTFF
   ./scripts/test-featcluster_minibatch_all_input_types.sh 2 TFFF
   ```

19. Some checkings for out_conv: out_conv_identity and out_conv_another3x3conv.

   Debug:
   ```bash
   python train_classifier.py --gpu 0 --config ./configs/current_configs/featcluster_minibatch/train_classifier_mini_conv4featcluster-minibatch-debug.yaml
   ```

   Train:
   ```bash
   ./scripts/traintest-featcluster_minibatch.sh 0 TFFF-out_conv_identity
   ./scripts/traintest-featcluster_minibatch.sh 1 TTFF-out_conv_identity
   ./scripts/traintest-featcluster_minibatch.sh 2 TTTF-out_conv_identity
   ./scripts/traintest-featcluster_minibatch.sh 3 TTTT-out_conv_identity
   ./scripts/traintest-featcluster_minibatch.sh 4 TFFF-out_conv_another3x3conv
   ./scripts/traintest-featcluster_minibatch.sh 5 TTFF-out_conv_another3x3conv
   ./scripts/traintest-featcluster_minibatch.sh 6 TTTF-out_conv_another3x3conv
   ./scripts/traintest-featcluster_minibatch.sh 7 TTTT-out_conv_another3x3conv
   ./scripts/traintest-featcluster_minibatch.sh 4 TFFF-out_conv_another1x1conv
   ./scripts/traintest-featcluster_minibatch.sh 5 TTFF-out_conv_another1x1conv
   ./scripts/traintest-featcluster_minibatch.sh 6 TTTF-out_conv_another1x1conv
   ./scripts/traintest-featcluster_minibatch.sh 7 TTTT-out_conv_another1x1conv
   ```

20. Res-12 featcluster minibatch experiments.
   Debug:
   ```bash
   python train_classifier.py --gpu 0 --config ./configs/current_configs/featcluster_minibatch_res12/train_classifier_mini_res12featcluster-minibatch-debug.yaml --vscode_debug
   ```

   Train and test:
   ```bash
   ./scripts/traintest-featcluster_minibatch_res12.sh 2 FFF-FFF-FFF-FFF
   ./scripts/traintest-featcluster_minibatch_res12.sh 3 TTT-TTT-TTT-TTT-out_conv
   ./scripts/traintest-featcluster_minibatch_res12.sh 4 TTT-TTT-TTT-FFF-out_conv
   ./scripts/traintest-featcluster_minibatch_res12.sh 5 TTT-TTT-FFF-FFF-out_conv
   ./scripts/traintest-featcluster_minibatch_res12.sh 6 TTT-FFF-FFF-FFF-out_conv
   ./scripts/traintest-featcluster_minibatch_res12.sh 7 TFF-TFF-TFF-TFF-out_conv
   ./scripts/traintest-featcluster_minibatch_res12.sh 0 TTT-FFF-FFF-FFF-input3x3  # Failed due to out of memory.
   ./scripts/traintest-featcluster_minibatch_res12.sh 0 TFF-FFF-FFF-FFF-input3x3
   ./scripts/traintest-featcluster_minibatch_res12.sh 1 TTT-FFF-FFF-FFF
   ./scripts/traintest-featcluster_minibatch_res12.sh 2 TFF-FFF-FFF-FFF-out_conv
   ./scripts/traintest-featcluster_minibatch_res12.sh 3 TFF-FFF-FFF-FFF
   
   ./scripts/traintest-featcluster_minibatch_res12_trial.sh 7 FFF-FFF-FFF-FFF 2-retrain
   ```

21. Test classification - Res-12.
   ```bash
   python test_classifier.py --gpu 2 --config ./configs/current_configs/featcluster_minibatch/test_classifier_mini.yaml --load=./save/res12featcluster-minibatch-FFF-FFF-FFF-FFF_trial2/max-va.pth
   
   ./scripts/test-featcluster_minibatch_res12.sh 2 FFF-FFF-FFF-FFF
   ./scripts/test-featcluster_minibatch_res12.sh 3 TTT-TTT-TTT-TTT-out_conv
   ./scripts/test-featcluster_minibatch_res12.sh 4 TTT-TTT-TTT-FFF-out_conv
   ./scripts/test-featcluster_minibatch_res12.sh 5 TTT-TTT-FFF-FFF-out_conv
   ./scripts/test-featcluster_minibatch_res12.sh 6 TTT-FFF-FFF-FFF-out_conv
   ./scripts/test-featcluster_minibatch_res12.sh 7 TFF-TFF-TFF-TFF-out_conv
   #./scripts/test-featcluster_minibatch_res12.sh 0 TTT-FFF-FFF-FFF-input3x3
   ./scripts/test-featcluster_minibatch_res12.sh 1 TFF-FFF-FFF-FFF-input3x3
   ./scripts/test-featcluster_minibatch_res12.sh 2 TTT-FFF-FFF-FFF
   ./scripts/test-featcluster_minibatch_res12.sh 3 TFF-FFF-FFF-FFF-out_conv
   ./scripts/test-featcluster_minibatch_res12.sh 4 TFF-FFF-FFF-FFF

   ```


22. Baseline improvement.
   - Fine-tune in meta-training way.
   ```bash
   python train_meta.py --gpu 0 --config configs/current_configs/finetune_meta_mini.yaml --load_encoder ./save/res12featcluster-minibatch-FFF-FFF-FFF-FFF_trial2/epoch-last.pth --name res12featcluster-minibatch-FFF-FFF-FFF-FFF_trial2_finetune-last-epoch --vscode_debug

   python train_meta.py --gpu 1 --config configs/current_configs/finetune_meta_mini.yaml --load_encoder ./save/res12featcluster-minibatch-FFF-FFF-FFF-FFF_trial2/max-va.pth --name res12featcluster-minibatch-FFF-FFF-FFF-FFF_trial2_finetune-max-va

   python train_meta.py --gpu 2 --config configs/current_configs/finetune_meta_mini.yaml --load_encoder ./save/res12featcluster-minibatch-TFF-FFF-FFF-FFF-out_conv_trial2/epoch-last.pth --name res12featcluster-minibatch-TFF-FFF-FFF-FFF-out_conv_trial2_finetune-last-epoch

   python train_meta.py --gpu 3 --config configs/current_configs/finetune_meta_mini.yaml --load_encoder ./save/res12featcluster-minibatch-TFF-FFF-FFF-FFF-out_conv_trial2/max-va.pth --name res12featcluster-minibatch-TFF-FFF-FFF-FFF-out_conv_trial2_finetune-max-va

   python train_meta.py --gpu 4 --config configs/current_configs/finetune_meta_mini.yaml --load_encoder ./save/res12featcluster-minibatch-TTT-FFF-FFF-FFF-out_conv_trial2/epoch-last.pth --name res12featcluster-minibatch-TTT-FFF-FFF-FFF-out_conv_trial2_finetune-last-epoch

   python train_meta.py --gpu 5 --config configs/current_configs/finetune_meta_mini.yaml --load_encoder ./save/res12featcluster-minibatch-TTT-FFF-FFF-FFF-out_conv_trial2/max-va.pth --name res12featcluster-minibatch-TTT-FFF-FFF-FFF-out_conv_trial2_finetune-max-va


   # Fix checkpoint and evaluate.
   python ./scripts/fix-checkpoint-config.py --src_ckpt_path=./save/res12featcluster-minibatch-FFF-FFF-FFF-FFF_trial2/epoch-last.pth --dest_ckpt_path=./save/res12featcluster-minibatch-FFF-FFF-FFF-FFF_trial2_finetune-last-epoch/max-va.pth

   python test_few_shot.py --shot 1 --gpu 0 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/res12featcluster-minibatch-FFF-FFF-FFF-FFF_trial2_finetune-last-epoch/max-va.pth
   # test epoch 10: acc=63.04 +- 0.23 (%), loss=0.9220  (seems last-epoch is better for baselines? needs validation!)

   # Fix checkpoint and evaluate.
   python ./scripts/fix-checkpoint-config.py \
   --src_ckpt_path=./save/res12featcluster-minibatch-FFF-FFF-FFF-FFF_trial2/max-va.pth \
   --dest_ckpt_path=./save/res12featcluster-minibatch-FFF-FFF-FFF-FFF_trial2_finetune-max-va/max-va.pth;

   python test_few_shot.py --shot 1 --gpu 1 --config ./configs/current_configs/test_few_shot_mini_general.yaml \
   --load_encoder=./save/res12featcluster-minibatch-FFF-FFF-FFF-FFF_trial2_finetune-max-va/max-va.pth
   # test epoch 10: acc=62.68 +- 0.23 (%), loss=0.9271

   # Fix checkpoint and evaluate.
   python ./scripts/fix-checkpoint-config.py \
   --src_ckpt_path=./save/res12featcluster-minibatch-TFF-FFF-FFF-FFF-out_conv_trial2/epoch-last.pth \
   --dest_ckpt_path=./save/res12featcluster-minibatch-TFF-FFF-FFF-FFF-out_conv_trial2_finetune-last-epoch/max-va.pth;

   python test_few_shot.py --shot 1 --gpu 2 --config ./configs/current_configs/test_few_shot_mini_general.yaml \
   --load_encoder=./save/res12featcluster-minibatch-TFF-FFF-FFF-FFF-out_conv_trial2_finetune-last-epoch/max-va.pth
   # test epoch 10: acc=62.82 +- 0.23 (%), loss=0.9262

   # Fix checkpoint and evaluate.
   python ./scripts/fix-checkpoint-config.py \
   --src_ckpt_path=./save/res12featcluster-minibatch-TFF-FFF-FFF-FFF-out_conv_trial2/max-va.pth \
   --dest_ckpt_path=./save/res12featcluster-minibatch-TFF-FFF-FFF-FFF-out_conv_trial2_finetune-max-va/max-va.pth;

   python test_few_shot.py --shot 1 --gpu 3 --config ./configs/current_configs/test_few_shot_mini_general.yaml \
   --load_encoder=./save/res12featcluster-minibatch-TFF-FFF-FFF-FFF-out_conv_trial2_finetune-max-va/max-va.pth
   # test epoch 10: acc=63.03 +- 0.23 (%), loss=0.9198
   
   # Fix checkpoint and evaluate.
   python ./scripts/fix-checkpoint-config.py \
   --src_ckpt_path=./save/res12featcluster-minibatch-TTT-FFF-FFF-FFF-out_conv_trial2/max-va.pth \
   --dest_ckpt_path=./save/res12featcluster-minibatch-TTT-FFF-FFF-FFF-out_conv_trial2_finetune-last-epoch/max-va.pth;

   python test_few_shot.py --shot 1 --gpu 4 --config ./configs/current_configs/test_few_shot_mini_general.yaml \
   --load_encoder=./save/res12featcluster-minibatch-TTT-FFF-FFF-FFF-out_conv_trial2_finetune-last-epoch/max-va.pth
   # test epoch 10: acc=62.55 +- 0.23 (%), loss=0.9312

   # Fix checkpoint and evaluate.
   python ./scripts/fix-checkpoint-config.py \
   --src_ckpt_path=./save/res12featcluster-minibatch-TTT-FFF-FFF-FFF-out_conv_trial2/epoch-last.pth \
   --dest_ckpt_path=./save/res12featcluster-minibatch-TTT-FFF-FFF-FFF-out_conv_trial2_finetune-max-va/max-va.pth;

   python test_few_shot.py --shot 1 --gpu 5 --config ./configs/current_configs/test_few_shot_mini_general.yaml \
   --load_encoder=./save/res12featcluster-minibatch-TTT-FFF-FFF-FFF-out_conv_trial2_finetune-max-va/max-va.pth
   # test epoch 10: acc=62.78 +- 0.23 (%), loss=0.9267
   
   # For debugging.
   python train_meta.py --gpu 0 --config configs/current_configs/finetune_meta_mini.yaml --load_encoder ./save/res12featcluster-minibatch-FFF-FFF-FFF-FFF_trial2/epoch-last.pth --name res12featcluster-minibatch-FFF-FFF-FFF-FFF_trial2_finetune-last-epoch_trial2 --vscode_debug --num_workers=0

   python test_few_shot.py --shot 1 --gpu 0 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/res12featcluster-minibatch-FFF-FFF-FFF-FFF_trial2_finetune-last-epoch_trial2/max-va.pth
   # test epoch 10: acc=62.57 +- 0.23 (%), loss=0.9287


   # Other experiments.
   ./scripts/finetune-featcluster_minibatch.sh 0 res12featcluster-minibatch-FFF-FFF-FFF-FFF_trial3
   ./scripts/finetune-featcluster_minibatch.sh 1 res12featcluster-minibatch-FFF-FFF-FFF-FFF_trial4
   ./scripts/finetune-featcluster_minibatch.sh 2 res12featcluster-minibatch-TFF-FFF-FFF-FFF-out_conv_trial3
   ./scripts/finetune-featcluster_minibatch.sh 3 res12featcluster-minibatch-TFF-FFF-FFF-FFF-out_conv_trial4
   ./scripts/finetune-featcluster_minibatch.sh 4 res12featcluster-minibatch-TTT-FFF-FFF-FFF-out_conv_trial3
   ./scripts/finetune-featcluster_minibatch.sh 5 res12featcluster-minibatch-TTT-FFF-FFF-FFF-out_conv_trial4
   ```


   - Use Lee's settings for fine-tune.
   ```bash
   python train_meta.py --gpu 6 --config configs/current_configs/finetune_meta_mini_lee-settings.yaml --load_encoder ./save/res12featcluster-minibatch-FFF-FFF-FFF-FFF_trial2/epoch-last.pth --name res12featcluster-minibatch-FFF-FFF-FFF-FFF_trial2_finetune_lee-settings-last-epoch

   python train_meta.py --gpu 7 --config configs/current_configs/finetune_meta_mini_lee-settings.yaml --load_encoder ./save/res12featcluster-minibatch-FFF-FFF-FFF-FFF_trial2/max-va.pth --name res12featcluster-minibatch-FFF-FFF-FFF-FFF_trial2_finetune_lee-settings-max-va

   python train_meta.py --gpu 6 --config configs/current_configs/finetune_meta_mini_lee-settings.yaml --load_encoder ./save/res12featcluster-minibatch-TFF-FFF-FFF-FFF-out_conv_trial2/epoch-last.pth --name res12featcluster-minibatch-TFF-FFF-FFF-FFF-out_conv_trial2_finetune_lee-settings-last-epoch

   python train_meta.py --gpu 7 --config configs/current_configs/finetune_meta_mini_lee-settings.yaml --load_encoder ./save/res12featcluster-minibatch-TFF-FFF-FFF-FFF-out_conv_trial2/max-va.pth --name res12featcluster-minibatch-TFF-FFF-FFF-FFF-out_conv_trial2_finetune_lee-settings-max-va

   # Fix checkpoint and evaluate.
   python ./scripts/fix-checkpoint-config.py \
   --src_ckpt_path=./save/res12featcluster-minibatch-FFF-FFF-FFF-FFF_trial2/epoch-last.pth \
   --dest_ckpt_path=./save/res12featcluster-minibatch-FFF-FFF-FFF-FFF_trial2_finetune_lee-settings-last-epoch/max-va.pth;

   python test_few_shot.py --shot 1 --gpu 6 --config ./configs/current_configs/test_few_shot_mini_general.yaml \
   --load_encoder=./save/res12featcluster-minibatch-FFF-FFF-FFF-FFF_trial2_finetune_lee-settings-last-epoch/max-va.pth
   # test epoch 10: acc=62.92 +- 0.23 (%), loss=0.9281

   # Fix checkpoint and evaluate.
   python ./scripts/fix-checkpoint-config.py \
   --src_ckpt_path=./save/res12featcluster-minibatch-FFF-FFF-FFF-FFF_trial2/max-va.pth \
   --dest_ckpt_path=./save/res12featcluster-minibatch-FFF-FFF-FFF-FFF_trial2_finetune_lee-settings-max-va/max-va.pth;

   python test_few_shot.py --shot 1 --gpu 7 --config ./configs/current_configs/test_few_shot_mini_general.yaml \
   --load_encoder=./save/res12featcluster-minibatch-FFF-FFF-FFF-FFF_trial2_finetune_lee-settings-max-va/max-va.pth
   # test epoch 10: acc=62.93 +- 0.23 (%), loss=0.9249

   # Fix checkpoint and evaluate.
   python ./scripts/fix-checkpoint-config.py \
   --src_ckpt_path=./save/res12featcluster-minibatch-TFF-FFF-FFF-FFF-out_conv_trial2/epoch-last.pth \
   --dest_ckpt_path=./save/res12featcluster-minibatch-TFF-FFF-FFF-FFF-out_conv_trial2_finetune_lee-settings-last-epoch/max-va.pth;

   python test_few_shot.py --shot 1 --gpu 6 --config ./configs/current_configs/test_few_shot_mini_general.yaml \
   --load_encoder=./save/res12featcluster-minibatch-TFF-FFF-FFF-FFF-out_conv_trial2_finetune_lee-settings-last-epoch/max-va.pth
   # test epoch 10: acc=62.65 +- 0.23 (%), loss=0.9327

   # Fix checkpoint and evaluate.
   python ./scripts/fix-checkpoint-config.py \
   --src_ckpt_path=./save/res12featcluster-minibatch-TFF-FFF-FFF-FFF-out_conv_trial2/max-va.pth \
   --dest_ckpt_path=./save/res12featcluster-minibatch-TFF-FFF-FFF-FFF-out_conv_trial2_finetune_lee-settings-max-va/max-va.pth;

   python test_few_shot.py --shot 1 --gpu 7 --config ./configs/current_configs/test_few_shot_mini_general.yaml \
   --load_encoder=./save/res12featcluster-minibatch-TFF-FFF-FFF-FFF-out_conv_trial2_finetune_lee-settings-max-va/max-va.pth
   # test epoch 10: acc=62.79 +- 0.23 (%), loss=0.9280
   ```


   - Train on more shots (still with Lee's settings).
   ```bash
   python train_meta.py --gpu 0 --config configs/current_configs/finetune_meta_mini_lee-settings-n_train_shot-5.yaml --load_encoder ./save/res12featcluster-minibatch-FFF-FFF-FFF-FFF_trial2/epoch-last.pth --name res12featcluster-minibatch-FFF-FFF-FFF-FFF_trial2_finetune_lee-settings-n_train_shot-5-last-epoch

   python train_meta.py --gpu 1 --config configs/current_configs/finetune_meta_mini_lee-settings-n_train_shot-5.yaml --load_encoder ./save/res12featcluster-minibatch-FFF-FFF-FFF-FFF_trial2/max-va.pth --name res12featcluster-minibatch-FFF-FFF-FFF-FFF_trial2_finetune_lee-settings-n_train_shot-5-max-va

   python train_meta.py --gpu 2 --config configs/current_configs/finetune_meta_mini_lee-settings-n_train_shot-5.yaml --load_encoder ./save/res12featcluster-minibatch-TFF-FFF-FFF-FFF-out_conv_trial2/epoch-last.pth --name res12featcluster-minibatch-TFF-FFF-FFF-FFF-out_conv_trial2_finetune_lee-settings-n_train_shot-5-last-epoch

   python train_meta.py --gpu 3 --config configs/current_configs/finetune_meta_mini_lee-settings-n_train_shot-5.yaml --load_encoder ./save/res12featcluster-minibatch-TFF-FFF-FFF-FFF-out_conv_trial2/max-va.pth --name res12featcluster-minibatch-TFF-FFF-FFF-FFF-out_conv_trial2_finetune_lee-settings-n_train_shot-5-max-va

   # python train_meta.py --gpu 0 --config configs/current_configs/finetune_meta_mini_lee-settings-n_train_shot-7.yaml --load_encoder ./save/res12featcluster-minibatch-FFF-FFF-FFF-FFF_trial2/epoch-last.pth --name res12featcluster-minibatch-FFF-FFF-FFF-FFF_trial2_finetune_lee-settings-n_train_shot-7-last-epoch

   # python train_meta.py --gpu 1 --config configs/current_configs/finetune_meta_mini_lee-settings-n_train_shot-7.yaml --load_encoder ./save/res12featcluster-minibatch-FFF-FFF-FFF-FFF_trial2/max-va.pth --name res12featcluster-minibatch-FFF-FFF-FFF-FFF_trial2_finetune_lee-settings-n_train_shot-7-max-va

   ./scripts/finetune-featcluster_minibatch_yaml.sh 0 finetune_meta_mini_lee-settings-n_train_shot-7.yaml res12featcluster-minibatch-FFF-FFF-FFF-FFF_trial2 res12featcluster-minibatch-FFF-FFF-FFF-FFF_trial2_finetune_lee-settings-n_train_shot-7

   python train_meta.py --gpu 4 --config configs/current_configs/finetune_meta_mini_lee-settings-n_train_shot-7.yaml --load_encoder ./save/res12featcluster-minibatch-TFF-FFF-FFF-FFF-out_conv_trial2/epoch-last.pth --name res12featcluster-minibatch-TFF-FFF-FFF-FFF-out_conv_trial2_finetune_lee-settings-n_train_shot-7-last-epoch

   python train_meta.py --gpu 5 --config configs/current_configs/finetune_meta_mini_lee-settings-n_train_shot-7.yaml --load_encoder ./save/res12featcluster-minibatch-TFF-FFF-FFF-FFF-out_conv_trial2/max-va.pth --name res12featcluster-minibatch-TFF-FFF-FFF-FFF-out_conv_trial2_finetune_lee-settings-n_train_shot-7-max-va


   # Fix checkpoint and evaluate.
   python ./scripts/fix-checkpoint-config.py \
   --src_ckpt_path=./save/res12featcluster-minibatch-FFF-FFF-FFF-FFF_trial2/epoch-last.pth \
   --dest_ckpt_path=./save/res12featcluster-minibatch-FFF-FFF-FFF-FFF_trial2_finetune_lee-settings-n_train_shot-5-last-epoch/max-va.pth;

   python test_few_shot.py --shot 1 --gpu 4 --config ./configs/current_configs/test_few_shot_mini_general.yaml \
   --load_encoder=./save/res12featcluster-minibatch-FFF-FFF-FFF-FFF_trial2_finetune_lee-settings-n_train_shot-5-last-epoch/max-va.pth
   # test epoch 10: acc=62.87 +- 0.23 (%), loss=0.9294

   # Fix checkpoint and evaluate.
   python ./scripts/fix-checkpoint-config.py \
   --src_ckpt_path=./save/res12featcluster-minibatch-FFF-FFF-FFF-FFF_trial2/max-va.pth \
   --dest_ckpt_path=./save/res12featcluster-minibatch-FFF-FFF-FFF-FFF_trial2_finetune_lee-settings-n_train_shot-5-max-va/max-va.pth;

   python test_few_shot.py --shot 1 --gpu 5 --config ./configs/current_configs/test_few_shot_mini_general.yaml \
   --load_encoder=./save/res12featcluster-minibatch-FFF-FFF-FFF-FFF_trial2_finetune_lee-settings-n_train_shot-5-max-va/max-va.pth
   # test epoch 10: acc=62.95 +- 0.23 (%), loss=0.9279


   # Fix checkpoint and evaluate.
   python ./scripts/fix-checkpoint-config.py \
   --src_ckpt_path=./save/res12featcluster-minibatch-TFF-FFF-FFF-FFF-out_conv_trial2/epoch-last.pth \
   --dest_ckpt_path=./save/res12featcluster-minibatch-TFF-FFF-FFF-FFF-out_conv_trial2_finetune_lee-settings-n_train_shot-5-last-epoch/max-va.pth;

   python test_few_shot.py --shot 1 --gpu 6 --config ./configs/current_configs/test_few_shot_mini_general.yaml \
   --load_encoder=./save/res12featcluster-minibatch-TFF-FFF-FFF-FFF-out_conv_trial2_finetune_lee-settings-n_train_shot-5-last-epoch/max-va.pth
   # test epoch 10: acc=62.92 +- 0.23 (%), loss=0.9319

   # Fix checkpoint and evaluate.
   python ./scripts/fix-checkpoint-config.py \
   --src_ckpt_path=./save/res12featcluster-minibatch-TFF-FFF-FFF-FFF-out_conv_trial2/max-va.pth \
   --dest_ckpt_path=./save/res12featcluster-minibatch-TFF-FFF-FFF-FFF-out_conv_trial2_finetune_lee-settings-n_train_shot-5-max-va/max-va.pth;

   python test_few_shot.py --shot 1 --gpu 7 --config ./configs/current_configs/test_few_shot_mini_general.yaml \
   --load_encoder=./save/res12featcluster-minibatch-TFF-FFF-FFF-FFF-out_conv_trial2_finetune_lee-settings-n_train_shot-5-max-va/max-va.pth
   # test epoch 10: acc=63.15 +- 0.23 (%), loss=0.9236

   # Fix checkpoint and evaluate.
   python ./scripts/fix-checkpoint-config.py \
   --src_ckpt_path=./save/res12featcluster-minibatch-TFF-FFF-FFF-FFF-out_conv_trial2/epoch-last.pth \
   --dest_ckpt_path=./save/res12featcluster-minibatch-TFF-FFF-FFF-FFF-out_conv_trial2_finetune_lee-settings-n_train_shot-7-last-epoch/max-va.pth;

   python test_few_shot.py --shot 1 --gpu 6 --config ./configs/current_configs/test_few_shot_mini_general.yaml \
   --load_encoder=./save/res12featcluster-minibatch-TFF-FFF-FFF-FFF-out_conv_trial2_finetune_lee-settings-n_train_shot-7-last-epoch/max-va.pth
   # test epoch 10: acc=63.15 +- 0.23 (%), loss=0.9259

   # Fix checkpoint and evaluate.
   python ./scripts/fix-checkpoint-config.py \
   --src_ckpt_path=./save/res12featcluster-minibatch-TFF-FFF-FFF-FFF-out_conv_trial2/max-va.pth \
   --dest_ckpt_path=./save/res12featcluster-minibatch-TFF-FFF-FFF-FFF-out_conv_trial2_finetune_lee-settings-n_train_shot-7-max-va/max-va.pth;

   python test_few_shot.py --shot 1 --gpu 7 --config ./configs/current_configs/test_few_shot_mini_general.yaml \
   --load_encoder=./save/res12featcluster-minibatch-TFF-FFF-FFF-FFF-out_conv_trial2_finetune_lee-settings-n_train_shot-7-max-va/max-va.pth
   # test epoch 10: acc=63.16 +- 0.23 (%), loss=0.9243
   ```


   - Train on more shots (still with Xiaolong's settings).
   Note: Actually, Xiaolong's settings are more fine-grained version of Lee's version.
   ```bash
   ./scripts/finetune-featcluster_minibatch_yaml.sh 1 finetune_meta_mini-n_train_shot-5.yaml res12featcluster-minibatch-FFF-FFF-FFF-FFF_trial2 res12featcluster-minibatch-FFF-FFF-FFF-FFF_trial2_finetune-n_train_shot-5;
   ./scripts/finetune-featcluster_minibatch_yaml.sh 1 finetune_meta_mini-n_train_shot-7.yaml res12featcluster-minibatch-FFF-FFF-FFF-FFF_trial2 res12featcluster-minibatch-FFF-FFF-FFF-FFF_trial2_finetune-n_train_shot-7

   ./scripts/finetune-featcluster_minibatch_yaml.sh 2 finetune_meta_mini-n_train_shot-5.yaml res12featcluster-minibatch-FFF-FFF-FFF-FFF_trial3 res12featcluster-minibatch-FFF-FFF-FFF-FFF_trial3_finetune-n_train_shot-5;
   ./scripts/finetune-featcluster_minibatch_yaml.sh 2 finetune_meta_mini-n_train_shot-7.yaml res12featcluster-minibatch-FFF-FFF-FFF-FFF_trial3 res12featcluster-minibatch-FFF-FFF-FFF-FFF_trial3_finetune-n_train_shot-7

   ./scripts/finetune-featcluster_minibatch_yaml.sh 3 finetune_meta_mini-n_train_shot-5.yaml res12featcluster-minibatch-FFF-FFF-FFF-FFF_trial4 res12featcluster-minibatch-FFF-FFF-FFF-FFF_trial4_finetune-n_train_shot-5;
   ./scripts/finetune-featcluster_minibatch_yaml.sh 3 finetune_meta_mini-n_train_shot-7.yaml res12featcluster-minibatch-FFF-FFF-FFF-FFF_trial4 res12featcluster-minibatch-FFF-FFF-FFF-FFF_trial4_finetune-n_train_shot-7

   ./scripts/finetune-featcluster_minibatch_yaml.sh 4 finetune_meta_mini-n_train_shot-5.yaml res12featcluster-minibatch-TFF-FFF-FFF-FFF-out_conv_trial2 res12featcluster-minibatch-TFF-FFF-FFF-FFF-out_conv_trial2_finetune-n_train_shot-5;
   ./scripts/finetune-featcluster_minibatch_yaml.sh 4 finetune_meta_mini-n_train_shot-7.yaml res12featcluster-minibatch-TFF-FFF-FFF-FFF-out_conv_trial2 res12featcluster-minibatch-TFF-FFF-FFF-FFF-out_conv_trial2_finetune-n_train_shot-7

   ./scripts/finetune-featcluster_minibatch_yaml.sh 5 finetune_meta_mini-n_train_shot-5.yaml res12featcluster-minibatch-TFF-FFF-FFF-FFF-out_conv_trial3 res12featcluster-minibatch-TFF-FFF-FFF-FFF-out_conv_trial3_finetune-n_train_shot-5;
   ./scripts/finetune-featcluster_minibatch_yaml.sh 5 finetune_meta_mini-n_train_shot-7.yaml res12featcluster-minibatch-TFF-FFF-FFF-FFF-out_conv_trial3 res12featcluster-minibatch-TFF-FFF-FFF-FFF-out_conv_trial3_finetune-n_train_shot-7

   ./scripts/finetune-featcluster_minibatch_yaml.sh 6 finetune_meta_mini-n_train_shot-5.yaml res12featcluster-minibatch-TFF-FFF-FFF-FFF-out_conv_trial4 res12featcluster-minibatch-TFF-FFF-FFF-FFF-out_conv_trial4_finetune-n_train_shot-5;
   ./scripts/finetune-featcluster_minibatch_yaml.sh 6 finetune_meta_mini-n_train_shot-7.yaml res12featcluster-minibatch-TFF-FFF-FFF-FFF-out_conv_trial4 res12featcluster-minibatch-TFF-FFF-FFF-FFF-out_conv_trial4_finetune-n_train_shot-7
   ```



   - Remove last activation function.
   ```bash
   python train_classifier.py --gpu 0 --config ./configs/current_configs/featcluster_minibatch_res12/FFF-FFF-FFF-FFF-no_last_act.yaml

   python train_classifier.py --gpu 1 --config ./configs/current_configs/featcluster_minibatch_res12/TFF-FFF-FFF-FFF-out_conv-no_last_act.yaml

   python train_classifier.py --gpu 2 --config ./configs/current_configs/featcluster_minibatch_res12/TTT-FFF-FFF-FFF-out_conv-no_last_act.yaml



   python test_few_shot.py --shot 1 --gpu 0 --config ./configs/current_configs/test_few_shot_mini_general.yaml \
   --load_encoder=./save/res12featcluster-minibatch-FFF-FFF-FFF-FFF-no_last_act/max-va.pth
   # test epoch 10: acc=60.97 +- 0.22 (%), loss=1.0145

   python test_few_shot.py --shot 1 --gpu 1 --config ./configs/current_configs/test_few_shot_mini_general.yaml \
   --load_encoder=./save/res12featcluster-minibatch-TFF-FFF-FFF-FFF-out_conv-no_last_act/max-va.pth
   # test epoch 10: acc=60.84 +- 0.22 (%), loss=1.0131

   python test_few_shot.py --shot 1 --gpu 2 --config ./configs/current_configs/test_few_shot_mini_general.yaml \
   --load_encoder=./save/res12featcluster-minibatch-TTT-FFF-FFF-FFF-out_conv-no_last_act/max-va.pth
   # test epoch 10: acc=61.04 +- 0.22 (%), loss=1.0057

   ```

   - More channels.
   ```bash
   python train_classifier.py --gpu 6 --config ./configs/current_configs/featcluster_minibatch_res12/FFF-FFF-FFF-FFF-morechannels.yaml

   python train_classifier.py --gpu 7 --config ./configs/current_configs/featcluster_minibatch_res12/TFF-FFF-FFF-FFF-morechannels-out_conv.yaml

   # python train_classifier.py --gpu 8 --config ./configs/current_configs/featcluster_minibatch_res12/TTT-FFF-FFF-FFF-morechannels-out_conv.yaml


   python test_few_shot.py --shot 1 --gpu 3 --config ./configs/current_configs/test_few_shot_mini_general.yaml \
   --load_encoder=./save/res12morechannelsfeatcluster-minibatch-FFF-FFF-FFF-FFF/max-va.pth
   # test epoch 10: acc=61.75 +- 0.23 (%), loss=0.9782

   python test_few_shot.py --shot 1 --gpu 4 --config ./configs/current_configs/test_few_shot_mini_general.yaml \
   --load_encoder=./save/res12morechannelsfeatcluster-minibatch-TFF-FFF-FFF-FFF-out_conv/max-va.pth
   # test epoch 10: acc=61.13 +- 0.22 (%), loss=1.0013
   ```

   - Directly train meta.
   ```bash
   python train_meta.py --gpu 0 --config ./configs/current_configs/featcluster_minibatch_res12/train_meta_mini_FFF-FFF-FFF-FFF.yaml --tag=trial2
   python test_few_shot.py --shot 1 --gpu 0 --config ./configs/current_configs/test_few_shot_mini_general.yaml \
   --load_encoder=./save/meta_res12featcluster-minibatch-FFF-FFF-FFF-FFF_trial2/max-va.pth

   python train_meta.py --gpu 1 --config ./configs/current_configs/featcluster_minibatch_res12/train_meta_mini_FFF-FFF-FFF-FFF.yaml --tag=trial3
   python test_few_shot.py --shot 1 --gpu 1 --config ./configs/current_configs/test_few_shot_mini_general.yaml \
   --load_encoder=./save/meta_res12featcluster-minibatch-FFF-FFF-FFF-FFF_trial3/max-va.pth
   
   python train_meta.py --gpu 2 --config ./configs/current_configs/featcluster_minibatch_res12/train_meta_mini_FFF-FFF-FFF-FFF.yaml --tag=trial4
   python test_few_shot.py --shot 1 --gpu 2 --config ./configs/current_configs/test_few_shot_mini_general.yaml \
   --load_encoder=./save/meta_res12featcluster-minibatch-FFF-FFF-FFF-FFF_trial4/max-va.pth


   python train_meta.py --gpu 7 --config ./configs/current_configs/featcluster_minibatch_res12/train_meta_mini_TFF-FFF-FFF-FFF-out_conv.yaml

   python train_meta.py --gpu 7 --config ./configs/current_configs/featcluster_minibatch_res12/train_meta_mini_TFF-FFF-FFF-FFF-out_conv.yaml --tag=trial2;
   python test_few_shot.py --shot 1 --gpu 3 --config ./configs/current_configs/test_few_shot_mini_general.yaml \
   --load_encoder=./save/meta_res12featcluster-minibatch-TFF-FFF-FFF-FFF-out_conv_trial2/max-va.pth

   python train_meta.py --gpu 7 --config ./configs/current_configs/featcluster_minibatch_res12/train_meta_mini_TFF-FFF-FFF-FFF-out_conv.yaml --tag=trial3;
   python test_few_shot.py --shot 1 --gpu 4 --config ./configs/current_configs/test_few_shot_mini_general.yaml \
   --load_encoder=./save/meta_res12featcluster-minibatch-TFF-FFF-FFF-FFF-out_conv_trial3/max-va.pth


   python train_meta.py --gpu 7 --config ./configs/current_configs/featcluster_minibatch_res12/train_meta_mini_TFF-FFF-FFF-FFF-out_conv.yaml --tag=trial4
   python test_few_shot.py --shot 1 --gpu 5 --config ./configs/current_configs/test_few_shot_mini_general.yaml \
   --load_encoder=./save/meta_res12featcluster-minibatch-TFF-FFF-FFF-FFF-out_conv_trial4/max-va.pth

   ```

   - No avgpool.
   Note: It would be more suitable for MetaOptNet since the SVM can absorb the linear transformation from avgpool.
   ```bash
   ./scripts/traintest-yaml.sh 3 FFF-FFF-FFF-FFF-no_avgpool.yaml res12featcluster-minibatch-FFF-FFF-FFF-FFF-no_avgpool
   # epoch 1, train 18048021.5457|0.0165, val 105882462.8503|0.0162, 1.2m 1.2m/2.0h
   ./scripts/traintest-yaml.sh 4 TFF-FFF-FFF-FFF-out_conv-no_avgpool.yaml res12featcluster-minibatch-TFF-FFF-FFF-FFF-out_conv-no_avgpool
   # epoch 1, train 47689.2208|0.0157, val 593385.3631|0.0119, 1.4m 1.4m/2.4h

   # May need warm-up or the init learning rate is too large.
   ```

   - Dropblock.
   ```bash
   ./scripts/traintest-yaml.sh 3 FFF-FFF-FFF-FFF-dropblock-rate0.1-size3.yaml res12featcluster-minibatch-FFF-FFF-FFF-FFF-dropblock-rate0.1-size3
   ./scripts/traintest-yaml.sh 4 FFF-FFF-FFF-FFF-dropblock-rate0.1-size5.yaml res12featcluster-minibatch-FFF-FFF-FFF-FFF-dropblock-rate0.1-size5
   ./scripts/traintest-yaml.sh 5 TFF-FFF-FFF-FFF-out_conv-dropblock-rate0.1-size3.yaml res12featcluster-minibatch-TFF-FFF-FFF-FFF-out_conv-dropblock-rate0.1-size3
   ./scripts/traintest-yaml.sh 6 TFF-FFF-FFF-FFF-out_conv-dropblock-rate0.1-size5.yaml res12featcluster-minibatch-TFF-FFF-FFF-FFF-out_conv-dropblock-rate0.1-size5

   # Debug:
   python train_classifier.py --gpu 3 --config ./configs/current_configs/featcluster_minibatch_res12/FFF-FFF-FFF-FFF-dropblock-rate0.1-size3.yaml --num_workers=0 --vscode_debug
   ```

23. Constellation visualization on ResNet. (Ref: 17. Visualization for clusters in minibatch training)

   Extract cluster centers.
   ```bash
   python weight_extract.py --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/res12featcluster-minibatch-TFF-FFF-FFF-FFF-out_conv_trial2/max-va.pth --layer_str="blocks.0.mergeblock1.feat_cluster.V_buffer"

   python weight_extract.py --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/res12featcluster-minibatch-TTT-FFF-FFF-FFF-out_conv_trial2/max-va.pth --layer_str="blocks.0.mergeblock1.feat_cluster.V_buffer"

   # TTT-TTT-TTT-TTT comparison.
   python weight_extract.py --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/res12featcluster-minibatch-TTT-TTT-TTT-TTT-out_conv_trial2/max-va.pth --layer_str="blocks.0.mergeblock1.feat_cluster.V_buffer"

   python weight_extract.py --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/res12featcluster-minibatch-TTT-TTT-TTT-TTT-out_conv_trial2/max-va.pth --layer_str="blocks.0.mergeblock2.feat_cluster.V_buffer"

   python weight_extract.py --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/res12featcluster-minibatch-TTT-TTT-TTT-TTT-out_conv_trial2/max-va.pth --layer_str="blocks.0.mergeblock3.feat_cluster.V_buffer"

   python weight_extract.py --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/res12featcluster-minibatch-TTT-TTT-TTT-TTT-out_conv_trial2/max-va.pth --layer_str="blocks.1.mergeblock3.feat_cluster.V_buffer"

   python weight_extract.py --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/res12featcluster-minibatch-TTT-TTT-TTT-TTT-out_conv_trial2/max-va.pth --layer_str="blocks.2.mergeblock3.feat_cluster.V_buffer"

   python weight_extract.py --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/res12featcluster-minibatch-TTT-TTT-TTT-TTT-out_conv_trial2/max-va.pth --layer_str="blocks.3.mergeblock3.feat_cluster.V_buffer"

   # More trials.
   python weight_extract.py --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/res12featcluster-minibatch-TTT-TTT-TTT-TTT-out_conv_trial3/max-va.pth --layer_str="blocks.3.mergeblock3.feat_cluster.V_buffer"

   python weight_extract.py --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/res12featcluster-minibatch-TTT-TTT-TTT-TTT-out_conv_trial4/max-va.pth --layer_str="blocks.3.mergeblock3.feat_cluster.V_buffer"
   ```

   Extract features.
   ```bash
   python feat_extract.py --gpu 0 --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/res12featcluster-minibatch-TFF-FFF-FFF-FFF-out_conv_trial2/max-va.pth --split train --sideout --feat_source="0.mergeblock1.conv" --save_path=/home/shawn/weijian/data/res12featcluster-minibatch-TFF-FFF-FFF-FFF-out_conv_trial2

   python feat_extract.py --gpu 1 --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/res12featcluster-minibatch-TTT-FFF-FFF-FFF-out_conv_trial2/max-va.pth --split train --sideout --feat_source="0.mergeblock1.conv" --save_path=/home/shawn/weijian/data/res12featcluster-minibatch-TTT-FFF-FFF-FFF-out_conv_trial2 --batch_size=32

   # TTT-TTT-TTT-TTT comparison.
   python feat_extract.py --gpu 0 --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/res12featcluster-minibatch-TTT-TTT-TTT-TTT-out_conv_trial2/max-va.pth --split train --sideout --feat_source="0.mergeblock1.conv" --save_path=/home/shawn/weijian/data/res12featcluster-minibatch-TTT-TTT-TTT-TTT-out_conv_trial2 --batch_size=32

   python feat_extract.py --gpu 0 --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/res12featcluster-minibatch-TTT-TTT-TTT-TTT-out_conv_trial2/max-va.pth --split train --sideout --feat_source="0.mergeblock2.conv" --save_path=/home/shawn/weijian/data/res12featcluster-minibatch-TTT-TTT-TTT-TTT-out_conv_trial2 --batch_size=32

   python feat_extract.py --gpu 0 --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/res12featcluster-minibatch-TTT-TTT-TTT-TTT-out_conv_trial2/max-va.pth --split train --sideout --feat_source="0.mergeblock3.conv" --save_path=/home/shawn/weijian/data/res12featcluster-minibatch-TTT-TTT-TTT-TTT-out_conv_trial2 --batch_size=32

   python feat_extract.py --gpu 1 --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/res12featcluster-minibatch-TTT-TTT-TTT-TTT-out_conv_trial2/max-va.pth --split train --sideout --feat_source="1.mergeblock3.conv" --save_path=/home/shawn/weijian/data/res12featcluster-minibatch-TTT-TTT-TTT-TTT-out_conv_trial2 --batch_size=32

   python feat_extract.py --gpu 2 --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/res12featcluster-minibatch-TTT-TTT-TTT-TTT-out_conv_trial2/max-va.pth --split train --sideout --feat_source="2.mergeblock3.conv" --save_path=/home/shawn/weijian/data/res12featcluster-minibatch-TTT-TTT-TTT-TTT-out_conv_trial2 --batch_size=32

   python feat_extract.py --gpu 3 --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/res12featcluster-minibatch-TTT-TTT-TTT-TTT-out_conv_trial2/max-va.pth --split train --sideout --feat_source="3.mergeblock3.conv" --save_path=/home/shawn/weijian/data/res12featcluster-minibatch-TTT-TTT-TTT-TTT-out_conv_trial2 --batch_size=32

   # More trials.
   python feat_extract.py --gpu 4 --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/res12featcluster-minibatch-TTT-TTT-TTT-TTT-out_conv_trial3/max-va.pth --split train --sideout --feat_source="3.mergeblock3.conv" --save_path=/home/shawn/weijian/data/res12featcluster-minibatch-TTT-TTT-TTT-TTT-out_conv_trial3 --batch_size=32

   python feat_extract.py --gpu 5 --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/res12featcluster-minibatch-TTT-TTT-TTT-TTT-out_conv_trial4/max-va.pth --split train --sideout --feat_source="3.mergeblock3.conv" --save_path=/home/shawn/weijian/data/res12featcluster-minibatch-TTT-TTT-TTT-TTT-out_conv_trial4 --batch_size=32
   ```

   Visualize clusters.
   ```bash
   python calc_clusters.py --feat_path="/home/shawn/weijian/data/res12featcluster-minibatch-TFF-FFF-FFF-FFF-out_conv_trial2/train_0.mergeblock1.conv_all_feat.pt" --feat_process="spatial" --save_name "load-clusters-0-subset0.01" --stage=-1 --kernel_size=1 --stride=1 --padding=0 --cluster_mode="load-clusters" --clusters_path="./save/res12featcluster-minibatch-TFF-FFF-FFF-FFF-out_conv_trial2/blocks.0.mergeblock1.feat_cluster.V_buffer_weights.pt" --subset_ratio=0.01

   python calc_clusters.py --feat_path="/home/shawn/weijian/data/res12featcluster-minibatch-TTT-FFF-FFF-FFF-out_conv_trial2/train_0.mergeblock1.conv_all_feat.pt" --feat_process="spatial" --save_name "load-clusters-0-subset0.01" --stage=-1 --kernel_size=1 --stride=1 --padding=0 --cluster_mode="load-clusters" --clusters_path="./save/res12featcluster-minibatch-TTT-FFF-FFF-FFF-out_conv_trial2/blocks.0.mergeblock1.feat_cluster.V_buffer_weights.pt" --subset_ratio=0.01

   # TTT-TTT-TTT-TTT comparison.
   python calc_clusters.py --feat_path="/home/shawn/weijian/data/res12featcluster-minibatch-TTT-TTT-TTT-TTT-out_conv_trial2/train_0.mergeblock1.conv_all_feat.pt" --feat_process="spatial" --save_name "load-clusters-resblock0-mergeblock1-subset0.01" --stage=-1 --kernel_size=1 --stride=1 --padding=0 --cluster_mode="load-clusters" --clusters_path="./save/res12featcluster-minibatch-TTT-TTT-TTT-TTT-out_conv_trial2/blocks.0.mergeblock1.feat_cluster.V_buffer_weights.pt" --subset_ratio=0.01

   python calc_clusters.py --feat_path="/home/shawn/weijian/data/res12featcluster-minibatch-TTT-TTT-TTT-TTT-out_conv_trial2/train_0.mergeblock2.conv_all_feat.pt" --feat_process="spatial" --save_name "load-clusters-resblock0-mergeblock2-subset0.01" --stage=-1 --kernel_size=1 --stride=1 --padding=0 --cluster_mode="load-clusters" --clusters_path="./save/res12featcluster-minibatch-TTT-TTT-TTT-TTT-out_conv_trial2/blocks.0.mergeblock2.feat_cluster.V_buffer_weights.pt" --subset_ratio=0.01

   python calc_clusters.py --feat_path="/home/shawn/weijian/data/res12featcluster-minibatch-TTT-TTT-TTT-TTT-out_conv_trial2/train_0.mergeblock3.conv_all_feat.pt" --feat_process="spatial" --save_name "load-clusters-resblock0-mergeblock3-subset0.01" --stage=-1 --kernel_size=1 --stride=1 --padding=0 --cluster_mode="load-clusters" --clusters_path="./save/res12featcluster-minibatch-TTT-TTT-TTT-TTT-out_conv_trial2/blocks.0.mergeblock3.feat_cluster.V_buffer_weights.pt" --subset_ratio=0.01

   python calc_clusters.py --feat_path="/home/shawn/weijian/data/res12featcluster-minibatch-TTT-TTT-TTT-TTT-out_conv_trial2/train_1.mergeblock3.conv_all_feat.pt" --feat_process="spatial" --save_name "load-clusters-resblock1-mergeblock3-subset0.01" --stage=0 --kernel_size=1 --stride=1 --padding=0 --cluster_mode="load-clusters" --clusters_path="./save/res12featcluster-minibatch-TTT-TTT-TTT-TTT-out_conv_trial2/blocks.1.mergeblock3.feat_cluster.V_buffer_weights.pt" --subset_ratio=0.01

   python calc_clusters.py --feat_path="/home/shawn/weijian/data/res12featcluster-minibatch-TTT-TTT-TTT-TTT-out_conv_trial2/train_2.mergeblock3.conv_all_feat.pt" --feat_process="spatial" --save_name "load-clusters-resblock2-mergeblock3-subset0.01" --stage=1 --kernel_size=1 --stride=1 --padding=0 --cluster_mode="load-clusters" --clusters_path="./save/res12featcluster-minibatch-TTT-TTT-TTT-TTT-out_conv_trial2/blocks.2.mergeblock3.feat_cluster.V_buffer_weights.pt" --subset_ratio=0.01

   python calc_clusters.py --feat_path="/home/shawn/weijian/data/res12featcluster-minibatch-TTT-TTT-TTT-TTT-out_conv_trial2/train_3.mergeblock3.conv_all_feat.pt" --feat_process="spatial" --save_name "load-clusters-resblock3-mergeblock3-subset0.01" --stage=2 --kernel_size=1 --stride=1 --padding=0 --cluster_mode="load-clusters" --clusters_path="./save/res12featcluster-minibatch-TTT-TTT-TTT-TTT-out_conv_trial2/blocks.3.mergeblock3.feat_cluster.V_buffer_weights.pt" --subset_ratio=0.01

   # Others.
   python calc_clusters.py --feat_path="/home/shawn/weijian/data/res12featcluster-minibatch-TTT-TTT-TTT-TTT-out_conv_trial2/train_3.mergeblock3.conv_all_feat.pt" --feat_process="spatial" --save_name "load-clusters-resblock3-mergeblock3" --stage=2 --kernel_size=1 --stride=1 --padding=0 --cluster_mode="load-clusters" --clusters_path="./save/res12featcluster-minibatch-TTT-TTT-TTT-TTT-out_conv_trial2/blocks.3.mergeblock3.feat_cluster.V_buffer_weights.pt"

   python calc_clusters.py --feat_path="/home/shawn/weijian/data/res12featcluster-minibatch-TTT-TTT-TTT-TTT-out_conv_trial2/train_3.mergeblock3.conv_all_feat.pt" --feat_process="spatial" --save_name "load-clusters-resblock3-mergeblock3-cos-shift" --stage=2 --kernel_size=1 --stride=1 --padding=0 --cluster_mode="load-clusters" --clusters_path="./save/res12featcluster-minibatch-TTT-TTT-TTT-TTT-out_conv_trial2/blocks.3.mergeblock3.feat_cluster.V_buffer_weights.pt"  --heatmap_mode='cos-shift'

   # More trials.
   python calc_clusters.py --feat_path="/home/shawn/weijian/data/res12featcluster-minibatch-TTT-TTT-TTT-TTT-out_conv_trial3/train_3.mergeblock3.conv_all_feat.pt" --feat_process="spatial" --save_name "load-clusters-resblock3-mergeblock3-subset0.01" --stage=2 --kernel_size=1 --stride=1 --padding=0 --cluster_mode="load-clusters" --clusters_path="./save/res12featcluster-minibatch-TTT-TTT-TTT-TTT-out_conv_trial3/blocks.3.mergeblock3.feat_cluster.V_buffer_weights.pt" --subset_ratio=0.01

   python calc_clusters.py --feat_path="/home/shawn/weijian/data/res12featcluster-minibatch-TTT-TTT-TTT-TTT-out_conv_trial4/train_3.mergeblock3.conv_all_feat.pt" --feat_process="spatial" --save_name "load-clusters-resblock3-mergeblock3-subset0.01" --stage=2 --kernel_size=1 --stride=1 --padding=0 --cluster_mode="load-clusters" --clusters_path="./save/res12featcluster-minibatch-TTT-TTT-TTT-TTT-out_conv_trial4/blocks.3.mergeblock3.feat_cluster.V_buffer_weights.pt" --subset_ratio=0.01
   ```

24. Fine-tuning experiments - Validation 1 (Check the effect of single clustering layer).
   ```bash
   ./scripts/traintest-featcluster_minibatch_res12.sh 0 FFT-FFF-FFF-FFF-out_conv
   ./scripts/traintest-featcluster_minibatch_res12.sh 1 FFF-FFT-FFF-FFF-out_conv
   ./scripts/traintest-featcluster_minibatch_res12.sh 2 FFF-FFF-FFT-FFF-out_conv
   ./scripts/traintest-featcluster_minibatch_res12.sh 3 FFF-FFF-FFF-FFT-out_conv
   ```

24. Fine-tuning experiments - Validation 2 (Check if sideout classifier can help).
   ```bash
   ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier.sh 2 FFF-FFF-FFF-FFF-out_conv
   ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier.sh 3 TTT-FFF-FFF-FFF-out_conv
   ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier.sh 4 FFF-TTT-FFF-FFF-out_conv
   ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier.sh 5 FFF-FFF-TTT-FFF-out_conv
   ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier.sh 6 FFF-FFF-FFF-TTT-out_conv
   ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier.sh 7 TTT-TTT-TTT-TTT-out_conv

   ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier.sh 0 FFF-FFF-TTT-FFF-dsn-conv
   ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier.sh 1 FFF-FFF-TTT-FFF-dsn-relu
   ```

   Debug:
   ```bash
   python train_classifier_sideout_classifier_temp.py --gpu 4 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier/TTT-FFF-FFF-FFF-out_conv.yaml --vscode_debug --num_workers=0

   python train_classifier_sideout_classifier_temp.py --gpu 0 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier/FFF-FFF-TTT-FFF-dsn-relu.yaml --vscode_debug --num_workers=0

   python test_few_shot.py --shot 1 --gpu 5 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load_encoder=./save/res12featcluster-minibatch-sideout-classifier-TTT-TTT-TTT-TTT-out_conv/max-va.pth
   ```

   Visualization:
   ```bash
   # FFF-FFF-TTT-FFF resblock2-mergeblock1.
   python weight_extract.py --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/res12featcluster-minibatch-sideout-classifier-FFF-FFF-TTT-FFF-out_conv_trial2/max-va.pth --layer_str="blocks.2.mergeblock1.feat_cluster.V_buffer"

   python feat_extract.py --gpu 2 --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/res12featcluster-minibatch-sideout-classifier-FFF-FFF-TTT-FFF-out_conv_trial2/max-va.pth --split train --sideout --feat_source="2.mergeblock1.conv" --save_path=/home/shawn/weijian/data/res12featcluster-minibatch-sideout-classifier-FFF-FFF-TTT-FFF-out_conv_trial2 --batch_size=32

   python calc_clusters.py --feat_path="/home/shawn/weijian/data/res12featcluster-minibatch-sideout-classifier-FFF-FFF-TTT-FFF-out_conv_trial2/train_2.mergeblock1.conv_all_feat.pt" --feat_process="spatial" --save_name "load-clusters-resblock2-mergeblock1-subset0.01" --stage=1 --kernel_size=1 --stride=1 --padding=0 --cluster_mode="load-clusters" --clusters_path="./save/res12featcluster-minibatch-sideout-classifier-FFF-FFF-TTT-FFF-out_conv_trial2/blocks.2.mergeblock1.feat_cluster.V_buffer_weights.pt" --subset_ratio=0.01


   # FFF-TTT-FFF-FFF resblock1-mergeblock1.
   python weight_extract.py --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/res12featcluster-minibatch-sideout-classifier-FFF-TTT-FFF-FFF-out_conv_trial2/max-va.pth --layer_str="blocks.1.mergeblock1.feat_cluster.V_buffer"

   python feat_extract.py --gpu 2 --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/res12featcluster-minibatch-sideout-classifier-FFF-TTT-FFF-FFF-out_conv_trial2/max-va.pth --split train --sideout --feat_source="1.mergeblock1.conv" --save_path=/home/shawn/weijian/data/res12featcluster-minibatch-sideout-classifier-FFF-TTT-FFF-FFF-out_conv_trial2 --batch_size=32

   python calc_clusters.py --feat_path="/home/shawn/weijian/data/res12featcluster-minibatch-sideout-classifier-FFF-TTT-FFF-FFF-out_conv_trial2/train_1.mergeblock1.conv_all_feat.pt" --feat_process="spatial" --save_name "load-clusters-resblock1-mergeblock1-subset0.01" --stage=0 --kernel_size=1 --stride=1 --padding=0 --cluster_mode="load-clusters" --clusters_path="./save/res12featcluster-minibatch-sideout-classifier-FFF-TTT-FFF-FFF-out_conv_trial2/blocks.1.mergeblock1.feat_cluster.V_buffer_weights.pt" --subset_ratio=0.01

   # TTT-TTT-TTT-TTT resblock2-mergeblock1.
   python weight_extract.py --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/res12featcluster-minibatch-sideout-classifier-TTT-TTT-TTT-TTT-out_conv_trial2/max-va.pth --layer_str="blocks.2.mergeblock1.feat_cluster.V_buffer"

   python feat_extract.py --gpu 2 --config ./configs/current_configs/train_feat_extract.yaml --load_encoder=./save/res12featcluster-minibatch-sideout-classifier-TTT-TTT-TTT-TTT-out_conv_trial2/max-va.pth --split train --sideout --feat_source="2.mergeblock1.conv" --save_path=/home/shawn/weijian/data/res12featcluster-minibatch-sideout-classifier-TTT-TTT-TTT-TTT-out_conv_trial2 --batch_size=32

   python calc_clusters.py --feat_path="/home/shawn/weijian/data/res12featcluster-minibatch-sideout-classifier-TTT-TTT-TTT-TTT-out_conv_trial2/train_2.mergeblock1.conv_all_feat.pt" --feat_process="spatial" --save_name "load-clusters-resblock2-mergeblock1-subset0.01" --stage=1 --kernel_size=1 --stride=1 --padding=0 --cluster_mode="load-clusters" --clusters_path="./save/res12featcluster-minibatch-sideout-classifier-TTT-TTT-TTT-TTT-out_conv_trial2/blocks.2.mergeblock1.feat_cluster.V_buffer_weights.pt" --subset_ratio=0.01
   ```

25. Fine-tuning experiments - Validation 2 extra (Check if sideout classifier and clustering concatenation can both help).
   ```bash
   ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier.sh 2 TFF-FFF-TTT-FFF-out_conv-firstTconcat
   ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier.sh 3 FFF-FFF-TTT-FFF-out_conv-withconcat
   ./scripts/traintest-featcluster_minibatch_res12_sideout_classifier.sh 4 TFF-FFF-FFF-FFF-out_conv-withconcat
   ```
   Debug:
   ```bash
   python train_classifier_sideout_classifier_temp.py --gpu 3 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier/TFF-FFF-TTT-FFF-out_conv-firstTconcat.yaml --vscode_debug --num_workers=0

   python train_classifier_sideout_classifier_temp.py --gpu 3 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier/FFF-FFF-TTT-FFF-out_conv-withconcat.yaml --vscode_debug --num_workers=0
   ```

26. Y-branch.
   Train:
   ```bash
   # moss103 0-6/2-4/5-7/8-9
   python train_classifier_sideout_classifier_ybranch.py --gpu 0,6 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_ybranch/FFF-FFF-FFF-FFF-out_conv-stage0.yaml --tag=trial2

   python train_classifier_sideout_classifier_ybranch.py --gpu 2,4 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_ybranch/FFF-FFF-FFF-FFF-out_conv-stage2.yaml --tag=trial2

   python train_classifier_sideout_classifier_ybranch.py --gpu 5,7 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_ybranch/FFF-FFF-FFF-FFF-out_conv-stage2.yaml --tag=trial3

   python train_classifier_sideout_classifier_ybranch.py --gpu 8,9 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_ybranch/FFF-FFF-FFF-FFF-out_conv-stage2.yaml --tag=trial4

   # moss101 1-3/2-4/5-7/8-9
   python train_classifier_sideout_classifier_ybranch.py --gpu 1,3 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_ybranch/FFF-FFF-FFF-FFF-out_conv-stage0.yaml --tag=trial3

   python train_classifier_sideout_classifier_ybranch.py --gpu 2,4 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_ybranch/FFF-FFF-TTT-FFF-out_conv-stage2.yaml --tag=trial2

   python train_classifier_sideout_classifier_ybranch.py --gpu 5,7 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_ybranch/FFF-FFF-TTT-FFF-out_conv-stage2.yaml --tag=trial3

   python train_classifier_sideout_classifier_ybranch.py --gpu 8,9 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_ybranch/FFF-FFF-TTT-FFF-out_conv-stage2.yaml --tag=trial4
  
   # moss102 2-4/5-7/8-9
   python train_classifier_sideout_classifier_ybranch.py --gpu 0,1 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_ybranch/FFF-FFF-FFF-FFF-out_conv-stage0.yaml --tag=trial4

   python train_classifier_sideout_classifier_ybranch.py --gpu 2,4 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_ybranch/TFF-FFF-TTT-FFF-out_conv-stage2-firstTconcat.yaml --tag=trial2

   python train_classifier_sideout_classifier_ybranch.py --gpu 5,7 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_ybranch/TFF-FFF-TTT-FFF-out_conv-stage2-firstTconcat.yaml --tag=trial3
   
   python train_classifier_sideout_classifier_ybranch.py --gpu 8,9 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_ybranch/TFF-FFF-TTT-FFF-out_conv-stage2-firstTconcat.yaml --tag=trial4

   # moss105 0-1/4-5/6-7
   python train_classifier_sideout_classifier_ybranch.py --gpu 2,3 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_ybranch/TFF-FFF-TTT-FFF-out_conv-stage2-firstTconcat.yaml --tag=trial5

   python train_classifier_sideout_classifier_ybranch.py --gpu 0,1 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_ybranch/TTT-FFF-TTT-FFF-out_conv-stage2-firstTTTconcat.yaml --tag=trial2

   python train_classifier_sideout_classifier_ybranch.py --gpu 4,5 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_ybranch/TTT-FFF-TTT-FFF-out_conv-stage2-firstTTTconcat.yaml --tag=trial3

   python train_classifier_sideout_classifier_ybranch.py --gpu 6,7 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_ybranch/TTT-FFF-TTT-FFF-out_conv-stage2-firstTTTconcat.yaml --tag=trial4






   # moss105 0-1/4-5/6-7
   python train_classifier_sideout_classifier_ybranch.py --gpu 0,1 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_ybranch/TFF-FFF-TTT-FFF-out_conv-stage2-firstTconcat-train7shot.yaml --tag=trial2

   python train_classifier_sideout_classifier_ybranch.py --gpu 4,5 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_ybranch/TFF-FFF-TTT-FFF-out_conv-stage2-firstTconcat-train7shot.yaml --tag=trial3

   python train_classifier_sideout_classifier_ybranch.py --gpu 6,7 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_ybranch/TFF-FFF-TTT-FFF-out_conv-stage2-firstTconcat-train7shot.yaml --tag=trial4
   
   python train_classifier_sideout_classifier_ybranch.py --gpu 2,3 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_ybranch/TFF-FFF-TTT-FFF-out_conv-stage2-firstTconcat-train3shot-60epoch.yaml --tag=trial2;

   python train_classifier_sideout_classifier_ybranch.py --gpu 2,3 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_ybranch/TFF-FFF-TTT-FFF-out_conv-stage2-firstTconcat-train3shot-60epoch.yaml --tag=trial3;

   python train_classifier_sideout_classifier_ybranch.py --gpu 2,3 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_ybranch/TFF-FFF-TTT-FFF-out_conv-stage2-firstTconcat-train3shot-60epoch.yaml --tag=trial4

   # moss106
   python train_classifier_sideout_classifier_ybranch.py --gpu 0 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_ybranch/FFF-FFF-FFF-FFF-out_conv-stage2-onlybranch1.yaml --tag=trial2

   python train_classifier_sideout_classifier_ybranch.py --gpu 1 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_ybranch/FFF-FFF-FFF-FFF-out_conv-stage2-onlybranch1.yaml --tag=trial3

   python train_classifier_sideout_classifier_ybranch.py --gpu 2 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_ybranch/FFF-FFF-FFF-FFF-out_conv-stage2-onlybranch1.yaml --tag=trial4

   python train_classifier_sideout_classifier_ybranch.py --gpu 3 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_ybranch/FFF-FFF-FFF-FFF-out_conv-stage2-onlybranch2.yaml --tag=trial2

   python train_classifier_sideout_classifier_ybranch.py --gpu 4 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_ybranch/FFF-FFF-FFF-FFF-out_conv-stage2-onlybranch2.yaml --tag=trial3

   python train_classifier_sideout_classifier_ybranch.py --gpu 6 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_ybranch/FFF-FFF-FFF-FFF-out_conv-stage2-onlybranch2.yaml --tag=trial4




   # moss106
   python train_classifier_sideout_classifier_ybranch.py --gpu 1 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_ybranch/FFF-FFF-FFF-FFF-out_conv-stage2-onlybranch1-train3shot.yaml --tag=trial2

   python train_classifier_sideout_classifier_ybranch.py --gpu 2 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_ybranch/FFF-FFF-FFF-FFF-out_conv-stage2-onlybranch1-train3shot.yaml --tag=trial3

   python train_classifier_sideout_classifier_ybranch.py --gpu 3 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_ybranch/FFF-FFF-FFF-FFF-out_conv-stage2-onlybranch1-train3shot.yaml --tag=trial4

   # moss105
   python train_classifier_sideout_classifier_ybranch.py --gpu 0 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_ybranch/FFF-FFF-FFF-FFF-out_conv-stage2-onlybranch2-train3shot.yaml --tag=trial2

   python train_classifier_sideout_classifier_ybranch.py --gpu 1 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_ybranch/FFF-FFF-FFF-FFF-out_conv-stage2-onlybranch2-train3shot.yaml --tag=trial3

   python train_classifier_sideout_classifier_ybranch.py --gpu 4 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_ybranch/FFF-FFF-FFF-FFF-out_conv-stage2-onlybranch2-train3shot.yaml --tag=trial4

   python train_classifier_sideout_classifier_ybranch.py --gpu 5 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_ybranch/FFF-FFF-FFF-FFF-out_conv-stage0-train3shot-60epoch.yaml --tag=trial2-1gpu

   python train_classifier_sideout_classifier_ybranch.py --gpu 6,7 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_ybranch/TFF-FFF-TTT-FFF-out_conv-stage2-firstTconcat-train3shot-60epoch-dropblock-rate0.1-size5.yaml --tag=trial2
   

   # moss101  12  5-7/8-9
   python train_classifier_sideout_classifier_ybranch.py --gpu 1,2 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_ybranch/FFF-FFF-FFF-FFF-out_conv-stage0-train3shot-60epoch.yaml --tag=trial2
   
   python train_classifier_sideout_classifier_ybranch.py --gpu 5,7 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_ybranch/FFF-FFF-FFF-FFF-out_conv-stage0-train3shot-60epoch.yaml --tag=trial3
   
   python train_classifier_sideout_classifier_ybranch.py --gpu 8,9 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_ybranch/FFF-FFF-FFF-FFF-out_conv-stage0-train3shot-60epoch.yaml --tag=trial4

   # moss106
   python train_classifier_sideout_classifier_ybranch.py --gpu 4,5,6,7 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_ybranch/FFF-FFF-FFF-FFF-out_conv-stage0-train3shot-60epoch.yaml --tag=trial2-4gpu


   # moss102
   python train_classifier_sideout_classifier_ybranch.py --gpu 1,3 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_ybranch/FFF-FFF-FFF-FFF-out_conv-stage2-cos-temp10.0.yaml --tag=trial2

   # moss103
   python train_classifier_sideout_classifier_ybranch.py --gpu 1,3 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_ybranch/FFF-FFF-FFF-FFF-out_conv-stage2-cos-temp10.0.yaml --tag=trial3
   

   ```
   
   Evaluation:
   ```bash
   # moss103 GPU0 has issue.

   # moss101
   ./scripts/test-featcluster_minibatch_res12_sideout_classifier_ybranch.sh 1 FFF-FFF-FFF-FFF-out_conv-stage2_trial2
   ./scripts/test-featcluster_minibatch_res12_sideout_classifier_ybranch.sh 2 FFF-FFF-FFF-FFF-out_conv-stage2_trial3
   ./scripts/test-featcluster_minibatch_res12_sideout_classifier_ybranch.sh 3 FFF-FFF-FFF-FFF-out_conv-stage2_trial4
   ./scripts/test-featcluster_minibatch_res12_sideout_classifier_ybranch.sh 4 FFF-FFF-FFF-FFF-out_conv-stage0_trial2
   ./scripts/test-featcluster_minibatch_res12_sideout_classifier_ybranch.sh 5 FFF-FFF-FFF-FFF-out_conv-stage0_trial3
   ./scripts/test-featcluster_minibatch_res12_sideout_classifier_ybranch.sh 6 FFF-FFF-FFF-FFF-out_conv-stage0_trial4

   # moss102
   ./scripts/test-featcluster_minibatch_res12_sideout_classifier_ybranch.sh 0 FFF-FFF-TTT-FFF-out_conv-stage2_trial2
   ./scripts/test-featcluster_minibatch_res12_sideout_classifier_ybranch.sh 1 FFF-FFF-TTT-FFF-out_conv-stage2_trial3
   ./scripts/test-featcluster_minibatch_res12_sideout_classifier_ybranch.sh 2 FFF-FFF-TTT-FFF-out_conv-stage2_trial4

   ./scripts/test-featcluster_minibatch_res12_sideout_classifier_ybranch.sh 3 TFF-FFF-TTT-FFF-out_conv-stage2-firstTconcat_trial2
   ./scripts/test-featcluster_minibatch_res12_sideout_classifier_ybranch.sh 4 TFF-FFF-TTT-FFF-out_conv-stage2-firstTconcat_trial3
   ./scripts/test-featcluster_minibatch_res12_sideout_classifier_ybranch.sh 5 TFF-FFF-TTT-FFF-out_conv-stage2-firstTconcat_trial4

   ./scripts/test-featcluster_minibatch_res12_sideout_classifier_ybranch.sh 6 TTT-FFF-TTT-FFF-out_conv-stage2-firstTTTconcat_trial2
   ./scripts/test-featcluster_minibatch_res12_sideout_classifier_ybranch.sh 7 TTT-FFF-TTT-FFF-out_conv-stage2-firstTTTconcat_trial3
   ./scripts/test-featcluster_minibatch_res12_sideout_classifier_ybranch.sh 8 TTT-FFF-TTT-FFF-out_conv-stage2-firstTTTconcat_trial4


   # moss106
   ./scripts/test-featcluster_minibatch_res12_sideout_classifier_ybranch.sh 1 TFF-FFF-TTT-FFF-out_conv-stage2-firstTconcat-train7shot_trial2
   ./scripts/test-featcluster_minibatch_res12_sideout_classifier_ybranch.sh 2 TFF-FFF-TTT-FFF-out_conv-stage2-firstTconcat-train7shot_trial3
   ./scripts/test-featcluster_minibatch_res12_sideout_classifier_ybranch.sh 3 TFF-FFF-TTT-FFF-out_conv-stage2-firstTconcat-train7shot_trial4

   # moss101 1-3/2-4/5-7/8-9
   ./scripts/test-featcluster_minibatch_res12_sideout_classifier_ybranch.sh 4 FFF-FFF-FFF-FFF-out_conv-stage0-train3shot-60epoch_trial2
   ./scripts/test-featcluster_minibatch_res12_sideout_classifier_ybranch.sh 2 FFF-FFF-FFF-FFF-out_conv-stage0-train3shot-60epoch_trial3
   ./scripts/test-featcluster_minibatch_res12_sideout_classifier_ybranch.sh 5 FFF-FFF-FFF-FFF-out_conv-stage0-train3shot-60epoch_trial4
   ./scripts/test-featcluster_minibatch_res12_sideout_classifier_ybranch.sh 7 FFF-FFF-FFF-FFF-out_conv-stage0-train3shot-60epoch_trial2-1gpu
   ./scripts/test-featcluster_minibatch_res12_sideout_classifier_ybranch.sh 8 FFF-FFF-FFF-FFF-out_conv-stage0-train3shot-60epoch_trial2-4gpu
   ./scripts/test-featcluster_minibatch_res12_sideout_classifier_ybranch.sh 9 FFF-FFF-FFF-FFF-out_conv-stage2-cos-temp10.0_trial3

   # moss105
   ./scripts/test-featcluster_minibatch_res12_sideout_classifier_ybranch.sh 6 TFF-FFF-TTT-FFF-out_conv-stage2-firstTconcat-train3shot-60epoch-dropblock-rate0.1-size5_trial2
   ./scripts/test-featcluster_minibatch_res12_sideout_classifier_ybranch.sh 7 FFF-FFF-FFF-FFF-out_conv-stage2-cos-temp10.0_trial2

   # moss105
   ./scripts/test-featcluster_minibatch_res12_sideout_classifier_ybranch.sh 4 TFF-FFF-TTT-FFF-out_conv-stage2-firstTconcat-train3shot-60epoch_trial2
   ./scripts/test-featcluster_minibatch_res12_sideout_classifier_ybranch.sh 5 TFF-FFF-TTT-FFF-out_conv-stage2-firstTconcat-train3shot-60epoch_trial3
   ./scripts/test-featcluster_minibatch_res12_sideout_classifier_ybranch.sh 6 TFF-FFF-TTT-FFF-out_conv-stage2-firstTconcat-train3shot-60epoch_trial4
  


   # Branch2 only - 1-shot training.
   python test_few_shot_ybranch.py --shot 1 --gpu 4 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load=./save/res12featcluster-minibatch-sideout-classifier-ybranch-FFF-FFF-FFF-FFF-out_conv-stage2-onlybranch2_trial2/max-f-va.pth --method='sqr' --dist_func_list='none' --logits_coeff_list='20' --disable_out_feat_flatten --sideout --feat_source_list='final' --branch_list='2'
   # 59.24%

   python test_few_shot_ybranch.py --shot 1 --gpu 4 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load=./save/res12featcluster-minibatch-sideout-classifier-ybranch-FFF-FFF-FFF-FFF-out_conv-stage2-onlybranch2_trial2/max-f-va.pth --method='cos' --dist_func_list='none' --logits_coeff_list='20' --disable_out_feat_flatten --sideout --feat_source_list='final' --branch_list='2'
   # 60.12%

   python test_few_shot_ybranch.py --shot 1 --gpu 5 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load=./save/res12featcluster-minibatch-sideout-classifier-ybranch-FFF-FFF-FFF-FFF-out_conv-stage2-onlybranch2_trial3/max-f-va.pth --method='cos' --dist_func_list='none' --logits_coeff_list='20' --disable_out_feat_flatten --sideout --feat_source_list='final' --branch_list='2'
   # 60.29%
   
   python test_few_shot_ybranch.py --shot 1 --gpu 6 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load=./save/res12featcluster-minibatch-sideout-classifier-ybranch-FFF-FFF-FFF-FFF-out_conv-stage2-onlybranch2_trial4/max-f-va.pth --method='cos' --dist_func_list='none' --logits_coeff_list='20' --disable_out_feat_flatten --sideout --feat_source_list='final' --branch_list='2'
   # 60.31%





   # Branch2 only - 3-shot training.
   python test_few_shot_ybranch.py --shot 1 --gpu 1 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load=./save/res12featcluster-minibatch-sideout-classifier-ybranch-FFF-FFF-FFF-FFF-out_conv-stage2-onlybranch2-train3shot_trial2/max-f-va.pth --method='cos' --dist_func_list='none' --logits_coeff_list='20' --disable_out_feat_flatten --sideout --feat_source_list='final' --branch_list='2'
   # 61.12%

   python test_few_shot_ybranch.py --shot 1 --gpu 2 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load=./save/res12featcluster-minibatch-sideout-classifier-ybranch-FFF-FFF-FFF-FFF-out_conv-stage2-onlybranch2-train3shot_trial3/max-f-va.pth --method='cos' --dist_func_list='none' --logits_coeff_list='20' --disable_out_feat_flatten --sideout --feat_source_list='final' --branch_list='2'
   # 60.86%
   
   python test_few_shot_ybranch.py --shot 1 --gpu 3 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load=./save/res12featcluster-minibatch-sideout-classifier-ybranch-FFF-FFF-FFF-FFF-out_conv-stage2-onlybranch2-train3shot_trial4/max-f-va.pth --method='cos' --dist_func_list='none' --logits_coeff_list='20' --disable_out_feat_flatten --sideout --feat_source_list='final' --branch_list='2'
   # 60.93%











   python test_few_shot_ybranch.py --shot 1 --gpu 7 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load=./save/res12featcluster-minibatch-sideout-classifier-ybranch-FFF-FFF-FFF-FFF-out_conv-stage2-onlybranch1_trial2/epoch-25.pth --method='cos' --dist_func_list='none' --logits_coeff_list='20' --disable_out_feat_flatten --sideout --feat_source_list='final' --branch_list='1'
   # 60.35%

   python test_few_shot_ybranch.py --shot 1 --gpu 5 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load=./save/res12featcluster-minibatch-sideout-classifier-ybranch-FFF-FFF-FFF-FFF-out_conv-stage2-onlybranch1_trial2/epoch-30.pth --method='cos' --dist_func_list='none' --logits_coeff_list='20' --disable_out_feat_flatten --sideout --feat_source_list='final' --branch_list='1'
   # 59.20%

   python test_few_shot_ybranch.py --shot 1 --gpu 6 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load=./save/res12featcluster-minibatch-sideout-classifier-ybranch-FFF-FFF-FFF-FFF-out_conv-stage2-onlybranch1_trial2/epoch-40.pth --method='cos' --dist_func_list='none' --logits_coeff_list='20' --disable_out_feat_flatten --sideout --feat_source_list='final' --branch_list='1'
   # 59.25%

   python test_few_shot_ybranch.py --shot 1 --gpu 5 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load=./save/res12featcluster-minibatch-sideout-classifier-ybranch-FFF-FFF-FFF-FFF-out_conv-stage2-onlybranch1_trial2/epoch-45.pth --method='cos' --dist_func_list='none' --logits_coeff_list='20' --disable_out_feat_flatten --sideout --feat_source_list='final' --branch_list='1'
   # 58.56%

   python test_few_shot_ybranch.py --shot 1 --gpu 6 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load=./save/res12featcluster-minibatch-sideout-classifier-ybranch-FFF-FFF-FFF-FFF-out_conv-stage2-onlybranch1_trial2/epoch-55.pth --method='cos' --dist_func_list='none' --logits_coeff_list='20' --disable_out_feat_flatten --sideout --feat_source_list='final' --branch_list='1'
   # 58.64%

   python test_few_shot_ybranch.py --shot 1 --gpu 7 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load=./save/res12featcluster-minibatch-sideout-classifier-ybranch-FFF-FFF-FFF-FFF-out_conv-stage2-onlybranch1_trial2/epoch-80.pth --method='cos' --dist_func_list='none' --logits_coeff_list='20' --disable_out_feat_flatten --sideout --feat_source_list='final' --branch_list='1'
   # 58.83%




   python test_few_shot_ybranch.py --shot 1 --gpu 7 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load=./save/res12featcluster-minibatch-sideout-classifier-ybranch-FFF-FFF-FFF-FFF-out_conv-stage0_trial2/epoch-25.pth --method='cos' --dist_func_list='none' --logits_coeff_list='20' --disable_out_feat_flatten --sideout --feat_source_list='final' --branch_list='1'
   # 59.75%

   python test_few_shot_ybranch.py --shot 1 --gpu 6 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load=./save/res12featcluster-minibatch-sideout-classifier-ybranch-FFF-FFF-FFF-FFF-out_conv-stage0_trial2/max-f-va.pth --method='cos' --dist_func_list='none' --logits_coeff_list='20' --disable_out_feat_flatten --sideout --feat_source_list='final' --branch_list='1'
   # 60.48%


   python test_few_shot_ybranch.py --shot 1 --gpu 0 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load=./save/res12featcluster-minibatch-sideout-classifier-ybranch-FFF-FFF-FFF-FFF-out_conv-stage0_trial2/epoch-45.pth --method='cos' --dist_func_list='none' --logits_coeff_list='20' --disable_out_feat_flatten --sideout --feat_source_list='final' --branch_list='1'
   # 58.54%

   python test_few_shot_ybranch.py --shot 1 --gpu 4 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load=./save/res12featcluster-minibatch-sideout-classifier-ybranch-FFF-FFF-FFF-FFF-out_conv-stage0_trial2/epoch-55.pth --method='cos' --dist_func_list='none' --logits_coeff_list='20' --disable_out_feat_flatten --sideout --feat_source_list='final' --branch_list='1'
   # 58.49%


   python test_few_shot_ybranch.py --shot 1 --gpu 5 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load=./save/res12featcluster-minibatch-sideout-classifier-ybranch-FFF-FFF-FFF-FFF-out_conv-stage0_trial2/epoch-80.pth --method='cos' --dist_func_list='none' --logits_coeff_list='20' --disable_out_feat_flatten --sideout --feat_source_list='final' --branch_list='1'
   # 58.26%











   # Debug.
   python test_few_shot_ybranch.py --shot 5 --gpu 4 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load=./save/res12featcluster-minibatch-sideout-classifier-ybranch-TFF-FFF-TTT-FFF-out_conv-stage2-firstTconcat_trial4/max-f-va.pth --method='cos,cos' --dist_func_list='none,none' --logits_coeff_list='20,20' --sideout --feat_source_list='final,final' --branch_list='1,2'

   python test_few_shot_ybranch.py --shot 5 --gpu 3 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load=./save/res12featcluster-minibatch-sideout-classifier-ybranch-TFF-FFF-TTT-FFF-out_conv-stage2-firstTconcat_trial4/max-f-va.pth --method='cos,cos' --dist_func_list='none,none' --logits_coeff_list='20,20' --disable_out_feat_flatten --sideout --feat_source_list='final,final' --branch_list='1,2'

   python test_few_shot_ybranch.py --shot 5 --gpu 2 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load=./save/res12featcluster-minibatch-sideout-classifier-ybranch-FFF-FFF-FFF-FFF-out_conv-stage2_trial4/max-f-va.pth --method='cos' --dist_func_list='none' --logits_coeff_list='20' --disable_out_feat_flatten --sideout --feat_source_list='final' --branch_list='1'

   python test_few_shot_ybranch.py --shot 1 --gpu 3 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load=./save/res12featcluster-minibatch-sideout-classifier-ybranch-FFF-FFF-FFF-FFF-out_conv-stage2_trial2/max-f-va.pth --method='cos' --dist_func_list='none' --logits_coeff_list='20' --disable_out_feat_flatten --sideout --feat_source_list='final' --branch_list='1'

   python test_few_shot_ybranch.py --shot 1 --gpu 2 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load=./save/res12featcluster-minibatch-sideout-classifier-ybranch-FFF-FFF-FFF-FFF-out_conv-stage2_trial2/max-f-va.pth --method='cos' --dist_func_list='none' --logits_coeff_list='20' --disable_out_feat_flatten --sideout --feat_source_list='final' --branch_list='2'

   python test_few_shot_ybranch.py --shot 1 --gpu 2 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load=./save/res12featcluster-minibatch-sideout-classifier-ybranch-FFF-FFF-FFF-FFF-out_conv-stage2_trial2/max-f-va.pth --method='cos,cos' --dist_func_list='none,none' --logits_coeff_list='20,20' --disable_out_feat_flatten --sideout --feat_source_list='final,final' --branch_list='1,2' --num_workers=0 --vscode_debug


   ```

   Debug:
   ```bash
   python train_classifier_sideout_classifier_ybranch.py --gpu 0 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_ybranch/FFF-FFF-FFF-FFF-out_conv-stage0.yaml --vscode_debug --num_workers=0

   python train_classifier_sideout_classifier_ybranch.py --gpu 6,8 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_ybranch/FFF-FFF-FFF-FFF-out_conv-stage2.yaml

   python train_classifier_sideout_classifier_ybranch.py --gpu 0,1,2,3 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_ybranch/FFF-FFF-FFF-FFF-out_conv-stage2.yaml

   python train_classifier_sideout_classifier_ybranch.py --gpu 4,5 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_ybranch/FFF-FFF-TTT-FFF-out_conv-stage2.yaml

   python train_classifier_sideout_classifier_ybranch.py --gpu 5,7 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_ybranch/TFF-FFF-TTT-FFF-out_conv-stage2-firstTconcat.yaml

   python train_classifier_sideout_classifier_ybranch.py --gpu 4,5 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_ybranch/TTT-FFF-TTT-FFF-out_conv-stage2-firstTTTconcat.yaml
   ```