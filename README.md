# Attentional Constellation Nets For Few-shot Learning

## Introduction
This repository contains the official code and pretrained models for [Attentional Constellation Nets For Few-shot Learning](https://openreview.net/pdf?id=vujTf_I8Kmc). It enhance structured features by expanding CNNs with a Constellation model, which performs (1) cell feature clustering and encoding with a dense part representation (2) cell feature relation modeling by self-attention mechanism, for the few-shot learning task


For more details, please refer to [Attentional Constellation Nets For Few-shot Learning](https://openreview.net/pdf?id=vujTf_I8Kmc) by [Weijian Xu*](https://weijianxu.com/), [Yifan Xu*](https://yfxu.com/), Huaijin Wang*, and [Zhuowen Tu](https://pages.ucsd.edu/~ztu/).

## Performance
1. Mini-ImageNet (ImageNet dataset)

| Model| Backbone | Acc@ 5-way 1-shot | Acc@ 5-way 5-shot| #Params |
| --- | --- | --- | --- | --- |
| ConstellationNets | Conv-4 | 59.52 ± 0.23 | 75.65 ± 0.18 | 200K |
| ConstellationNets | ResNet-12 | 65.07 ± 0.23 | 80.38 ± 0.16 | 8.4M |

2. CIFAR-FS 

| Model| Backbone | Acc@ 5-way 1-shot | Acc@ 5-way 5-shot| #Params |
| --- | --- | --- | --- | --- |
| ConstellationNets | Conv-4 | 69.0 ± 0.3 | 82.7 ± 0.2 | 200K |
| ConstellationNets | ResNet-12 | 75.7 ± 0.2 | 87.3 ± 0.2 | 8.4M |


3. FC100 

| Model| Backbone | Acc@ 5-way 1-shot | Acc@ 5-way 5-shot| #Params |
| --- | --- | --- | --- | --- |
| ConstellationNets | ResNet-12 | 43.5 ± 0.2 | 59.4 ± 0.2 | 8.4M |




## Changelog

06/25/2021: Code and pre-trained checkpoint for ConstellationNet are released.

## Usage


### Environment Preparation
1. Set up a new conda environment and activate it.
   ```bash
   # Create an environment with Python 3.8.
   conda create -n constells python==3.8
   conda activate constells
   ```

2. Install required packages.
   ```bash
   # Install PyTorch 1.8.0 w/ CUDA 11.1.
   conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge

   # Install yaml
   conda install -c anaconda pyyaml

   # Install tensorboardx.
   conda install -c conda-forge tensorboardx tqdm
   ```

### Code and Datasets Preparation
1. Clone the repo.
   ```bash
   git clone https://github.com/mlpc-ucsd/ConstellationNet.git
   cd ConstellationNet
   ```

2. Download datasets
   ```bash
   ```

### Evaluate Pre-trained Checkpoint

We provide the Constellation Nets checkpoints pre-trained on the Mini-Imagenet, CIFAR-FS and FC100.


| Dataset | Model| Backbone | Acc@ 5-way 1-shot | Acc@ 5-way 5-shot| #Params | SHA-256 (first 8 chars) | URL |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Mini-ImageNet | ConstellationNets | Conv-4 | 59.52 ± 0.23 | 75.65 ± 0.18 | 200K |   |   |
| Mini-ImageNet | ConstellationNets | ResNet-12 | 65.07 ± 0.23 | 80.38 ± 0.16 | 8.4M |  |   |
| CIFAR-FS | ConstellationNets | Conv-4 | 69.0 ± 0.3 | 82.7 ± 0.2 | 200K |  |  |
| CIFAR-FS | ConstellationNets | ResNet-12 | 75.7 ± 0.2 | 87.3 ± 0.2 | 8.4M | |  |
| FC100 | ConstellationNets | ResNet-12 | 43.5 ± 0.2 | 59.4 ± 0.2 | 8.4M | | |



   
### Train
   The following commands provide an example to train the Constellation Net .
   ```bash
   # Usage: bash ./scripts/train.sh [Dataset (mini, cifar-fs, fc100)] [Backbone (conv4, res12)] [GPU index] [Tag]
   bash ./scripts/train.sh mini conv4 0 trial1
   ```

### Evaluate
   The following commands provide an example to evaluate the checkpoint after training.
   ```bash
   # Usage: bash ./scripts/test.sh [Dataset (mini, cifar-fs, fc100)] [Backbone (conv4, res12)] [GPU index] [Tag]
   bash ./scripts/eval.sh mini conv4 0 trial1
   ```

## Citation
```
@article{xuattentional,
  title={ATTENTIONAL CONSTELLATION NETS FOR FEW-SHOT LEARNING},
  author={Xu, Weijian and Xu, Yifan and Wang, Huaijin and Tu, Zhuowen}
}
```

## License
This repository is released under the Apache License 2.0. License can be found in [LICENSE](LICENSE) file.

## Acknowledgment
