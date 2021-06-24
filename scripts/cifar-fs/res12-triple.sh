#!/bin/bash
echo "Train GPU index:" $1 "Test GPU index:" $2
bash ./scripts/cifar-fs/res12-cifar-fs.sh $1 $2 trial1
bash ./scripts/cifar-fs/res12-cifar-fs.sh $1 $2 trial2
bash ./scripts/cifar-fs/res12-cifar-fs.sh $1 $2 trial3