#!/bin/bash
echo "Train GPU index:" $1 "Test GPU index:" $2
bash ./scripts/mini-imagenet/conv4-mini.sh $1 $2 trial1
bash ./scripts/mini-imagenet/conv4-mini.sh $1 $2 trial2
bash ./scripts/mini-imagenet/conv4-mini.sh $1 $2 trial3