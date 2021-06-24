#!/bin/bash
echo "Train GPU index:" $1 "Test GPU index:" $2
bash ./scripts/fc100/res12-fc100.sh $1 $2 trial1
bash ./scripts/fc100/res12-fc100.sh $1 $2 trial2
bash ./scripts/fc100/res12-fc100.sh $1 $2 trial3