#!/bin/bash
# run_cifar10.sh

DATASET="cifar10"
EPOCHS=100
BATCH_SIZE=128
LR=0.1
MOMENTUM=0.9
WD=5e-4
OUT_DIR="Results"
LOG_PATH="Results/cifar10.log"

python main.py \
    --dataset $DATASET \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --momentum $MOMENTUM \
    --wd $WD \
    --out_dir $OUT_DIR \
    --log_path $LOG_PATH
