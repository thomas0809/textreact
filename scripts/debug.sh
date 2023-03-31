#!/bin/bash

NUM_GPUS_PER_NODE=4
SAVE_PATH=output/debug

mkdir -p ${SAVE_PATH}

NCCL_P2P_DISABLE=1 python main.py \
    --encoder allenai/scibert_scivocab_uncased \
    --decoder prajjwal1/bert-mini \
    --data_path data/USPTO_condition_MIT/ \
    --train_file USPTO_condition_train.csv \
    --valid_file USPTO_condition_val.csv \
    --test_file USPTO_condition_test.csv \
    --vocab_file vocab.txt \
    --save_path ${SAVE_PATH} \
    --max_length 160 \
    --lr 2e-4 \
    --batch_size 4 \
    --epochs 4 \
    --warmup 0.02 \
    --do_train --do_valid --do_test \
    --gpus ${NUM_GPUS_PER_NODE} --debug
