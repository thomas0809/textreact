#!/bin/bash

NUM_GPUS_PER_NODE=2
BATCH_SIZE=128
ACCUM_STEP=1

SAVE_PATH=output/rxn_scibert_1e-4_len256_ep20

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
    --max_length 256 \
    --lr 1e-4 \
    --batch_size $((BATCH_SIZE / NUM_GPUS_PER_NODE / ACCUM_STEP)) \
    --gradient_accumulation_steps ${ACCUM_STEP} \
    --epochs 20 \
    --warmup 0.02 \
    --do_train --do_valid --do_test \
    --num_beams 15 \
    --precision 16 \
    --gpus ${NUM_GPUS_PER_NODE}
