#!/bin/bash

NUM_GPUS_PER_NODE=2
BATCH_SIZE=128
ACCUM_STEP=1

SAVE_PATH=output/RCR_transformer

mkdir -p ${SAVE_PATH}

NCCL_P2P_DISABLE=1 python main.py \
    --task condition \
    --encoder allenai/scibert_scivocab_uncased \
    --decoder textreact/configs/bert_l6.json \
    --encoder_pretrained \
    --data_path data/RCR/ \
    --train_file train.csv \
    --valid_file val.csv \
    --test_file test.csv \
    --vocab_file textreact/vocab/vocab_condition.txt \
    --save_path ${SAVE_PATH} \
    --max_length 256 \
    --shuffle_smiles \
    --lr 1e-4 \
    --batch_size $((BATCH_SIZE / NUM_GPUS_PER_NODE / ACCUM_STEP)) \
    --gradient_accumulation_steps ${ACCUM_STEP} \
    --epochs 20 \
    --warmup 0.02 \
    --do_train --do_valid --do_test \
    --num_beams 15 \
    --precision 16-mixed \
    --gpus ${NUM_GPUS_PER_NODE}
