#!/bin/bash

NUM_GPUS_PER_NODE=2
BATCH_SIZE=128
ACCUM_STEP=1

SAVE_PATH=output/RetroSyn_transformer

mkdir -p ${SAVE_PATH}

NCCL_P2P_DISABLE=1 python main.py \
    --task retro \
    --encoder textreact/configs/bert_l6.json \
    --decoder textreact/configs/bert_l6.json \
    --vocab_file textreact/vocab/vocab_smiles.txt \
    --data_path data/RetroSyn/ \
    --train_file train.csv \
    --valid_file valid.csv \
    --test_file test.csv \
    --save_path ${SAVE_PATH} \
    --max_length 256 \
    --max_dec_length 256 \
    --lr 1e-4 \
    --batch_size $((BATCH_SIZE / NUM_GPUS_PER_NODE / ACCUM_STEP)) \
    --gradient_accumulation_steps ${ACCUM_STEP} \
    --epochs 200 \
    --eval_per_epoch 25 \
    --warmup 0.02 \
    --do_train --do_valid --do_test \
    --num_beams 10 \
    --precision 16 \
    --gpus ${NUM_GPUS_PER_NODE}
