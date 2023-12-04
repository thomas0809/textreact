#!/bin/bash

NUM_GPUS_PER_NODE=4
BATCH_SIZE=128
ACCUM_STEP=1

SAVE_PATH=output/RCR_TS_textreact
NN_PATH=data/Tevatron_output/RCR_TS/

mkdir -p ${SAVE_PATH}

NCCL_P2P_DISABLE=1 python main.py \
    --task condition \
    --encoder allenai/scibert_scivocab_uncased \
    --decoder textreact/configs/bert_l6.json \
    --encoder_pretrained \
    --data_path data/RCR_TS/ \
    --train_file train.csv \
    --valid_file val.csv \
    --test_file test.csv \
    --vocab_file textreact/vocab/vocab_condition.txt \
    --corpus_file data/USPTO_rxn_corpus.csv \
    --nn_path ${NN_PATH} \
    --train_nn_file train_rank.json \
    --valid_nn_file val_rank_full.json \
    --test_nn_file test_rank_full.json \
    --num_neighbors 3 \
    --use_gold_neighbor \
    --save_path ${SAVE_PATH} \
    --max_length 512 \
    --shuffle_smiles \
    --mlm --mlm_ratio 0.15 --mlm_layer mlp --mlm_lambda 0.1 \
    --lr 1e-4 \
    --batch_size $((BATCH_SIZE / NUM_GPUS_PER_NODE / ACCUM_STEP)) \
    --gradient_accumulation_steps ${ACCUM_STEP} \
    --epochs 20 \
    --warmup 0.02 \
    --do_train --do_valid --do_test \
    --num_beams 15 \
    --precision 16-mixed \
    --gpus ${NUM_GPUS_PER_NODE}
