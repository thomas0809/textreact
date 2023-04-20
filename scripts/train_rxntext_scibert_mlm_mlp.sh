#!/bin/bash

NUM_GPUS_PER_NODE=4
BATCH_SIZE=128
ACCUM_STEP=1

SAVE_PATH=output/rxntext_scibert_nn3_mlm_mlp_ep20

mkdir -p ${SAVE_PATH}

NCCL_P2P_DISABLE=1 python main.py \
    --task condition \
    --encoder allenai/scibert_scivocab_uncased \
    --decoder prajjwal1/bert-mini \
    --encoder_pretrained \
    --data_path data/USPTO_condition_MIT/ \
    --train_file USPTO_condition_train.csv \
    --valid_file USPTO_condition_val.csv \
    --test_file USPTO_condition_test.csv \
    --vocab_file textreact/vocab/vocab_condition.txt \
    --corpus_file data/USPTO_rxn_corpus.csv \
    --cache_path /scratch/yujieq/textreact/ \
    --nn_path ../tevatron/output/rxntext_ep50 \
    --train_nn_file train_rank.json \
    --valid_nn_file val_rank.json \
    --test_nn_file test_rank.json \
    --num_neighbors 3 \
    --save_path ${SAVE_PATH} \
    --max_length 512 \
    --shuffle_smiles \
    --val_metric val_acc \
    --mlm --mlm_ratio 0.15 --mlm_layer mlp \
    --lr 1e-4 \
    --batch_size $((BATCH_SIZE / NUM_GPUS_PER_NODE / ACCUM_STEP)) \
    --gradient_accumulation_steps ${ACCUM_STEP} \
    --epochs 20 \
    --warmup 0.02 \
    --do_train --do_valid --do_test \
    --num_beams 15 \
    --precision 16-mixed \
    --gpus ${NUM_GPUS_PER_NODE}
