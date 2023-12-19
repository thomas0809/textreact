#!/bin/bash

NUM_GPUS_PER_NODE=4
BATCH_SIZE=128
ACCUM_STEP=1

SAVE_PATH=output/RetroSyn_tf_textreact
NN_PATH=data/Tevatron_output/RetroSyn/

mkdir -p ${SAVE_PATH}

NCCL_P2P_DISABLE=1 python main.py \
    --task retro \
    --encoder allenai/scibert_scivocab_uncased \
    --decoder textreact/configs/bert_l6.json \
    --encoder_pretrained \
    --vocab_file textreact/vocab/vocab_smiles.txt \
    --data_path data/RetroSyn/ \
    --train_file train.csv \
    --valid_file valid.csv \
    --test_file test.csv \
    --corpus_file data/USPTO_rxn_corpus.csv \
    --nn_path ${NN_PATH} \
    --train_nn_file train_rank.json \
    --valid_nn_file valid_rank.json \
    --test_nn_file test_rank.json \
    --num_neighbors 3 \
    --use_gold_neighbor \
    --random_neighbor_ratio 0.2 \
    --save_path ${SAVE_PATH} \
    --load_ckpt best.ckpt \
    --max_length 512 \
    --max_dec_length 160 \
    --mlm --mlm_ratio 0.15 --mlm_layer mlp --mlm_lambda 0.1 \
    --lr 1e-4 \
    --batch_size $((BATCH_SIZE / NUM_GPUS_PER_NODE / ACCUM_STEP)) \
    --gradient_accumulation_steps ${ACCUM_STEP} \
    --test_batch_size 32 \
    --epochs 200 \
    --eval_per_epoch 25 \
    --warmup 0.02 \
    --do_train --do_valid --do_test \
    --num_beams 20 \
    --precision 16 \
    --gpus ${NUM_GPUS_PER_NODE}
