#!/bin/bash
mkdir -p log
CUDA_VISIBLE_DEVICES=2 nohup python -u 2.0.clean_up_rxn_condition.py --split_group 4 --group 0 > log/get_rxn_condition_uspto_0.log &
CUDA_VISIBLE_DEVICES=3 nohup python -u 2.0.clean_up_rxn_condition.py --split_group 4 --group 1 > log/get_rxn_condition_uspto_1.log &
CUDA_VISIBLE_DEVICES=6 nohup python -u 2.0.clean_up_rxn_condition.py --split_group 4 --group 2 > log/get_rxn_condition_uspto_2.log &
CUDA_VISIBLE_DEVICES=7 nohup python -u 2.0.clean_up_rxn_condition.py --split_group 4 --group 3 > log/get_rxn_condition_uspto_3.log &
