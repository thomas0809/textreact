#!/bin/bash
mkdir log
nohup python -u 2.0.clean_up_rxn_condition.py --gpu 0 --split_group 4 --group 0 > log/get_rxn_condition_uspto_0.log &
nohup python -u 2.0.clean_up_rxn_condition.py --gpu 0 --split_group 4 --group 1 > log/get_rxn_condition_uspto_1.log & 
nohup python -u 2.0.clean_up_rxn_condition.py --gpu 1 --split_group 4 --group 2 > log/get_rxn_condition_uspto_2.log &
nohup python -u 2.0.clean_up_rxn_condition.py --gpu 1 --split_group 4 --group 3 > log/get_rxn_condition_uspto_3.log &
