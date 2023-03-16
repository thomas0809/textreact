import os
import pandas as pd


'''
取出clean_map_rxn用于masked lm模型训练
'''

if __name__ == '__main__':
    pretrain_data_path = '../../dataset/pretrain_data'
    if not os.path.exists(pretrain_data_path):
        os.makedirs(pretrain_data_path)
    database = pd.read_csv(
        '../../dataset/source_dataset/USPTO_remapped_remove_same_rxn_templates.csv')
    canonicalize_rxn = database['clean_map_rxn']
    canonicalize_rxn_val = canonicalize_rxn.sample(1000, random_state=123)
    canonicalize_rxn_train = canonicalize_rxn.drop(canonicalize_rxn_val.index)

    with open(os.path.join(pretrain_data_path, 'mlm_rxn_train.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(canonicalize_rxn_train.tolist()))
    with open(os.path.join(pretrain_data_path, 'mlm_rxn_val.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(canonicalize_rxn_val.tolist()))
