from collections import defaultdict
import os
import pandas as pd
import random
from tqdm import tqdm


if __name__ == '__main__':
    debug = False
    print('Debug:', debug)
    seed = 123
    random.seed(seed)
    # 数据集： canonical_rxn --> catalyst, solvent1, solvent2, reagent1, reagent2
    split_token = '分'
    split_frac = (0.8, 0.1, 0.1) # train:val:test
    source_data_path = '../../dataset/source_dataset/'
    database_remove_below_threshold_fname = 'uspto_rxn_condition_remapped_and_reassign_condition_role_rm_duplicate_rm_excess.csv'
    final_condition_data_path = os.path.join(
        source_data_path, 'USPTO_condition_final')
    if not os.path.exists(final_condition_data_path):
        os.makedirs(final_condition_data_path)
    if debug:
        database = pd.read_csv(os.path.join(
        source_data_path, database_remove_below_threshold_fname), nrows=10000)
    else:
        database = pd.read_csv(os.path.join(
        source_data_path, database_remove_below_threshold_fname))
    
    database = database[['source','canonical_rxn', 'catalyst_split', 'solvent_split', 'reagent_split']]
    split_solvent = database['solvent_split'].str.split(split_token, 1, expand=True)
    database['solvent1'], database['solvent2'] = split_solvent[0], split_solvent[1]
    split_reagent = database['reagent_split'].str.split(split_token, 1, expand=True)
    database['reagent1'], database['reagent2'] = split_reagent[0], split_reagent[1]
    database = database[['source','canonical_rxn', 'catalyst_split',
                         'solvent1', 'solvent2', 'reagent1', 'reagent2']]
    database.columns = ['source','canonical_rxn', 'catalyst1',
                         'solvent1', 'solvent2', 'reagent1', 'reagent2']
    database_sample = database.sample(frac=1, random_state=seed)

    can_rxn2idx_dict = defaultdict(list)
    for idx, row in tqdm(database_sample.iterrows(), total=len(database_sample)):
        can_rxn2idx_dict[row.canonical_rxn].append(idx)
    train_idx, val_idx, test_idx = [], [], []
    can_rxn2idx_dict_items = list(can_rxn2idx_dict.items())
    random.shuffle(can_rxn2idx_dict_items)
    all_data_number = len(database_sample)
    for rxn, idx_list in tqdm(can_rxn2idx_dict_items):
        if len(idx_list) == 1:
            if len(test_idx) < split_frac[2] * all_data_number:
                test_idx += idx_list
            elif len(val_idx) < split_frac[1] * all_data_number:
                val_idx += idx_list
            else:
                train_idx += idx_list
        else:
            train_idx += idx_list
    
    database_sample.loc[train_idx, 'dataset'] = 'train'
    database_sample.loc[val_idx, 'dataset'] = 'val'
    database_sample.loc[test_idx, 'dataset'] = 'test'
    database_sample.to_csv(os.path.join(final_condition_data_path, 'USPTO_condition.csv'), index=False)