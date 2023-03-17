import json
import os
import pickle
import numpy as np
import pandas as pd
import sys

from tqdm import tqdm
sys.path.extend(['../../baseline_model/'])

from baseline_condition_model import create_rxn_Morgan2FP_separately

BOS, EOS, PAD, MASK = '[BOS]', '[EOS]', '[PAD]', '[MASK]'

def get_idx_dict(data):
    unique_data = list(set(data))
    unique_data.sort()
    idx2data = {i: x for i, x in enumerate(unique_data)}
    data2idx = {x: i for i, x in enumerate(unique_data)}
    return idx2data, data2idx


def get_condition2idx_mapping(all_condition_data: pd.DataFrame):
    col_unique_data = [BOS, EOS, PAD, MASK]
    for col in all_condition_data.columns.tolist():
        one_col_unique = list(set(all_condition_data[col].tolist()))
        col_unique_data.extend(one_col_unique)
    col_unique_data = list(set(col_unique_data))
    col_unique_data.sort()
    idx2data = {i: x for i, x in enumerate(col_unique_data)}
    data2idx = {x: i for i, x in enumerate(col_unique_data)}
    return idx2data, data2idx



if __name__ == '__main__':
    
    with open('../pistachio_rxnclass2name.json', 'r', encoding='utf-8') as f:
        rxn_class2name = json.load(f)
    
    name2rxn_class = {v:k for k,v in rxn_class2name.items()}
    
    
    source_data_path = os.path.join('../../dataset/source_dataset/')
    uspto_condition_dataset_with_rxn_category = pd.read_csv(os.path.join(source_data_path, 'USPTO_condition_final', 'USPTO_condition_pred_category.csv'))
    
    uspto_condition_dataset_rm_suzuki_data = uspto_condition_dataset_with_rxn_category.loc[uspto_condition_dataset_with_rxn_category['rxn_category'] != float(name2rxn_class['Suzuki coupling'])].reset_index(drop=True)
    
    uspto_condition_dataset_rm_suzuki_data_save_path = os.path.join(source_data_path, 'USPTO_condition_final_rm_suzuki')
    if not os.path.exists(uspto_condition_dataset_rm_suzuki_data_save_path):
        os.makedirs(uspto_condition_dataset_rm_suzuki_data_save_path)
    
    uspto_condition_dataset_rm_suzuki_data.to_csv(os.path.join(uspto_condition_dataset_rm_suzuki_data_save_path, 'USPTO_condition_pred_category.csv'), index=False)
    
    uspto_condition_dataset_rm_suzuki_data_rm_category = uspto_condition_dataset_rm_suzuki_data.drop(columns=['rxn_category'])
    
    uspto_condition_dataset_rm_suzuki_data_rm_category.to_csv(os.path.join(uspto_condition_dataset_rm_suzuki_data_save_path, 'USPTO_condition.csv'), index=False)
    
    
    fp_size = 16384

    final_condition_data_path = uspto_condition_dataset_rm_suzuki_data_save_path
    database_fname = 'USPTO_condition.csv'


    database = pd.read_csv(os.path.join(
            final_condition_data_path, database_fname))
    canonical_rxn = database.canonical_rxn.tolist()
    prod_fps = []
    rxn_fps = []
    for rxn in tqdm(canonical_rxn):
        rsmi, psmi = rxn.split('>>')
        [pfp, rfp] = create_rxn_Morgan2FP_separately(
            rsmi, psmi, rxnfpsize=fp_size, pfpsize=fp_size, useFeatures=False, calculate_rfp=True, useChirality=True)
        prod_fps.append(pfp)
        rxn_fps.append(rfp)
    prod_fps = np.array(prod_fps)
    rxn_fps = np.array(rxn_fps)
    np.savez_compressed(os.path.join(final_condition_data_path, '{}_prod_fps'.format(
        database_fname.split('.')[0])), fps=prod_fps)
    np.savez_compressed(os.path.join(final_condition_data_path, '{}_rxn_fps'.format(
        database_fname.split('.')[0])), fps=rxn_fps)
    condition_cols = ['catalyst1', 'solvent1',
                      'solvent2', 'reagent1', 'reagent2']
    for col in condition_cols:
        database[col][pd.isna(database[col])] = ''
        fdata = database[col]
        fpath = os.path.join(final_condition_data_path, '{}_{}.pkl'.format(
            database_fname.split('.')[0], col))
        with open(fpath, 'wb') as f:
            pickle.dump(get_idx_dict(fdata.tolist()), f)

    all_idx2data, all_data2idx = get_condition2idx_mapping(
        database[condition_cols])
    all_idx_mapping_data_fpath = os.path.join(
        final_condition_data_path, '{}_alldata_idx.pkl'.format(database_fname.split('.')[0]))
    with open(all_idx_mapping_data_fpath, 'wb') as f:
        pickle.dump((all_idx2data, all_data2idx), f)
    all_condition_labels = []
    for _, row in tqdm(database[condition_cols].iterrows(), total=len(database)):
        row = list(row)
        row = ['[BOS]'] + row + ['[EOS]']
        all_condition_labels.append([all_data2idx[x] for x in row])

    all_condition_labels_fpath = os.path.join(
        final_condition_data_path, '{}_condition_labels.pkl'.format(database_fname.split('.')[0]))
    with open(all_condition_labels_fpath, 'wb') as f:
        pickle.dump((all_condition_labels), f)
    print('Done!')

    
    
    
    
    