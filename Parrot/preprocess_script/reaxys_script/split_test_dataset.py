import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
import sys
sys.path.append('../../baseline_model/')
from baseline_condition_model import create_rxn_Morgan2FP_separately

def get_idx_dict(data):
    unique_data = list(set(data))
    unique_data.sort()
    idx2data = {i: x for i, x in enumerate(unique_data)}
    data2idx = {x: i for i, x in enumerate(unique_data)}
    return idx2data, data2idx

def proprecess_baseline_dataset(database: pd.DataFrame, final_condition_data_path, database_fname):
    canonical_rxn = database.canonical_rxn.tolist()
    prod_fps = []
    rxn_fps = []
    for rxn in tqdm(canonical_rxn):
        rsmi, psmi = rxn.split('>>')
        [pfp, rfp] = create_rxn_Morgan2FP_separately(
            rsmi, psmi, rxnfpsize=fp_size, pfpsize=fp_size, useFeatures=False, calculate_rfp=True, useChirality=True)
        rxn_fp = pfp - rfp
        prod_fps.append(pfp)
        rxn_fps.append(rxn_fp)
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
    print('save to {}'.format(os.path.join(final_condition_data_path, database_fname)))
    database.to_csv(os.path.join(final_condition_data_path, database_fname), index=False)



if __name__ == '__main__':
    fp_size = 16384
    source_data_path = '../../dataset/source_dataset/'
    final_condition_data_path = os.path.join(
        source_data_path, 'Reaxys_total_syn_condition_final')
    database_fname = 'Reaxys_total_syn_condition.csv'


    database = pd.read_csv(os.path.join(
            final_condition_data_path, database_fname))
    
    test_dataset = database.loc[database['dataset'] == 'test']
    
    have_catalyst_test_datset = test_dataset.loc[~test_dataset['catalyst1'].isna()].reset_index(drop=True)
    print('Include Catalyst dataset # {}'.format(len(have_catalyst_test_datset)))
    
    proprecess_baseline_dataset(have_catalyst_test_datset, final_condition_data_path=final_condition_data_path, database_fname='Reaxys_total_syn_condition_test_have_catalyst.csv')
    
    catalyst_na_test_datset = test_dataset.loc[test_dataset['catalyst1'].isna()].reset_index(drop=True)
    print('Include Catalyst dataset # {}'.format(len(catalyst_na_test_datset)))
    
    proprecess_baseline_dataset(catalyst_na_test_datset, final_condition_data_path=final_condition_data_path, database_fname='Reaxys_total_syn_condition_test_catalyst_na.csv')
    print('Catalyst na dataset # {}'.format(len(catalyst_na_test_datset)))
    
    
    