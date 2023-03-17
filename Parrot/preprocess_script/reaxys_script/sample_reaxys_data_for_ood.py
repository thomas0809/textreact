import json
import os
import pandas as pd
from collections import defaultdict

from tqdm import tqdm
from calculate_dataset_similarity import calculate_nearest_similarity, convert_condition_name_to_smiles

import json
from multiprocessing import Pool
import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdChemReactions
from pandarallel import pandarallel
import sys
import torch
from rdkit import DataStructs
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

def check_in_list(x, ls):
    if x in ls:
        return True
    else:
        return False


if __name__ == '__main__':
    bins = 200
    dpi = 150
    
    
    fp_size = 1024
    num_workers = 36
    nearest_topk = 5
    pandarallel.initialize(nb_workers=num_workers, progress_bar=True)
    condition_cols = ['catalyst1', 'solvent1', 'solvent2', 'reagent1', 'reagent2']
    source_data_path = '../../dataset/source_dataset/'
    uspto_root = os.path.join(source_data_path, 'USPTO_condition_final')
    reaxys_root = os.path.join(source_data_path, 'Reaxys_total_syn_condition_final_balance')
    condition_cols = ['catalyst1', 'solvent1', 'solvent2', 'reagent1', 'reagent2']
    
    condition_name2smiles_fpath = os.path.join(reaxys_root, 'reaxys_condition_names2smiles_end.json')
    with open(condition_name2smiles_fpath, 'r', encoding='utf-8') as f:
        condition_name2smiles = json.load(f)
    condition_name2smiles = {k:v for k, v in condition_name2smiles.items() if v!=''}
        
    uspto_caculated_fname = os.path.join(uspto_root, 'USPTO_condition_pred_category_rxn_diff_nearest.pkl')
    
    if not os.path.exists(uspto_caculated_fname):
        uspto_condition_dataset = pd.read_csv(os.path.join(uspto_root, 'USPTO_condition_pred_category.csv'))
        for col in condition_cols:
            uspto_condition_dataset[f'{col}_smiles'] = uspto_condition_dataset[col]
    
    else:
        print('Loading calculated USPTO condition dataset...')
        uspto_condition_dataset = pd.read_pickle(uspto_caculated_fname)

    uspto_condition_labels = {'catalyst':[], 'solvent':[], 'reagent':[]}
    uspto_condition_labels_split = {k:[] for k in condition_cols}
    for col in condition_cols:
        uspto_condition_dataset.loc[pd.isna(uspto_condition_dataset[f'{col}_smiles']), f'{col}_smiles'] = ''
        set_col_labels = list(set(uspto_condition_dataset[f'{col}_smiles'].tolist()))
        uspto_condition_labels_split[col] = set_col_labels
        uspto_condition_labels[col.replace('1', '').replace('2', '')] += set_col_labels
        uspto_condition_labels[col.replace('1', '').replace('2', '')] = sorted(uspto_condition_labels[col.replace('1', '').replace('2', '')])
    
    
    reaxys_caculated_fname = os.path.join(reaxys_root, 'reaxys_total_condition_dataset_rm_convert_fail_pred_category_rxn_diff_nearest.pkl')
    
    if not os.path.exists(reaxys_caculated_fname):
        reaxys_total_condition_dataset = pd.read_csv(os.path.join(reaxys_root, 'Reaxys_total_syn_condition_pred_category.csv'))
        have_convert_fail_index = pd.Series([False] * len(reaxys_total_condition_dataset))
        for col in condition_cols:
            reaxys_total_condition_dataset[f'{col}_smiles'] = reaxys_total_condition_dataset[col].apply(lambda x:convert_condition_name_to_smiles(x, condition_name2smiles))
            
            have_convert_fail_index |= reaxys_total_condition_dataset[f'{col}_smiles'] == 'F'

        reaxys_total_condition_dataset_rm_convert_fail_sample = reaxys_total_condition_dataset.loc[~have_convert_fail_index].reset_index(drop=True)
    
    else:
        print('Loading calculated reaxys condition dataset...')
        reaxys_total_condition_dataset_rm_convert_fail_sample = pd.read_pickle(reaxys_caculated_fname).reset_index(drop=True)
        
        have_convert_fail_index = pd.Series([False] * len(reaxys_total_condition_dataset_rm_convert_fail_sample))
        for col in condition_cols:
            reaxys_total_condition_dataset_rm_convert_fail_sample[f'{col}_smiles'] = reaxys_total_condition_dataset_rm_convert_fail_sample[col].apply(lambda x:convert_condition_name_to_smiles(x, condition_name2smiles))
            
            have_convert_fail_index |= reaxys_total_condition_dataset_rm_convert_fail_sample[f'{col}_smiles'] == 'F'


    reaxys_total_condition_dataset_rm_convert_fail_sample = reaxys_total_condition_dataset_rm_convert_fail_sample.loc[~have_convert_fail_index].reset_index(drop=True)
    condition_not_in_index = pd.Series([False] * len(reaxys_total_condition_dataset_rm_convert_fail_sample))
    for col in condition_cols:
        reaxys_total_condition_dataset_rm_convert_fail_sample.loc[pd.isna(reaxys_total_condition_dataset_rm_convert_fail_sample[f'{col}_smiles']), f'{col}_smiles'] = ''
        reaxys_total_condition_dataset_rm_convert_fail_sample[f'{col}_in_uspto'] = reaxys_total_condition_dataset_rm_convert_fail_sample[f'{col}_smiles'].apply(lambda x: check_in_list(x, uspto_condition_labels[col.replace('1', '').replace('2', '')]))
        condition_not_in_index |= reaxys_total_condition_dataset_rm_convert_fail_sample[f'{col}_in_uspto'] == False
    reaxys_total_condition_dataset_rm_convert_fail_sample = reaxys_total_condition_dataset_rm_convert_fail_sample.loc[~condition_not_in_index].reset_index(drop=True)
    
    reaxys_smiles_cols = [f'{col}_smiles' for col in condition_cols]
    exchange_dict = {
        'solvent1_smiles': 'solvent2_smiles',
        'solvent2_smiles': 'solvent1_smiles',
        'reagent1_smiles': 'reagent2_smiles',
        'reagent2_smiles': 'reagent1_smiles',
        }
    all_new_conditions = defaultdict(list)
    condition_not_like_uspto_index = pd.Series([False] * len(reaxys_total_condition_dataset_rm_convert_fail_sample))
    for index, row in tqdm(reaxys_total_condition_dataset_rm_convert_fail_sample.iterrows(), total=len(reaxys_total_condition_dataset_rm_convert_fail_sample)):
        conditions = row[reaxys_smiles_cols]
        all_new_conditions['catalyst1_smiles'].append(row['catalyst1_smiles'])
        if (row['solvent1_smiles'] in uspto_condition_labels_split['solvent1']) and (row['solvent2_smiles'] in uspto_condition_labels_split['solvent2']):
            pass
            # all_new_conditions['solvent1_smiles'].append(row['solvent1_smiles'])
            # all_new_conditions['solvent2_smiles'].append(row['solvent2_smiles'])
        # elif (row['solvent1_smiles'] not in uspto_condition_labels_split['solvent1']) and (row['solvent2_smiles'] == ''):
        #     all_new_conditions['solvent2_smiles'].append(row['solvent1_smiles'])
        #     all_new_conditions['solvent1_smiles'].append(row['solvent2_smiles'])
        # elif (row['solvent2_smiles'] not in uspto_condition_labels_split['solvent2']) and (row['solvent1_smiles'] == ''):
        #     all_new_conditions['solvent2_smiles'].append(row['solvent1_smiles'])
        #     all_new_conditions['solvent1_smiles'].append(row['solvent2_smiles'])
        # elif (row['solvent2_smiles'] not in uspto_condition_labels_split['solvent2']) and (row['solvent1_smiles'] not in uspto_condition_labels_split['solvent1']):
        #     condition_not_like_uspto_index[index] = True
        else:
            condition_not_like_uspto_index[index] = True
        
        if (row['reagent1_smiles'] in uspto_condition_labels_split['reagent1']) and (row['reagent2_smiles'] in uspto_condition_labels_split['reagent2']):
            pass
            # all_new_conditions['reagent1_smiles'].append(row['reagent1_smiles'])
            # all_new_conditions['reagent2_smiles'].append(row['reagent2_smiles'])
        # elif (row['reagent1_smiles'] not in uspto_condition_labels_split['reagent1']) and (row['reagent2_smiles'] == ''):
        #     all_new_conditions['reagent2_smiles'].append(row['reagent1_smiles'])
        #     all_new_conditions['reagent1_smiles'].append(row['reagent2_smiles'])
        # elif (row['reagent2_smiles'] not in uspto_condition_labels_split['reagent2']) and (row['reagent1_smiles'] == ''):
        #     all_new_conditions['reagent2_smiles'].append(row['reagent1_smiles'])
        #     all_new_conditions['reagent1_smiles'].append(row['reagent2_smiles'])
        # elif (row['reagent2_smiles'] not in uspto_condition_labels_split['reagent2']) and (row['reagent1_smiles'] not in uspto_condition_labels_split['reagent1']):
        #     all_new_conditions['reagent1_smiles'].append(row['reagent1_smiles'])
        #     all_new_conditions['reagent2_smiles'].append(row['reagent2_smiles'])
        #     condition_not_like_uspto_index[index] = True
        else:
            # all_new_conditions['reagent1_smiles'].append(row['reagent1_smiles'])
            # all_new_conditions['reagent2_smiles'].append(row['reagent2_smiles'])
            condition_not_like_uspto_index[index] = True
    
    reaxys_total_condition_dataset_rm_convert_fail_sample = reaxys_total_condition_dataset_rm_convert_fail_sample.loc[~condition_not_like_uspto_index].reset_index(drop=True)
    
    reaxys_total_condition_dataset_rm_convert_fail_sample = reaxys_total_condition_dataset_rm_convert_fail_sample.loc[~reaxys_total_condition_dataset_rm_convert_fail_sample.canonical_rxn.isin(uspto_condition_dataset.canonical_rxn)].reset_index(drop=True)
        
            
    
                
                
                
        
        
    
    for col in condition_cols:
        print('na {}: {:.4f}%'.format(col, (reaxys_total_condition_dataset_rm_convert_fail_sample[f'{col}_smiles']=='').sum()/len(reaxys_total_condition_dataset_rm_convert_fail_sample)))
    reaxys_total_condition_dataset_rm_convert_fail_rxn_diff_fps = reaxys_total_condition_dataset_rm_convert_fail_sample['rxn_diff_fps'].tolist()
    reaxys_total_condition_dataset_rm_convert_fail_sample['self_rxn_diff_nearest_{}'.format(nearest_topk)] = reaxys_total_condition_dataset_rm_convert_fail_sample['rxn_diff_fps'].parallel_apply(lambda x: calculate_nearest_similarity(x, reaxys_total_condition_dataset_rm_convert_fail_rxn_diff_fps, topk=nearest_topk))
    
    figure_save_path = './figure'
    if not os.path.exists(figure_save_path):
        os.makedirs(figure_save_path)
    

    reaxys2uspto_rxn_diff = reaxys_total_condition_dataset_rm_convert_fail_sample['reaxys2uspto_rxn_diff_nearest_{}'.format(nearest_topk)].loc[(reaxys_total_condition_dataset_rm_convert_fail_sample['reaxys2uspto_rxn_diff_nearest_{}'.format(nearest_topk)]>0)&(reaxys_total_condition_dataset_rm_convert_fail_sample['reaxys2uspto_rxn_diff_nearest_{}'.format(nearest_topk)]<1)]

    reaxys_rxn_diff = reaxys_total_condition_dataset_rm_convert_fail_sample['self_rxn_diff_nearest_{}'.format(nearest_topk)].loc[(reaxys_total_condition_dataset_rm_convert_fail_sample['self_rxn_diff_nearest_{}'.format(nearest_topk)]>0)&(reaxys_total_condition_dataset_rm_convert_fail_sample['self_rxn_diff_nearest_{}'.format(nearest_topk)]<1)]

    uspto_rxn_diff = uspto_condition_dataset['self_rxn_diff_nearest_{}'.format(nearest_topk)].loc[(uspto_condition_dataset['self_rxn_diff_nearest_{}'.format(nearest_topk)]>0)&(uspto_condition_dataset['self_rxn_diff_nearest_{}'.format(nearest_topk)]<1)]
    
    plt.figure(dpi=dpi)
    palette = plt.get_cmap('tab20c')#'Pastel2')



    sns.histplot(reaxys_rxn_diff,bins=bins,kde=False,color=palette.colors[5], alpha=.6,
                 label="Similarity of Reaxys-TotalSyn-Condition-sampled", stat='probability')
    sns.histplot(uspto_rxn_diff,bins=bins,kde=False,color=palette.colors[10], alpha=.6,
                 label="Similarity of USPTO-Condition", stat='probability')
    sns.histplot(reaxys2uspto_rxn_diff,bins=bins,kde=False,color=palette.colors[0], alpha=.6,
                label="Similarity between USPTO-Condition and Reaxys-TotalSyn-Condition-sampled", stat='probability')


    # plt.title()
    plt.xlabel("Tanimoto similarity")
    plt.ylabel("Probability")
    plt.legend()
    plt.xlim((0,1))

    plt.show()
    
    reaxys_total_condition_dataset_rm_convert_fail_sample_ood = reaxys_total_condition_dataset_rm_convert_fail_sample[[
        'ReaxysRXNID',
        'canonical_rxn',
        'mapped_rxn',
        'catalyst1',
        'solvent1',
        'solvent2',
        'reagent1',
        'reagent2',
        'catalyst1_smiles',
        'solvent1_smiles',
        'solvent2_smiles',
        'reagent1_smiles',
        'reagent2_smiles',
        'rxn_category',
        'rxn_class',

    ]]
    
    reaxys_total_condition_dataset_rm_convert_fail_sample_ood.columns = [
        'ReaxysRXNID',
        'canonical_rxn',
        'mapped_rxn',
        'catalyst1_name',
        'solvent1_name',
        'solvent2_name',
        'reagent1_name',
        'reagent2_name',
        'catalyst1',
        'solvent1',
        'solvent2',
        'reagent1',
        'reagent2',
        'rxn_category',
        'rxn_class',
    ]

    reaxys_total_condition_dataset_rm_convert_fail_sample_ood.to_csv(os.path.join(reaxys_root, 'Reaxys_total_syn_condition_test_for_ood.csv'), index=False)
    
    
'''
catalyst1 : not nan #249 nan, 97.04029478188518 %
solvent1 : not nan #7650 nan, 9.069297515749435 %
solvent2 : not nan #1875 nan, 77.71306311660526 %
reagent1 : not nan #6440 nan, 23.45180078450018 %
reagent2 : not nan #487 nan, 94.21133959348627 %
'''