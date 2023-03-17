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


def create_rxn_Morgan2FP_separately_arrint8(rsmi, psmi, rxnfpsize=16384, pfpsize=16384, useFeatures=False, calculate_rfp=True, useChirality=False):
    # Similar as the above function but takes smiles separately and returns pfp and rfp separately

    rsmi = rsmi.encode('utf-8')
    psmi = psmi.encode('utf-8')
    try:
        mol = Chem.MolFromSmiles(rsmi)
    except Exception as e:
        print(e)
        return
    try:
        fp_bit = AllChem.GetMorganFingerprintAsBitVect(
            mol=mol, radius=2, nBits=rxnfpsize, useFeatures=useFeatures, useChirality=useChirality)
        fp = np.empty(rxnfpsize, dtype='int8')
        DataStructs.ConvertToNumpyArray(fp_bit, fp)
    except Exception as e:
        print("Cannot build reactant fp due to {}".format(e))
        return
    rfp = fp

    try:
        mol = Chem.MolFromSmiles(psmi)
    except Exception as e:
        return
    try:
        fp_bit = AllChem.GetMorganFingerprintAsBitVect(
            mol=mol, radius=2, nBits=pfpsize, useFeatures=useFeatures, useChirality=useChirality)
        fp = np.empty(pfpsize, dtype='int8')
        DataStructs.ConvertToNumpyArray(fp_bit, fp)
    except Exception as e:
        print("Cannot build product fp due to {}".format(e))
        return
    pfp = fp
    return [pfp, rfp]

def convert_condition_name_to_smiles(name, condition_name2smiles):
    try:
        return condition_name2smiles[name]
    except:
        if pd.isna(name):
            return name
        else:
            return 'F'

def calculate_similarity(x, y_list):
    return DataStructs.BulkTanimotoSimilarity(x, y_list)


def calculate_reactants_fps(rxn, fp_size):
    rsmi, psmi = rxn.split('>>')
    [pfp, rfp] = create_rxn_Morgan2FP_separately_arrint8(
                rsmi, psmi, rxnfpsize=fp_size, pfpsize=fp_size, useFeatures=False, calculate_rfp=True, useChirality=True)
    rxnfp = pfp - rfp
    rxn_symbol_fp = np.concatenate([pfp, rxnfp])             # 使用产物指纹和反应差异指纹同时描述一个反应
    
    # bitstring="".join(rxn_symbol_fp.astype(str))
    # rxn_symbol_fp_bit = DataStructs.CreateFromBitString(bitstring)
    return rxn_symbol_fp

def calculate_reactants_fps(rxn, fp_size):
    rsmi, psmi = rxn.split('>>')
    rmol = Chem.MolFromSmiles(rsmi)
    rfp_bit = AllChem.GetMorganFingerprintAsBitVect(
            mol=rmol, radius=2, nBits=fp_size, useFeatures=True, useChirality=True)
    return rfp_bit

def calculate_rxn_diff_fps(rxn):
    rdrxn = rdChemReactions.ReactionFromSmarts(rxn)
    return rdChemReactions.CreateDifferenceFingerprintForReaction(rdrxn)


def calculate_nearest_similarity(x, y_list, topk=5):
    similar_vector = torch.tensor(calculate_similarity(x, y_list))
    mean_topk = similar_vector.topk(topk)[0].mean()
    return mean_topk.item()

def run_calculate_nearest_similarity_tasks(tuple_data):
    i, all_len, fp, fps, topk = tuple_data
    if i > 0 and i % 1000 == 0:
        print(f"Processing {i}th Reaction / {all_len}")
    return calculate_nearest_similarity(fp, fps, topk=topk)
    
    
    

if __name__ == '__main__':
    debug = True
    print('Debug:', debug)
    source_data_path = '../../dataset/source_dataset/'
    
    bins = 200
    dpi = 150
    
    
    fp_size = 1024
    num_workers = 36
    nearest_topk = 5
    pandarallel.initialize(nb_workers=num_workers, progress_bar=True)
    condition_cols = ['catalyst1', 'solvent1', 'solvent2', 'reagent1', 'reagent2']
    
    

    
    uspto_root = os.path.join(source_data_path, 'USPTO_condition_final')
    reaxys_root = os.path.join(source_data_path, 'Reaxys_total_syn_condition_final')
    
    condition_name2smiles_fpath = os.path.join(reaxys_root, 'reaxys_condition_names2smiles_end.json')
    with open(condition_name2smiles_fpath, 'r', encoding='utf-8') as f:
        condition_name2smiles = json.load(f)
    
    uspto_caculated_fname = os.path.join(uspto_root, 'USPTO_condition_pred_category_rxn_diff_nearest.pkl')
    if debug:
        uspto_caculated_fname = uspto_caculated_fname.replace('.pkl', '_debug.pkl')
    if not os.path.exists(uspto_caculated_fname):
        uspto_condition_dataset = pd.read_csv(os.path.join(uspto_root, 'USPTO_condition_pred_category.csv'), nrows=3000 if debug else None)
        print('Calculating uspto condition reaction difference fps...')
        uspto_condition_dataset['rxn_diff_fps'] = uspto_condition_dataset['canonical_rxn'].parallel_apply(lambda x: calculate_rxn_diff_fps(x))
        uspto_condition_dataset_rxn_diff_fps = uspto_condition_dataset['rxn_diff_fps'].tolist()
        print('Calculating uspto_condition self_rxn_diff_nearest_{}'.format(nearest_topk))
        uspto_condition_dataset['self_rxn_diff_nearest_{}'.format(nearest_topk)] = uspto_condition_dataset['rxn_diff_fps'].parallel_apply(lambda x: calculate_nearest_similarity(x, uspto_condition_dataset_rxn_diff_fps, topk=nearest_topk))

        for col in condition_cols:
            uspto_condition_dataset[f'{col}_smiles'] = uspto_condition_dataset[col]
        print('Saving calculated USPTO condition dataset...')
        uspto_condition_dataset.to_pickle(uspto_caculated_fname)
    else:
        print('Loading calculated USPTO condition dataset...')
        uspto_condition_dataset = pd.read_pickle(uspto_caculated_fname)
        uspto_condition_dataset_rxn_diff_fps = uspto_condition_dataset['rxn_diff_fps'].tolist()
    
    
    reaxys_caculated_fname = os.path.join(reaxys_root, 'reaxys_total_condition_dataset_rm_convert_fail_pred_category_rxn_diff_nearest.pkl')
    reaxys_caculated_fname = reaxys_caculated_fname.replace('.pkl', '_debug.pkl')
    if not os.path.exists(reaxys_caculated_fname):
        reaxys_total_condition_dataset = pd.read_csv(os.path.join(reaxys_root, 'Reaxys_total_syn_condition_pred_category.csv'), nrows=3000 if debug else None)

        
        
        have_convert_fail_index = pd.Series([False] * len(reaxys_total_condition_dataset))
        for col in condition_cols:
            reaxys_total_condition_dataset[f'{col}_smiles'] = reaxys_total_condition_dataset[col].apply(lambda x:convert_condition_name_to_smiles(x, condition_name2smiles))
            
            have_convert_fail_index |= reaxys_total_condition_dataset[f'{col}_smiles'] == 'F'
        
        reaxys_total_condition_dataset_rm_convert_fail = reaxys_total_condition_dataset.loc[~have_convert_fail_index].reset_index(drop=True)
        print('Calculating reaxys condition reaction difference fps...')
        reaxys_total_condition_dataset_rm_convert_fail['rxn_diff_fps'] = reaxys_total_condition_dataset_rm_convert_fail['canonical_rxn'].parallel_apply(lambda x: calculate_rxn_diff_fps(x))
        reaxys_total_condition_dataset_rm_convert_fail_rxn_diff_fps = reaxys_total_condition_dataset_rm_convert_fail['rxn_diff_fps'].tolist()
        print('Calculating reaxys condition self_rxn_diff_nearest_{}'.format(nearest_topk))
        reaxys_total_condition_dataset_rm_convert_fail['self_rxn_diff_nearest_{}'.format(nearest_topk)] = reaxys_total_condition_dataset_rm_convert_fail['rxn_diff_fps'].parallel_apply(lambda x: calculate_nearest_similarity(x, reaxys_total_condition_dataset_rm_convert_fail_rxn_diff_fps, topk=nearest_topk))
        print('Calculating reaxys condition reaxys2uspto_rxn_diff_nearest_{}'.format(nearest_topk))
        reaxys_total_condition_dataset_rm_convert_fail['reaxys2uspto_rxn_diff_nearest_{}'.format(nearest_topk)] = reaxys_total_condition_dataset_rm_convert_fail['rxn_diff_fps'].parallel_apply(lambda x: calculate_nearest_similarity(x, uspto_condition_dataset_rxn_diff_fps, topk=nearest_topk))
        print('Saving calculated reaxys condition dataset...')
        reaxys_total_condition_dataset_rm_convert_fail.to_pickle(reaxys_caculated_fname)
    else:
        print('Loading calculated reaxys condition dataset...')
        reaxys_total_condition_dataset_rm_convert_fail = pd.read_pickle(reaxys_caculated_fname)
    
    
    figure_save_path = './figure'
    if not os.path.exists(figure_save_path):
        os.makedirs(figure_save_path)
    

    reaxys2uspto_rxn_diff = reaxys_total_condition_dataset_rm_convert_fail['reaxys2uspto_rxn_diff_nearest_{}'.format(nearest_topk)].loc[(reaxys_total_condition_dataset_rm_convert_fail['reaxys2uspto_rxn_diff_nearest_{}'.format(nearest_topk)]>0)&(reaxys_total_condition_dataset_rm_convert_fail['reaxys2uspto_rxn_diff_nearest_{}'.format(nearest_topk)]<1)]

    reaxys_rxn_diff = reaxys_total_condition_dataset_rm_convert_fail['self_rxn_diff_nearest_{}'.format(nearest_topk)].loc[(reaxys_total_condition_dataset_rm_convert_fail['self_rxn_diff_nearest_{}'.format(nearest_topk)]>0)&(reaxys_total_condition_dataset_rm_convert_fail['self_rxn_diff_nearest_{}'.format(nearest_topk)]<1)]

    uspto_rxn_diff = uspto_condition_dataset['self_rxn_diff_nearest_{}'.format(nearest_topk)].loc[(uspto_condition_dataset['self_rxn_diff_nearest_{}'.format(nearest_topk)]>0)&(uspto_condition_dataset['self_rxn_diff_nearest_{}'.format(nearest_topk)]<1)]
    
    plt.figure(dpi=dpi)
    palette = plt.get_cmap('tab20c')#'Pastel2')



    sns.histplot(reaxys_rxn_diff,bins=bins,kde=False,color=palette.colors[5], alpha=.6,
                 label="Similarity of Reaxys-Total-Syn-Condition", stat='probability')
    sns.histplot(uspto_rxn_diff,bins=bins,kde=False,color=palette.colors[10], alpha=.6,
                 label="Similarity of USPTO-Condition", stat='probability')
    sns.histplot(reaxys2uspto_rxn_diff,bins=bins,kde=False,color=palette.colors[0], alpha=.6,
                label="Similarity between USPTO-Condition and Reaxys-Total-Syn-Condition", stat='probability')


    # plt.title()
    plt.xlabel("Tanimoto similarity")
    plt.ylabel("Probability")
    plt.legend()
    plt.xlim((0,1))

    plt.show()
