import json
import pickle

from utils import canonicalize_smiles, read_pickle
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


if __name__ == '__main__':
    condition_compound = read_pickle('../check_data/condition_compound.pkl')

    merge_condition_compound = {}

    for key in ['c1', 's1', 'r1']:
        merge_condition_compound[key] = condition_compound[key]
        idx = len(merge_condition_compound[key])
        old_smiles_list = list(merge_condition_compound[key].values())
        print('{} old: {}'.format(key, len(old_smiles_list)))
        with open('../check_data/qurey_reaxys_names_{}_smi.txt'.format(key), 'r', encoding='utf-8') as f:
            new_smiles_list = list(set([canonicalize_smiles(x.strip()) for x in f.readlines()]))
        for smi in new_smiles_list:
            if (smi != '') and (smi not in old_smiles_list):
                merge_condition_compound[key][idx] = smi
                idx += 1
        print('{} now: {}'.format(key, len(merge_condition_compound[key])))
    
    with open('../check_data/condition_compound_add_name2smiles.pkl', 'wb') as f:
        pickle.dump(merge_condition_compound, f)
    with open('../check_data/condition_compound_add_name2smiles.json', 'w', encoding='utf-8') as f:
        json.dump(merge_condition_compound, f)
    