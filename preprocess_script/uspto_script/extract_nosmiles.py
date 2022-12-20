import json
import os
import pickle
# from show_coincidence import read_json
from rdkit import Chem
from rdkit import RDLogger

from utils import canonicalize_smiles, read_pickle
RDLogger.DisableLog('rdApp.*')



if __name__ == '__main__':
    all_data = {}
    for fname in os.listdir('../'):
        name, ext = os.path.splitext(fname)
        fpath = os.path.join('..', fname)
        if ext == '.pickle':
            all_data['{}'.format(name.replace('_dict', ''))] = read_pickle(fpath)
    all_reaxys_name = []
    all_data_reaxys_name_comp = {}
    all_data_rm = {}
    for k in all_data:
        can_err = 0
        all_data_rm[k] = {}
        all_data_reaxys_name_comp[k] = []
        idx = 0
        for comp_idx in all_data[k]:
            comp = all_data[k][comp_idx]
            if 'Reaxys' not in comp:
                if comp == '':
                    all_data_rm[k][idx] = comp
                    idx += 1
                    continue
                comp_can = canonicalize_smiles(comp)
                if comp_can == '':
                    can_err += 1
                    continue
                all_data_rm[k][idx] = comp_can
                idx += 1
            elif 'Reaxys Name' in comp:
                all_data_reaxys_name_comp[k].append(comp)
                all_reaxys_name.append(comp)
        
        print(f'{k} canonicalize fail: {can_err}')
        print('{}: {}'.format(k, len(all_data_rm[k])))
        print('{}: Reaxys Name {}'.format(k, len(all_data_reaxys_name_comp[k])))
    
    with open('../check_data/condition_compound.pkl', 'wb') as f:
        pickle.dump(all_data_rm, f)
    with open('../check_data/reaxys_name_comp.pkl', 'wb') as f:
        pickle.dump(all_data_reaxys_name_comp, f)
    
    with open('../check_data/condition_compound.json', 'w', encoding='utf-8') as f:
        json.dump(all_data_rm, f)
    with open('../check_data/reaxys_name_comp.json', 'w', encoding='utf-8') as f:
        json.dump(all_data_reaxys_name_comp, f)
    
    with open('../check_data/all_reaxys_name.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(all_reaxys_name))
    
