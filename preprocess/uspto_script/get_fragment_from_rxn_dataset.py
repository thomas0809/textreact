import json
import os
import re
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Recap
from rdkit.Chem import AllChem as Chem
from multiprocessing import Pool
from rdkit.Chem import BRICS
from tqdm import tqdm
import pickle
from collections import defaultdict
from get_pretrain_dataset_with_rxn_center import timeout


def read_txt(fpath):
    with open(fpath, 'r', encoding='utf-8') as f:
        return [x.strip() for x in f.readlines()]
    
    
# def get_frag_from_rxn_recap(rxn):
#     # one_set = set()
#     react, prod = rxn.split('>>')
#     reacts = react.split('.')
#     prods = prod.split('.')
#     # react_mol, prod_mol = Chem.MolFromSmiles(react), Chem.MolFromSmiles(prod)
#     one_fragmet_dict = set()
#     mols = [Chem.MolFromSmiles(smi) for smi in reacts+prods]
#     for mol in mols:
#         hierarch = Recap.RecapDecompose(mol)
#         one_fragmet_dict.update(set(hierarch.GetLeaves().keys()))
#     # one_fragmet_dict = set(list(hierarch_react.GetLeaves().keys())+list(hierarch_prod.GetLeaves().keys()))
#     return one_fragmet_dict
    
    
def get_frag_from_rxn_brics(rxn):
    # one_set = set()
    react, prod = rxn.split('>>')
    reacts = react.split('.')
    prods = prod.split('.')
    # react_mol, prod_mol = Chem.MolFromSmiles(react), Chem.MolFromSmiles(prod)
    one_fragmet_dict = defaultdict(int)
    mols = [Chem.MolFromSmiles(smi) for smi in reacts+prods]
    for mol in mols:
        try:
            with timeout():
                
                frags = list(BRICS.BRICSDecompose(mol))
                for frag in frags:
                    frag = re.sub('\[([0-9]+)\*\]', '*', frag)
                    if frag not in reacts + prods:
                        one_fragmet_dict[frag] += 1
        except Exception as e:
            print(e)
            pass
        
        # one_fragmet_dict.update()
    return one_fragmet_dict
    


if __name__ == '__main__':
    
    
    
    if not os.path.exists('../check_data/frag.pkl'):
        pretrain_dataset_path = '../../dataset/pretrain_data/'
        fnames = ['mlm_rxn_train.txt', 'mlm_rxn_val.txt']
        pretrain_reactions = []
        
        for fname in fnames:
            pretrain_reactions += read_txt(os.path.join(pretrain_dataset_path, fname))
        
        
        fragments = defaultdict(int)
        pool = Pool(12)
        for one_fragmet_dict in tqdm(pool.imap(get_frag_from_rxn_brics, pretrain_reactions), total=len(pretrain_reactions)):
        # for rxn in tqdm(pretrain_reactions):
            # one_fragmet_dict = get_frag_from_rxn_recap(rxn)
            for frag in one_fragmet_dict:
                fragments[frag] += one_fragmet_dict[frag]
        pool.close()
        
        with open('../check_data/frag.pkl', 'wb') as f:
            pickle.dump(fragments, f)
        with open('../check_data/frag.json', 'w', encoding='utf-8') as f:
            json.dump(fragments, f)
        
    else:
        with open('../check_data/frag.pkl', 'rb') as f:
            fragments = pickle.load(f)
        print('Fragments #:', len(fragments))
        fragments_items = list(fragments.items())
        fragments_items.sort(key=lambda x:x[1])
        
        top_number = 10000
        write_top_frag = ['{},{}'.format(x[0], x[1]) for x in fragments_items if x[1] > top_number]
        print(f'fragment count > {top_number}: {len(write_top_frag)}')
        with open(f'../check_data/frag_cnt_nubmer_{top_number}.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(write_top_frag))
            
        
        
    
    
    
    
    