import os
try:
    import cirpy
    from tqdm import tqdm
except:
    pass
import json
import time
try:
    import sys
    sys.path.append('D:\software\ChemOfficePrime\ChemScript\Lib')
    from ChemScript16 import *
except:
    pass




if __name__ == '__main__':
    
    
    
    
    reaxys_condition_names_fpath = '../../dataset/source_dataset/Reaxys_total_syn_condition_final/Reaxys_total_syn_condition_name.txt'
    
    with open(reaxys_condition_names_fpath, 'r', encoding='utf-8') as f:
        condition_names = [x.strip() for x in f.readlines()]
        
    print('Condtion categories:', len(condition_names))
    
    conditionName2Smiles = {}
    for name in tqdm(condition_names):
        if '||' in name:
            name = name.split('||')[0]
        try:
            smiles = cirpy.resolve(name, 'smiles')
        except:
            m = StructureData.LoadData(name)
            if hasattr(m, 'Smiles'):
                smiles = m.Smiles
            else:
                smiles = ''
        if smiles:
            conditionName2Smiles[name] = smiles
        else:
            conditionName2Smiles[name] = ''
        # time.sleep(1)
    
    reaxys_data_path, reaxys_condition_names_fname = os.path.split(reaxys_condition_names_fpath)
    reaxys_condition_names2smiles_fpath = os.path.join(reaxys_data_path, 'reaxys_condition_names2smiles.json')
    
    with open(reaxys_condition_names2smiles_fpath, 'w', encoding='utf-8') as f:
        json.dump(conditionName2Smiles, f)