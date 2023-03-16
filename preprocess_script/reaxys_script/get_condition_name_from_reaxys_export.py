import json
import os
import pandas as pd
from rdkit import Chem
import sys
sys.path.append('../uspto_script/')
from notebooks.figure.utils import canonicalize_smiles



if __name__ == '__main__':
    reaxys_export_condition_name_data_fname = '../../dataset/source_dataset/Reaxys_total_syn_condition_final/export_reaxys_total_syn_compound_name_not_convert.xlsx'
    
    reaxys_total_syn_path = os.path.dirname(reaxys_export_condition_name_data_fname)
    
    df_condition_name_to_smiles = pd.read_excel(reaxys_export_condition_name_data_fname,)
    
    df_condition_name_to_smiles = df_condition_name_to_smiles[['SMILES', 'Chemical Name']]
    df_condition_name_to_smiles.drop_duplicates(inplace=True)
    
    exported_condition_name_to_smiles = {k:canonicalize_smiles(v) for k,v in zip(df_condition_name_to_smiles['Chemical Name'].tolist(), df_condition_name_to_smiles['SMILES'].tolist())}
    
    dup_exported_condition_name_to_smiles = {}
    for name, smiles in exported_condition_name_to_smiles.items():
        name_list = [x.strip() for x in name.split(';')]
        for _name in name_list:
            dup_exported_condition_name_to_smiles[_name] = smiles
    
    with open(os.path.join(reaxys_total_syn_path, 'reaxys_condition_names2smiles.json'), 'r', encoding='utf-8') as f:
        condition_name2smiles = json.load(f)
    
    for name, smiles in condition_name2smiles.items():
        if name in dup_exported_condition_name_to_smiles:
            condition_name2smiles[name] = dup_exported_condition_name_to_smiles[name]
        name_list = [x.strip() for x in name.split(';')]
        for _name in name_list:
            if _name in dup_exported_condition_name_to_smiles:
                condition_name2smiles[name] = dup_exported_condition_name_to_smiles[_name]
    
    with open(os.path.join(reaxys_total_syn_path, 'reaxys_condition_names2smiles_end.json'), 'w', encoding='utf-8') as f:
        json.dump(condition_name2smiles, f)