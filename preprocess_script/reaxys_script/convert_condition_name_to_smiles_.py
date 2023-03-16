import json
import os
from tqdm import tqdm
from urllib.request import urlopen
from urllib.parse import quote

def CIRconvert(ids):
    try:
        url = 'http://cactus.nci.nih.gov/chemical/structure/' + quote(ids) + '/smiles'
        ans = urlopen(url).read().decode('utf8')
        return ans
    except:
        return ''

if __name__ == '__main__':
    
    
    reaxys_not_convert_condition_names_fpath = '../../dataset/source_dataset/Reaxys_total_syn_condition_final/reaxys_condition_names2smiles_not_convert.txt'
    # reaxys_data_path, reaxys_condition_names_fname = os.path.split(reaxys_not_convert_condition_names_fpath)
    # reaxys_condition_names2smiles_fpath = os.path.join(reaxys_data_path, 'reaxys_condition_names2smiles.json')
    # with open(reaxys_condition_names2smiles_fpath, 'r', encoding='utf-8') as f:
    #     conditionName2Smiles = json.load(f)
    
    with open(reaxys_not_convert_condition_names_fpath, 'r', encoding='utf-8') as f:
        condition_names = [x.strip() for x in f.readlines()]
        
    # print('Condtion categories:', len(condition_names))
    

    not_convert_name = []
    conditionName2Smiles = {}
    for name in tqdm(condition_names) :
        smiles = CIRconvert(name)
        if smiles == '':
            not_convert_name.append(name)
        conditionName2Smiles[name] = smiles
            
        
    
    # reaxys_condition_names2smiles_new_fpath = os.path.join(reaxys_data_path, 'reaxys_condition_names2smiles_.json')
    
    # with open(reaxys_condition_names2smiles_new_fpath, 'w', encoding='utf-8') as f:
    #     json.dump(conditionName2Smiles, f)
    
    # with open(reaxys_condition_names2smiles_fpath.replace('.json', '_not_convert_.txt'), 'w', encoding='utf-8') as f:
    #     f.write('\n'.join(not_convert_name))
    
    
    qurey_list = ["IDE.CN=\"{}\"".format(x) for x in condition_names]
    with open('./qurey_file_reaxys_total_syn_condition_names.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(qurey_list))