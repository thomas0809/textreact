from tqdm import tqdm
import xmltodict
from collections import OrderedDict
import pandas as pd
import os

from utils import get_writer

def read_xml2dict(xml_fpath):
    with open(xml_fpath, 'r') as f:
        data = xmltodict.parse(f.read())
    reaction_and_condition_dict = {
        'source': [],
        'rxn_smiles': [],
        'solvent': [],
        'catalyst': [],
        'reagent': [],
    }
    try:
        reaction_data_list = data['reactionList']['reaction']
    except:
        return pd.DataFrame(reaction_and_condition_dict)


    for rxn_data in reaction_data_list:
        if type(rxn_data) is str:
            continue
        if rxn_data['spectatorList'] is None:
            continue
        if rxn_data['dl:reactionSmiles'] is None:
            continue
        try:
            spectator_obj = rxn_data['spectatorList']['spectator']
        except:
            continue

        s_list = []
        c_list = []
        r_list = []
        if type(spectator_obj) is list:
            for one_spectator in spectator_obj:
                if 'identifier' not in one_spectator:
                    continue
                if one_spectator['@role'] == 'solvent':
                    if type(one_spectator['identifier']) is list:
                        for identifier in one_spectator['identifier']:
                            if identifier['@dictRef'] == 'cml:smiles':
                                s_list.append(identifier['@value'])
                    elif type(one_spectator['identifier']) is OrderedDict:
                        identifier = one_spectator['identifier']
                        if identifier['@dictRef'] == 'cml:smiles':
                            s_list.append(identifier['@value'])
                elif one_spectator['@role'] == 'catalyst':
                    if type(one_spectator['identifier']) is list:
                        for identifier in one_spectator['identifier']:
                            if identifier['@dictRef'] == 'cml:smiles':
                                c_list.append(identifier['@value'])
                    elif type(one_spectator['identifier']) is OrderedDict:
                        identifier = one_spectator['identifier']
                        if identifier['@dictRef'] == 'cml:smiles':
                            c_list.append(identifier['@value'])
                elif one_spectator['@role'] == 'reagent':
                    if type(one_spectator['identifier']) is list:
                        for identifier in one_spectator['identifier']:
                            if identifier['@dictRef'] == 'cml:smiles':
                                r_list.append(identifier['@value'])
                    elif type(one_spectator['identifier']) is OrderedDict:
                        identifier = one_spectator['identifier']
                        if identifier['@dictRef'] == 'cml:smiles':
                            r_list.append(identifier['@value'])
                else:
                    print(one_spectator['@role'])

        elif type(spectator_obj) is OrderedDict:
            one_spectator = spectator_obj
            if 'identifier' not in one_spectator:
                continue
            if one_spectator['@role'] == 'solvent':
                if type(one_spectator['identifier']) is list:
                    for identifier in one_spectator['identifier']:
                        if identifier['@dictRef'] == 'cml:smiles':
                            s_list.append(identifier['@value'])
                elif type(one_spectator['identifier']) is OrderedDict:
                    identifier = one_spectator['identifier']
                    if identifier['@dictRef'] == 'cml:smiles':
                        s_list.append(identifier['@value'])
            elif one_spectator['@role'] == 'catalyst':
                if type(one_spectator['identifier']) is list:
                    for identifier in one_spectator['identifier']:
                        if identifier['@dictRef'] == 'cml:smiles':
                            c_list.append(identifier['@value'])
                elif type(one_spectator['identifier']) is OrderedDict:
                    identifier = one_spectator['identifier']
                    if identifier['@dictRef'] == 'cml:smiles':
                        c_list.append(identifier['@value'])
            elif one_spectator['@role'] == 'reagent':
                if type(one_spectator['identifier']) is list:
                    for identifier in one_spectator['identifier']:
                        if identifier['@dictRef'] == 'cml:smiles':
                            r_list.append(identifier['@value'])
                elif type(one_spectator['identifier']) is OrderedDict:
                    identifier = one_spectator['identifier']
                    if identifier['@dictRef'] == 'cml:smiles':
                        r_list.append(identifier['@value'])
            else:
                print(one_spectator['@role'])
        else:
            print('Warning spectator_obj is not in (list, OrderedDict)!!!')
        s_list = list(set(s_list))
        c_list = list(set(c_list))
        r_list = list(set(r_list))
        reaction_and_condition_dict['solvent'].append('.'.join(s_list))
        reaction_and_condition_dict['catalyst'].append('.'.join(c_list))
        reaction_and_condition_dict['reagent'].append('.'.join(r_list))
        reaction_and_condition_dict['rxn_smiles'].append(
            rxn_data['dl:reactionSmiles'])
        reaction_and_condition_dict['source'].append(
            rxn_data['dl:source']['dl:documentId'])
    return pd.DataFrame(reaction_and_condition_dict)


if __name__ == '__main__':
    uspto_org_path = '../../dataset/source_dataset/uspto_org_xml/'

    # one_xml_path = '/home/xiaoruiwang/data/ubuntu_work_beta/single_step_work/rxn_condition_predictor/dataset/source_dataset/uspto_org_xml/grants/1976/pftaps19760106_wk01.xml'
    # reaction_and_condition_df = pd.DataFrame()
    xml_path_list = []
    for root, _, files in os.walk(uspto_org_path, topdown=False):
        for fname in files:
            if fname.endswith('.xml'):
                xml_path_list.append(os.path.join(root, fname))
    reaction_and_condition_df = pd.DataFrame({
        'source': [],
        'rxn_smiles': [],
        'solvent': [],
        'catalyst': [],
        'reagent': [],
    })
    fout, writer = get_writer('../../dataset/source_dataset/uspto_rxn_condition.csv',
                              reaction_and_condition_df.columns.tolist())
    cnt = 0
    for i, path in tqdm(enumerate(xml_path_list), total=len(xml_path_list)):
        df = read_xml2dict(path)
        for row in df.itertuples():
            writer.writerow(list(row)[1:])
            fout.flush()
        # df.to_csv('../../dataset/source_dataset/uspto_rxn_condition.csv', mode='a', header=False, index=False)
        cnt += len(df)
        if i % 20 == 0:
            print(f'step {i}: {cnt} data')
    fout.close()
    # reaction_and_condition_df = pd.read_csv('../../dataset/source_dataset/uspto_rxn_condition.csv')
    # reaction_and_condition_df.columns = df.columns
    # reaction_and_condition_df.to_csv('../../dataset/source_dataset/uspto_rxn_condition_done.csv')
    print('Done')
    
