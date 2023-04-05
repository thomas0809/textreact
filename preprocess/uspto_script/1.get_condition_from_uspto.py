from tqdm import tqdm
import xmltodict
from collections import OrderedDict, Counter
import pandas as pd
import os
import copy
import json

from utils import get_writer


patent_cnt = Counter()
patent_info = {}
CONDITION_DICT = {
        'id': [],
        'source': [],
        'year': [],
        'patent_type': [],
        'rxn_smiles': [],
        'solvent': [],
        'catalyst': [],
        'reagent': [],
    }
CORPUS_DICT = {
        'id': [],
        'year': [],
        'patent_type': [],
        'xml': [],
        'heading_text': [],
        'paragraph_text': [],
    }


def read_xml2dict(xml_fpath):
    with open(xml_fpath, 'r') as f:
        data = xmltodict.parse(f.read())
    reaction_and_condition_dict = copy.deepcopy(CONDITION_DICT)
    corpus_dict = copy.deepcopy(CORPUS_DICT)

    try:
        reaction_data_list = data['reactionList']['reaction']
    except:
        return reaction_and_condition_dict, corpus_dict


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

        patent_id = rxn_data['dl:source']['dl:documentId']
        heading_text = rxn_data['dl:source'].get('dl:headingText', '')
        paragraph_text = rxn_data['dl:source'].get('dl:paragraphText', '')
        year = os.path.dirname(xml_fpath).split('/')[-1]
        patent_type = 'grant' if 'grants' in xml_fpath else 'application'
        patent_info[patent_id] = {
            'year': int(year),
            'type': patent_type
        }

        s_list = []
        c_list = []
        r_list = []
        if type(spectator_obj) is list:
            pass
        elif type(spectator_obj) is OrderedDict:
            spectator_obj = [spectator_obj]
        else:
            print('Warning spectator_obj is not in (list, OrderedDict)!!!')

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

        rxn_id = patent_id + '_' + str(patent_cnt[patent_id])
        patent_cnt[patent_id] += 1

        s_list = list(set(s_list))
        c_list = list(set(c_list))
        r_list = list(set(r_list))
        reaction_and_condition_dict['solvent'].append('.'.join(s_list))
        reaction_and_condition_dict['catalyst'].append('.'.join(c_list))
        reaction_and_condition_dict['reagent'].append('.'.join(r_list))
        reaction_and_condition_dict['rxn_smiles'].append(rxn_data['dl:reactionSmiles'])
        reaction_and_condition_dict['source'].append(patent_id)
        reaction_and_condition_dict['id'].append(rxn_id)
        reaction_and_condition_dict['year'].append(year)
        reaction_and_condition_dict['patent_type'].append(patent_type)

        corpus_dict['id'].append(rxn_id)
        corpus_dict['xml'].append(os.path.basename(xml_fpath))
        corpus_dict['heading_text'].append(heading_text)
        corpus_dict['paragraph_text'].append(paragraph_text)
        corpus_dict['year'].append(year)
        corpus_dict['patent_type'].append(patent_type)

    return reaction_and_condition_dict, corpus_dict


if __name__ == '__main__':
    uspto_org_path = '/Mounts/rbg-storage1/users/yujieq/USPTO/'

    # reaction_and_condition_df = pd.DataFrame()
    xml_path_list = []
    for root, _, files in os.walk(uspto_org_path, topdown=False):
        if '.ipynb_checkpoints' in root:
            continue
        for fname in files:
            if fname.endswith('.xml'):
                xml_path_list.append(os.path.join(root, fname))
    xml_path_list = sorted(xml_path_list)
    fout, writer = get_writer('uspto_rxn_condition.csv', CONDITION_DICT.keys())
    corpus_fout, corpus_writer = get_writer('uspto_rxn_corpus.csv', CORPUS_DICT.keys())
    cnt = 0
    for i, path in tqdm(enumerate(xml_path_list), total=len(xml_path_list)):
        reaction_and_condition_dict, corpus_dict = read_xml2dict(path)
        reaction_and_condition_df = pd.DataFrame(reaction_and_condition_dict)
        for row in reaction_and_condition_df.itertuples():
            writer.writerow(list(row)[1:])
            fout.flush()
        corpus_df = pd.DataFrame(corpus_dict)
        for row in corpus_df.itertuples():
            corpus_writer.writerow(list(row)[1:])
            corpus_fout.flush()
        cnt += len(reaction_and_condition_df)
        if i % 100 == 0:
            print(f'step {i}: {cnt} data')
    fout.close()
    corpus_fout.close()

    for patent_id in patent_cnt:
        patent_info[patent_id]['num_rxn'] = patent_cnt[patent_id]
    with open('patent_info.json', 'w') as f:
        json.dump(patent_info, f)

    print('Done')
    
