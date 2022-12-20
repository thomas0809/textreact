from collections import defaultdict
import os
import random
import pandas as pd
from tqdm import tqdm
import numpy as np


def average_temperature(temp_range):
    t_list = temp_range.split(' - ')
    t_list = [float(x) for x in t_list]
    t = float(np.array(t_list).mean())
    return t


def get_exactly_temperature(t):
    try:
        t = float(t)
        return t
    except:
        if '; ' in t:
            t_group = t.split('; ')
            t_list = []
            t_mark = []
            for _t in t_group:
                try:
                    _t = float(_t)
                    t_list.append(_t)
                    t_mark.append(True)
                except:
                    t_list.append(average_temperature(_t))
                    t_mark.append(False)
            t_mark = np.array(t_mark)
            if t_mark.sum() == 0:
                return t_list[0]
            else:
                __t = np.array(t_list)[t_mark][0]
                return __t

        else:
            return average_temperature(t)



if __name__ == '__main__':
    seed = 123
    random.seed(seed)
    split_frac = (0.8, 0.1, 0.1)
    del_cnt = 10   # If the number of occurrences of the condition is less than this item, the entire data is deleted





    source_data_path = '../../dataset/source_dataset/'
    final_condition_data_path = os.path.join(
        source_data_path, 'Reaxys_total_syn_condition_final')
    if not os.path.exists(final_condition_data_path):
        os.makedirs(final_condition_data_path)
    reaxys_raw_data = pd.read_csv(os.path.join(
        source_data_path, 'reaxys_total_syn_merge_Preparation_clean_up_mapped.csv'))
    print(len(reaxys_raw_data))
    condition_select_name = [
        'Reaction ID',
        'canonical_rxn_smiles',
        'mapped_rxn_smiles',
        'Catalyst',
        'Solvent (Reaction Details)',
        'Reagent',
        'Temperature (Reaction Details) [C]',
    ]
    reaxys_raw_condition_data = reaxys_raw_data[condition_select_name]
    print(len(reaxys_raw_condition_data))
    
    other_reaxys_raw_condition_data = pd.read_csv(os.path.join(
        source_data_path, 'reaxys_yield_difference20_can_condition.csv'))
    

    
    other_reaxys_raw_condition_data = other_reaxys_raw_condition_data[
        [
        'Reaction ID',
        'can_rxn_smiles',
        'mapped_rxn_smiles',
        'new_catalyst',
        'new_solvent',
        'new_reagent',
        'new_temperature',
        ]
  
                                                                      ]
    other_reaxys_raw_condition_data = other_reaxys_raw_condition_data.loc[~(other_reaxys_raw_condition_data['new_temperature'].str.contains("日")==True)]
        
    other_reaxys_raw_condition_data.columns = ['Reaction ID', 'canonical_rxn_smiles', 'mapped_rxn_smiles', 'Catalyst', 'Solvent (Reaction Details)', 'Reagent', 'Temperature (Reaction Details) [C]']
    
    other_reaxys_raw_condition_data = other_reaxys_raw_condition_data.dropna(
        subset=['Catalyst']).reset_index(drop=True)



    
    reaxys_raw_condition_data = reaxys_raw_condition_data.append(other_reaxys_raw_condition_data)
    print(len(reaxys_raw_condition_data))
    
    reaxys_raw_condition_data = reaxys_raw_condition_data.dropna(
        subset=['Temperature (Reaction Details) [C]']).reset_index(drop=True)

    # In the case of multiple temperatures, the exact temperature is kept in preference, if there is no exact temperature, the temperature range is averaged, and if there are multiple exact values, the first temperature is selected.
    temperatures = reaxys_raw_condition_data['Temperature (Reaction Details) [C]'].tolist()
    new_temperatures = []
    for t in tqdm(temperatures):
        new_temperatures.append(get_exactly_temperature(t))
    assert len(new_temperatures) == len(temperatures)
    reaxys_raw_condition_data['Exactly Temperature (Reaction Details) [C]'] = new_temperatures

    reaxys_raw_condition_data = reaxys_raw_condition_data.drop_duplicates(subset=[
        'canonical_rxn_smiles',
        'mapped_rxn_smiles',
        'Catalyst',
        'Solvent (Reaction Details)',
        'Reagent',
        'Exactly Temperature (Reaction Details) [C]',
    ], keep='first').reset_index(drop=True)
    print(len(reaxys_raw_condition_data))

    print('Exceeding one catalyst, two solvents, or two reagents --> remove')
    remove_index_for_excess = pd.isna(reaxys_raw_condition_data.index)
    catalyst_list = []
    solvent_list = []
    reagent_list = []
    for idx, catalyst, solvent, reagent in tqdm(
            zip(
                reaxys_raw_condition_data.index.tolist(),
                reaxys_raw_condition_data['Catalyst'].tolist(),
                reaxys_raw_condition_data['Solvent (Reaction Details)'].tolist(
                ),
                reaxys_raw_condition_data['Reagent'].tolist(),
            ), total=len(reaxys_raw_condition_data)):
        if pd.isna(catalyst):
            catalyst = ''
        if pd.isna(solvent):
            solvent = ''
        if pd.isna(reagent):
            reagent = ''

        new_catalyst = list(set(catalyst.split('; ')))
        num_c = len(new_catalyst)
        new_catalyst = '; '.join(new_catalyst)
        catalyst_list.append(new_catalyst)

        new_solvent = list(set(solvent.split('; ')))
        num_s = len(new_solvent)
        new_solvent = '; '.join(new_solvent)
        solvent_list.append(new_solvent)

        new_reagent = list(set(reagent.split('; ')))
        num_r = len(new_reagent)
        new_reagent = '; '.join(new_reagent)
        reagent_list.append(new_reagent)

        if (num_c > 1) or (num_s > 2) or (num_r > 2):
            remove_index_for_excess[idx] = True
    reaxys_raw_condition_data['catalyst'] = catalyst_list
    reaxys_raw_condition_data['solvent'] = solvent_list
    reaxys_raw_condition_data['reagent'] = reagent_list
    reaxys_raw_condition_data = reaxys_raw_condition_data.loc[~remove_index_for_excess].reset_index(
        drop=True)
    print(len(reaxys_raw_condition_data))

    split_solvent = reaxys_raw_condition_data['Solvent (Reaction Details)'].str.split(
        '; ', 1, expand=True)
    reaxys_raw_condition_data['solvent1'], reaxys_raw_condition_data['solvent2'] = split_solvent[0], split_solvent[1]
    split_reagent = reaxys_raw_condition_data['Reagent'].str.split(
        '; ', 1, expand=True)
    reaxys_raw_condition_data['reagent1'], reaxys_raw_condition_data['reagent2'] = split_reagent[0], split_reagent[1]

    reaxys_raw_condition_data = reaxys_raw_condition_data[[
        'Reaction ID',
        'canonical_rxn_smiles',
        'mapped_rxn_smiles',
        'Catalyst',
        'solvent1',
        'solvent2',
        'reagent1',
        'reagent2',
        'Exactly Temperature (Reaction Details) [C]',
    ]]

    reaxys_raw_condition_data.columns = [
        'ReaxysRXNID',
        'canonical_rxn',
        'mapped_rxn',
        'catalyst1',
        'solvent1',
        'solvent2',
        'reagent1',
        'reagent2',
        'temperature',
    ]
    reaxys_raw_condition_data.dropna(
        subset=['temperature']).reset_index(drop=True)

    # Count the frequency of occurrence of each type of condition and delete data sets that contain conditions that occur very infrequently
    statistics_col = [
        'catalyst1',
        'solvent1',
        'solvent2',
        'reagent1',
        'reagent2',
    ]

    statistics_name_dict = {}
    print(
        f'Count the frequency and delete the data group less than {del_cnt}...')
    for name in tqdm(statistics_col):
        curr_name_dict = defaultdict(int)
        for cdn in reaxys_raw_condition_data[name].tolist():
            curr_name_dict[cdn] += 1
        statistics_name_dict[name] = curr_name_dict
    cdn_name2statistics_name = {}
    for name in statistics_col:
        cdn_name2statistics_name[name] = name.replace('1', '').replace('2', '')
    # statistics_name2cdn_name = defaultdict(list)
    # for k, v in cdn_name2statistics_name.items():
    #     statistics_name2cdn_name[v].append(k)
    statistics_name_dict['catalyst'] = statistics_name_dict['catalyst1']
    statistics_name_dict['solvent'] = defaultdict(int)

    for name in set(list(statistics_name_dict['solvent1'].keys()) + list(statistics_name_dict['solvent2'].keys())):
        statistics_name_dict['solvent'][name] = statistics_name_dict['solvent1'][name] + statistics_name_dict['solvent2'][name]
    statistics_name_dict['reagent'] = defaultdict(int)
    for name in set(list(statistics_name_dict['reagent1'].keys()) + list(statistics_name_dict['reagent2'].keys())):
        statistics_name_dict['reagent'][name] = statistics_name_dict['reagent1'][name] + statistics_name_dict['reagent2'][name]
    
    for name in statistics_col:
        del statistics_name_dict[name]
    
    del_dict = {}
    need_dict = {}
    
    for name in statistics_name_dict:
        del_dict[name] = [x[0] for x in list(statistics_name_dict[name].items()) if x[1] < del_cnt]
        need_dict[name] = [x[0] for x in list(statistics_name_dict[name].items()) if x[1] >= del_cnt]
        print('{} kinds of {} are reserved and {} kinds are deleted'.format(len(need_dict[name]), name, len(del_dict[name])))
    
    del_index = pd.isna(reaxys_raw_condition_data.index)
    for cdn_name in statistics_col:
        del_index = del_index | reaxys_raw_condition_data[cdn_name].isin(del_dict[cdn_name2statistics_name[cdn_name]])
    reaxys_raw_condition_data = reaxys_raw_condition_data.loc[~del_index].reset_index(
        drop=True)
    print('Reaxys condition data number:',len(reaxys_raw_condition_data))
    print('Split train:val:test={}:{}:{}'.format(split_frac[0], split_frac[1], split_frac[2]))

    # split dataset   train:val:test = 8:1:1
    reaxys_raw_condition_data_sample = reaxys_raw_condition_data.sample(
        frac=1, random_state=seed)
    
    for condition_item in statistics_col:
        print('{} : not nan #{} nan, {} %'.format(condition_item, reaxys_raw_condition_data_sample.loc[~pd.isna(reaxys_raw_condition_data_sample[condition_item])].shape[0], 100*reaxys_raw_condition_data_sample.loc[pd.isna(reaxys_raw_condition_data_sample[condition_item])].shape[0]/reaxys_raw_condition_data_sample.shape[0]))
    
    print('{} : not nan #{} nan, {} %'.format('temperature', reaxys_raw_condition_data_sample.loc[~pd.isna(reaxys_raw_condition_data_sample['temperature'])].shape[0], 100*reaxys_raw_condition_data_sample.loc[pd.isna(reaxys_raw_condition_data_sample['temperature'])].shape[0]/reaxys_raw_condition_data_sample.shape[0]))
        
        
    # can_rxn2idx_dict = defaultdict(list)
    # for idx, row in tqdm(reaxys_raw_condition_data_sample.iterrows(), total=len(reaxys_raw_condition_data_sample)):
    #     can_rxn2idx_dict[row.canonical_rxn].append(idx)
    # train_idx, val_idx, test_idx = [], [], []
    # can_rxn2idx_dict_items = list(can_rxn2idx_dict.items())
    # random.shuffle(can_rxn2idx_dict_items)
    # all_data_number = len(reaxys_raw_condition_data_sample)
    # for rxn, idx_list in tqdm(can_rxn2idx_dict_items):
    #     if len(idx_list) == 1:
    #         if len(test_idx) < split_frac[2] * all_data_number:
    #             test_idx += idx_list
    #         elif len(val_idx) < split_frac[1] * all_data_number:
    #             val_idx += idx_list
    #         else:
    #             train_idx += idx_list
    #     else:
    #         train_idx += idx_list

    # reaxys_raw_condition_data_sample.loc[train_idx, 'dataset'] = 'train'
    # reaxys_raw_condition_data_sample.loc[val_idx, 'dataset'] = 'val'
    # reaxys_raw_condition_data_sample.loc[test_idx, 'dataset'] = 'test'
    reaxys_raw_condition_data_sample.to_csv(os.path.join(
        final_condition_data_path, 'Reaxys_total_syn_condition.csv'), index=False)
    # for dataset in ['train', 'val', 'test']:
    #     print(dataset, ':', (reaxys_raw_condition_data_sample['dataset']==dataset).sum())
