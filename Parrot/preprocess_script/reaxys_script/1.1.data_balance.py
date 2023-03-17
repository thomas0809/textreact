'''
首先要获取反应类别（用精度为97.8%的反应分类器预测），
'''
from collections import defaultdict
import os
import random
import pandas as pd
from tqdm import tqdm




if __name__ == '__main__':
    # 载入已经分配好反应类别的数据集
    
    seed = 123
    random.seed(seed)
    split_frac = (0.8, 0.1, 0.1)
    
    source_data_path = '../../dataset/source_dataset/'
    final_condition_data_path = os.path.join(
    source_data_path, 'Reaxys_total_syn_condition_final')
    
    reaxys_raw_condition_data_with_category = pd.read_csv(os.path.join(final_condition_data_path, 'Reaxys_total_syn_condition_pred_category.csv'))
    reaxys_raw_condition_data_with_category['rxn_class'] = reaxys_raw_condition_data_with_category.rxn_category.apply(lambda x:str(x).split('.')[0])
    
    w_catalyst_reaxys_raw_condition_data_with_category = reaxys_raw_condition_data_with_category.loc[~pd.isna(reaxys_raw_condition_data_with_category['catalyst1'])].reset_index(drop=True)
    print('# {} w catalyst'.format(w_catalyst_reaxys_raw_condition_data_with_category.shape[0]))
    wo_catalyst_reaxys_raw_condition_data_with_category = reaxys_raw_condition_data_with_category.loc[pd.isna(reaxys_raw_condition_data_with_category['catalyst1'])].reset_index(drop=True)
    print('# {} w/o catalyst'.format(wo_catalyst_reaxys_raw_condition_data_with_category.shape[0]))
    
    
    # 对没有催化剂的反应数据分层抽样，使得有催化剂的数据大约占15%左右(与USPTO数据集比例接近)
    
    need_wo_catalyst_data_number = int(w_catalyst_reaxys_raw_condition_data_with_category.shape[0]/0.15)
    select_frac = need_wo_catalyst_data_number / wo_catalyst_reaxys_raw_condition_data_with_category.shape[0]
    
    all_rxn_class = list(set(wo_catalyst_reaxys_raw_condition_data_with_category['rxn_class']))
    all_rxn_class.sort(key=lambda x:int(x))
    
    selected_dataset_df = w_catalyst_reaxys_raw_condition_data_with_category
    
    for rxn_class in all_rxn_class:
        sub_class_df = wo_catalyst_reaxys_raw_condition_data_with_category.loc[wo_catalyst_reaxys_raw_condition_data_with_category['rxn_class']==rxn_class].sample(frac=select_frac, random_state=123)
        selected_dataset_df = selected_dataset_df.append(sub_class_df) 
    
    selected_dataset_df = selected_dataset_df.sample(frac=1, random_state=123).reset_index(drop=True)
    print(selected_dataset_df.head())
    
    statistics_col = [
        'catalyst1',
        'solvent1',
        'solvent2',
        'reagent1',
        'reagent2',
    ]
    for condition_item in statistics_col:
        print('{} : not nan #{} nan, {} %'.format(condition_item, selected_dataset_df.loc[~pd.isna(selected_dataset_df[condition_item])].shape[0], 100*selected_dataset_df.loc[pd.isna(selected_dataset_df[condition_item])].shape[0]/selected_dataset_df.shape[0]))
        
    reaxys_raw_condition_data_sample = selected_dataset_df
    can_rxn2idx_dict = defaultdict(list)
    for idx, row in tqdm(reaxys_raw_condition_data_sample.iterrows(), total=len(reaxys_raw_condition_data_sample)):
        can_rxn2idx_dict[row.canonical_rxn].append(idx)
    train_idx, val_idx, test_idx = [], [], []
    can_rxn2idx_dict_items = list(can_rxn2idx_dict.items())
    random.shuffle(can_rxn2idx_dict_items)
    all_data_number = len(reaxys_raw_condition_data_sample)
    for rxn, idx_list in tqdm(can_rxn2idx_dict_items):
        if len(idx_list) == 1:
            if len(test_idx) < split_frac[2] * all_data_number:
                test_idx += idx_list
            elif len(val_idx) < split_frac[1] * all_data_number:
                val_idx += idx_list
            else:
                train_idx += idx_list
        else:
            train_idx += idx_list

    reaxys_raw_condition_data_sample.loc[train_idx, 'dataset'] = 'train'
    reaxys_raw_condition_data_sample.loc[val_idx, 'dataset'] = 'val'
    reaxys_raw_condition_data_sample.loc[test_idx, 'dataset'] = 'test'
    
    
    
    reaxys_raw_condition_data_sample.to_csv(os.path.join(
        final_condition_data_path, 'Reaxys_total_syn_condition.csv'), index=False)
    for dataset in ['train', 'val', 'test']:
        print(dataset, ':', (reaxys_raw_condition_data_sample['dataset']==dataset).sum())
    
    reaxys_raw_condition_data_with_category.to_csv(os.path.join(final_condition_data_path, 'Reaxys_total_syn_condition_pred_category_full.csv'), index=False)
    reaxys_raw_condition_data_sample.to_csv(os.path.join(
        final_condition_data_path, 'Reaxys_total_syn_condition_pred_category.csv'), index=False)
    '''
    catalyst1 : not nan 5592 nan, 86.95621748967834 %
    solvent1 : not nan 38714 nan, 9.696531454829605 %
    solvent2 : not nan 10549 nan, 75.39362272865107 %
    reagent1 : not nan 39469 nan, 7.935434209605561 %
    reagent2 : not nan 13238 nan, 69.12131744069417 %
    '''