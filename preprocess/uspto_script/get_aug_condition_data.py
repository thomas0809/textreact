from multiprocessing import Pool
import os
import pandas as pd
import sys
sys.path.append('../pretrain_data_script/')
from get_pretrain_dataset_augmentation import get_random_rxn


'''
用于USPTO-Condition数据集的数据增强，在本脚本中我们只增强训练集'''
def generate_one_rxn_condition_aug(tuple_data):
    i, len_all, source, rxn, c1, s1, s2, r1, r2, dataset_flag = tuple_data
    if i > 0 and i % 1000 == 0:
        print(f"Processing {i}th Reaction / {len_all}")
    results = [(source, rxn, c1, s1, s2, r1, r2, dataset_flag)]
    for i in range(N-1):
        random_rxn = get_random_rxn(rxn)
        results.append((source, random_rxn, c1, s1, s2, r1, r2, dataset_flag))
    return results

if __name__ == '__main__':
    debug = False
    
    N = 5
    num_workers = 10
    
    
    
    source_data_path = '../../dataset/source_dataset/'
    final_condition_data_path = os.path.join(source_data_path, 'USPTO_condition_final')
    database_fname = 'USPTO_condition.csv'
    
    
    uspto_condition_dataset = pd.read_csv(os.path.join(final_condition_data_path, database_fname))
    
    
    uspto_condition_train_df = uspto_condition_dataset.loc[uspto_condition_dataset['dataset']=='train']
    uspto_condition_val_test_df = uspto_condition_dataset.loc[uspto_condition_dataset['dataset']!='train']
    if debug:
        uspto_condition_train_df = uspto_condition_train_df.sample(3000)
    
    p = Pool(num_workers)
    all_len = len(uspto_condition_train_df)
    
    augmentation_rxn_condition_train_data = p.imap(
        generate_one_rxn_condition_aug,
        ((i, all_len, *row.tolist()) for i, (index, row) in enumerate(uspto_condition_train_df.iterrows()))
    )
    
    p.close()
    p.join()    
    augmentation_rxn_condition_train_data = list(augmentation_rxn_condition_train_data) 
    augmentation_rxn_condition_train = []                                         # 一个标准rxn smiles N-1个random smiles
    for one in augmentation_rxn_condition_train_data:
        augmentation_rxn_condition_train += one
    
    aug_train_df = pd.DataFrame(augmentation_rxn_condition_train)
    aug_train_df.columns = uspto_condition_train_df.columns.tolist()
    
    aug_uspto_conditon_dataset = aug_train_df.append(uspto_condition_val_test_df).reset_index(drop=True)
    aug_uspto_conditon_dataset.to_csv(os.path.join(final_condition_data_path, 'USPTO_condition_aug_n5.csv'), index=False)