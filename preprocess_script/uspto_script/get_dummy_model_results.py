import os
import numpy as np
import pandas as pd
import sys

from tqdm import tqdm
sys.path = [os.path.abspath(os.path.join(os.path.abspath(__file__), '../../../'))] + sys.path
from models.utils import load_dataset
import torch

dummy_prediction = [
    [0, 0, 0, 0, 0],            # [na, na, na, na, na]
    [0, 160, 0, 0, 0],          # [na, DCM, na, na, na]
    [0, 16, 0, 0, 0],           # [na, THF, na, na, na]
    [0, 0, 0, 85, 0],           # [na, na, na, TEA, na]
    [0, 0, 0, 202, 0],          # [na, na, na, K2CO3, na]
    [0, 160, 0, 85, 0],         # [na, DCM, na, TEA, na]
    [0, 16, 0, 85, 0],          # [na, THF, na, TEA, na]
    [0, 160, 0, 202, 0],        # [na, DCM, na, K2CO3, na]
    [0, 16, 0, 202, 0],         # [na, THF, na, K2CO3, na]
    [298, 0, 0, 0, 0],          # [[Pd], na, na, na, na]
]

def _get_accuracy_for_one(one_pred, one_ground_truth, topk_get=[1, 3, 5, 10, 15]):
    repeat_number = one_pred.size(0)
    hit_mat = one_ground_truth.unsqueeze(
        0).repeat(repeat_number, 1) == one_pred
    overall_hit_mat = hit_mat.sum(1) == hit_mat.size(1)
    topk_hit_df = pd.DataFrame()
    for k in topk_get:
        hit_mat_k = hit_mat[:k, :]
        overall_hit_mat_k = overall_hit_mat[:k]
        topk_hit = []
        for col_idx in range(hit_mat.size(1)):
            if hit_mat_k[:, col_idx].sum() != 0:
                topk_hit.append(1)
            else:
                topk_hit.append(0)
        if overall_hit_mat_k.sum() != 0:
            topk_hit.append(1)
        else:
            topk_hit.append(0)
        topk_hit_df[k] = topk_hit
    # topk_hit_df.index = ['c1', 's1', 's2', 'r1', 'r2']
    return topk_hit_df

def _calculate_batch_topk_hit(batch_preds, batch_ground_truth, topk_get=[1, 3, 5, 10, 15]):
    '''
    batch_pred                         <-- tgt_tokens_list
    batch_ground_truth                 <-- inputs['labels']
    '''
    batch_preds = torch.tensor(batch_preds)[:, :, :].to(torch.device('cpu'))
    batch_ground_truth = batch_ground_truth[:, 1:-1]

    one_batch_topk_acc_mat = np.zeros((6, 5))
    # topk_get = [1, 3, 5, 10, 15]
    for idx in range(batch_preds.size(0)):
        topk_hit_df = _get_accuracy_for_one(
            batch_preds[idx], batch_ground_truth[idx], topk_get=topk_get)
        one_batch_topk_acc_mat += topk_hit_df.values
    return one_batch_topk_acc_mat

if __name__ == '__main__':
    topk_get = [1, 3, 5, 10, 15]
    source_data_path = '../../dataset/source_dataset/'
    uspto_root = os.path.abspath(os.path.join(source_data_path, 'USPTO_condition_final'))
    database_df, condition_label_mapping = load_dataset(dataset_root=uspto_root, database_fname='USPTO_condition.csv', use_temperature=False)
    
    test_df = database_df.loc[database_df['dataset']=='test']
    topk_acc_mat = np.zeros((6, 5))
    for gt in tqdm(test_df['condition_labels'].tolist()):
        one_batch_topk_acc_mat = _calculate_batch_topk_hit(batch_preds=[dummy_prediction], batch_ground_truth=torch.tensor([gt]))
        topk_acc_mat += one_batch_topk_acc_mat
    topk_acc_mat /= len(test_df['condition_labels'].tolist())
    topk_acc_df = pd.DataFrame(topk_acc_mat)
    topk_acc_df.columns = [f'top-{k} accuracy' for k in topk_get]
    topk_acc_df.index = ['c1', 's1', 's2', 'r1', 'r2', 'overall']
    print(topk_acc_df)
    
'''
         top-1 accuracy  top-3 accuracy  top-5 accuracy  top-10 accuracy  top-15 accuracy
c1             0.869600        0.869600        0.869600         0.914682         0.914682
s1             0.010180        0.309805        0.309805         0.309805         0.309805
s2             0.808549        0.808549        0.808549         0.808549         0.808549
r1             0.260859        0.260859        0.377216         0.377216         0.377216
r2             0.746515        0.746515        0.746515         0.746515         0.746515
overall        0.000059        0.043085        0.043085         0.066074         0.066074
'''