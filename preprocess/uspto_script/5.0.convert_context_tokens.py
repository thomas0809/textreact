import os
import pandas as pd
from tqdm import tqdm
import pickle

BOS, EOS, PAD, MASK = '[BOS]', '[EOS]', '[PAD]', '[MASK]'


def get_condition2idx_mapping(all_condition_data: pd.DataFrame):
    col_unique_data = [BOS, EOS, PAD, MASK]
    for col in all_condition_data.columns.tolist():
        one_col_unique = list(set(all_condition_data[col].tolist()))
        col_unique_data.extend(one_col_unique)
    col_unique_data = list(set(col_unique_data))
    col_unique_data.sort()
    idx2data = {i: x for i, x in enumerate(col_unique_data)}
    data2idx = {x: i for i, x in enumerate(col_unique_data)}
    return idx2data, data2idx


if __name__ == '__main__':
    calculate_fps = False
    convert_aug_data = True

    debug = False
    print('Debug:', debug, 'Convert data:', convert_aug_data)
    fp_size = 16384
    source_data_path = '../../dataset/source_dataset/'
    final_condition_data_path = os.path.join(source_data_path,
                                             'USPTO_condition_final')
    if convert_aug_data:
        database_fname = 'USPTO_condition_aug_n5.csv'
    else:
        database_fname = 'USPTO_condition.csv'

    if debug:
        database = pd.read_csv(os.path.join(final_condition_data_path,
                                            database_fname),
                               nrows=10000)
        final_condition_data_path = os.path.join(
            source_data_path, 'USPTO_condition_final_debug')
        if not os.path.exists(final_condition_data_path):
            os.makedirs(final_condition_data_path)
        database.to_csv(os.path.join(final_condition_data_path,
                                     database_fname),
                        index=False)
    else:
        database = pd.read_csv(
            os.path.join(final_condition_data_path, database_fname))

    condition_cols = [
        'catalyst1', 'solvent1', 'solvent2', 'reagent1', 'reagent2'
    ]

    all_idx2data, all_data2idx = get_condition2idx_mapping(
        database[condition_cols])
    all_idx_mapping_data_fpath = os.path.join(
        final_condition_data_path,
        '{}_alldata_idx.pkl'.format(database_fname.split('.')[0]))
    with open(all_idx_mapping_data_fpath, 'wb') as f:
        pickle.dump((all_idx2data, all_data2idx), f)
    all_condition_labels = []
    for _, row in tqdm(database[condition_cols].iterrows(),
                       total=len(database)):
        row = list(row)
        row = ['[BOS]'] + row + ['[EOS]']
        all_condition_labels.append([all_data2idx[x] for x in row])

    all_condition_labels_fpath = os.path.join(
        final_condition_data_path,
        '{}_condition_labels.pkl'.format(database_fname.split('.')[0]))
    with open(all_condition_labels_fpath, 'wb') as f:
        pickle.dump((all_condition_labels), f)
    print('Done!')
