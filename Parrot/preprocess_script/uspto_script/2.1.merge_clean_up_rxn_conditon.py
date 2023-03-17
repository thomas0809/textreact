from collections import defaultdict
import os
import pandas as pd
from tqdm import tqdm

from utils import calculate_frequency, get_writer

def write_freq(fpath, freq_data):
    fout, writer = get_writer(fpath, ['smiles', 'freq_cnt'])
    for data in freq_data:
        writer.writerow(data)
        fout.flush()
    fout.close()

if __name__ == '__main__':
    debug = False

    source_data_path = '../../dataset/source_dataset/'
    merge_data_fname = 'uspto_rxn_condition_remapped_and_reassign_condition_role.csv'
    duplicate_removal_fname = 'uspto_rxn_condition_remapped_and_reassign_condition_role_rm_duplicate.csv'
    freq_info_path = os.path.join(source_data_path, 'freq_info')
    if not os.path.exists(freq_info_path):
        os.makedirs(freq_info_path)
    if not os.path.exists(os.path.join(source_data_path, merge_data_fname)):
        database = pd.DataFrame()
        for group_idx in [0, 1, 2, 3]:
            database = database.append(pd.read_csv(os.path.join(
                source_data_path, f'uspto_rxn_condition_remapped_and_reassign_condition_role_group_{group_idx}.csv')))
        database.reset_index(inplace=True, drop=True)
        database.to_csv(os.path.join(os.path.join(
            source_data_path, merge_data_fname)), index=False)
    else:
        if not debug:
            database = pd.read_csv(os.path.join(
                source_data_path, merge_data_fname))
        else:
            database = pd.read_csv(os.path.join(
                source_data_path, merge_data_fname), nrows=10000)

    #  按照 remapped_rxn + canonical_rxn + catalyst + solvent + reagent 比照标准去除重复， source, org_rxn, fragment, confidence 这几列取类别中的第一个
    info_row_name = ['remapped_rxn', 'canonical_rxn', 'catalyst', 'solvent', 'reagent']
    database_duplicate_removal = database.drop_duplicates(subset=info_row_name, keep='first')
    database_duplicate_removal.reset_index(inplace=True, drop=True)
    print()
    print('catalyst count:', len(set(database_duplicate_removal['catalyst'])))
    catalyst_freq = calculate_frequency(database_duplicate_removal['catalyst'].tolist())
    write_freq(os.path.join(freq_info_path, 'catalyst_freq.csv'), catalyst_freq)
    print()
    print('solvent count:', len(set(database_duplicate_removal['solvent'])))
    solvent_freq = calculate_frequency(database_duplicate_removal['solvent'].tolist())
    write_freq(os.path.join(freq_info_path, 'solvent_freq.csv'), solvent_freq)
    print()
    print('reagent count:', len(set(database_duplicate_removal['reagent'])))
    reagent_freq = calculate_frequency(database_duplicate_removal['reagent'].tolist())
    write_freq(os.path.join(freq_info_path, 'reagent_freq.csv'), reagent_freq)
    print()

    print('All dataset count:', len(database_duplicate_removal))
    database_duplicate_removal.to_csv(os.path.join(source_data_path, duplicate_removal_fname), index=False)
    
