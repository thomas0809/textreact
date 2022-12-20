import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from makeReactionFromParsed import canonicalize_smiles


def split_series(df, split_col_name='solvents', split_number_col=2, split_mark='[SPLIT]', remove_redundant_line=True):
    target_series = df[split_col_name]
    selected_line_ids = np.array([True]*len(df))
    new_series_data = []
    for i, x in enumerate(target_series):
        if pd.isna(x):
            x_split = ['']
        else:
            x_split = x.split(split_mark)
        if len(x_split) > split_number_col:
            selected_line_ids[i] = False
            x_split = x_split[:split_number_col]
        elif len(x_split) < split_number_col:
            x_split = x_split + ['']*(split_number_col-len(x_split))
        new_series_data.append(x_split)

    splited_data = zip(*new_series_data)
    for i, data in enumerate(splited_data):
        df['{}{}'.format(split_col_name, i+1)] = list(data)
    if remove_redundant_line:
        df = df.loc[selected_line_ids]
    return df


def merge_rxn(df):

    rxn_smiles = []

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        reacts = row[['Reactant1', 'Reactant2']]
        reacts = reacts[~pd.isna(reacts)]
        prod = row['Product']

        reacts = canonicalize_smiles('.'.join(reacts.tolist()))
        prod = canonicalize_smiles(prod)

        can_rxn = f"{reacts}>>{prod}"

        rxn_smiles.append(can_rxn)

    df.loc[:, 'canonical_rxn'] = rxn_smiles
    return df


if __name__ == '__main__':
    SEED = 123
    split_frac = (0.6, 0.2, 0.2)
    uspto_suzuki_path = '../../dataset/source_dataset/USPTO_suzuki_final'
    raw_data_name = 'USPTO_suzuki_dataset.csv'
    
    uspto_condition_path = '../../dataset/source_dataset/USPTO_condition_final/USPTO_condition.csv'
    ref_dataset_df = pd.read_csv(uspto_condition_path)
    ref_dataset_df = ref_dataset_df[ref_dataset_df['dataset']=='train']
    ref_dataset_rxn = ref_dataset_df['canonical_rxn']
    

    raw_data_df = pd.read_csv(os.path.join(
        uspto_suzuki_path, raw_data_name), sep=';')

    raw_data_df = split_series(raw_data_df, split_col_name='solvents',
                               split_number_col=2, split_mark='[SPLIT]', remove_redundant_line=True)

    raw_data_df = merge_rxn(raw_data_df)

    condition_dataset_df = raw_data_df[[
        'canonical_rxn', 'catalysts', 'ligands', 'bases', 'solvents1', 'solvents2', 'Yield']]

    condition_dataset_df.columns = [
        'canonical_rxn', 'catalyst1', 'reagent1', 'reagent2', 'solvent1', 'solvent2', 'Yield']
    condition_dataset_df = condition_dataset_df[[
        'canonical_rxn', 'catalyst1', 'solvent1', 'solvent2', 'reagent1', 'reagent2', 'Yield']]
    condition_dataset_df_shuffle = condition_dataset_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    
    train_number = int(len(condition_dataset_df_shuffle)*split_frac[0])
    val_number = int(len(condition_dataset_df_shuffle)*split_frac[1])
    test_number = int(len(condition_dataset_df_shuffle)*split_frac[2])
    
    train_ids = np.array([False]*len(condition_dataset_df_shuffle))
    val_ids = np.array([False]*len(condition_dataset_df_shuffle))
    test_ids = np.array([False]*len(condition_dataset_df_shuffle))
    
    for idx, rxn in tqdm(enumerate(condition_dataset_df_shuffle['canonical_rxn']), total=len(condition_dataset_df_shuffle)):
        if rxn not in ref_dataset_rxn.tolist():
            if test_ids.sum() <= test_number:
                test_ids[idx] = True
            elif val_ids.sum() <= val_number:
                val_ids[idx] = True
            else:
                train_ids[idx] = True
        else:
            train_ids[idx] = True
        
    print('train:', train_ids.sum())
    print('val:', val_ids.sum())
    print('test:', test_ids.sum())
    
    # split_point1 = int(len(condition_dataset_df_shuffle)*split_frac[0])
    # split_point2 = split_point1 + int(len(condition_dataset_df_shuffle)*split_frac[1])
    
    condition_dataset_df_shuffle.loc[test_ids, 'dataset'] = 'test'
    condition_dataset_df_shuffle.loc[val_ids, 'dataset'] = 'val'
    condition_dataset_df_shuffle.loc[train_ids, 'dataset'] = 'train'
    
    
    
    condition_dataset_df_shuffle.to_csv(os.path.join(uspto_suzuki_path, 'USPTO_suzuki_condition.csv'), index=False)