import json
import os
import pandas as pd


def mark_label_with_category(x,col):
    try:
        new_x = []
        for s in x:
            new_x.append(f'{s}$${col}')
        return new_x
    except:
        return x
    

if __name__ == '__main__':
    source_data_path = os.path.join('../../dataset/source_dataset/')
    uspto_suzuki_dataset = pd.read_csv(os.path.join(source_data_path, 'USPTO_suzuki_final', 'USPTO_suzuki_condition.csv'))
    
    print(uspto_suzuki_dataset.head())
    

    uspto_suzuki_dataset[['Reactant', 'Product']] = uspto_suzuki_dataset['canonical_rxn'].str.split('>>',expand=True)
    uspto_suzuki_dataset = uspto_suzuki_dataset.drop(['canonical_rxn'], axis=1)
    uspto_suzuki_dataset[['Reactant1', 'Reactant2']] = uspto_suzuki_dataset['Reactant'].str.split('.', expand=True)
    uspto_suzuki_dataset = uspto_suzuki_dataset.drop(['Reactant'], axis=1)
    # col_splits_2 = ['Reactant', 'Product']
    # for col in col_splits_2:
    #     uspto_suzuki_dataset[[col]] = uspto_suzuki_dataset[col].str.split('.')
    
    dataset_info = {}
    class_dict = {}
    category_dataframes = {}
    translate_dict_collect = {}
    for col in ['catalyst1', 'solvent1', 'solvent2', 'reagent1', 'reagent2']:
        # uspto_suzuki_dataset[col].isna()
        # uspto_suzuki_dataset.loc[uspto_suzuki_dataset[col].isna(), col] = f'{col}_nan'
        uspto_suzuki_dataset[[col]] = uspto_suzuki_dataset[col].str.split(';')
        uspto_suzuki_dataset[col] = uspto_suzuki_dataset[col].apply(lambda x: mark_label_with_category(x, col))
        col_all = uspto_suzuki_dataset[col].explode()
        col_na = uspto_suzuki_dataset[col].isna()
        uspto_suzuki_dataset.loc[col_na, col] = pd.Series([[]] * col_na.sum()).values
        col_value_cnt_list = list(zip(col_all.value_counts().index,
                            col_all.value_counts(),
                            col_all.value_counts(normalize=True)
                           )
                       )
        current_columns = [f'labels', 'count', 'frequency']
        col_value_cnt_df = pd.DataFrame(col_value_cnt_list, columns=current_columns)
        col_value_cnt_df['category'] = col[0] + col[-1]
        category_dataframes[f'{col}_df'] = col_value_cnt_df
        col_value_cnt_df.to_csv(os.path.join(source_data_path, 'USPTO_suzuki_final', f'USPTO_suzuki_condition_{col}.csv'))
        translate_dict = {v:k for k,v in zip(col_value_cnt_df.index.tolist(), col_value_cnt_df['labels'].tolist())}
        # uspto_suzuki_dataset[col] = uspto_suzuki_dataset[col].apply(lambda x:[translate_dict[j] for j in x])
        # with open(os.path.join(source_data_path, 'USPTO_suzuki_final', f'{col}_translation_dict.json'), 'w', encoding='utf-8') as f:
        #     json.dump(translate_dict, f)
        # translate_labels = []
        # for condition in uspto_suzuki_dataset[col].tolist():
        #     if not pd.isna(condition):
        #         translate_labels.append(translate_dict[condition])
        #     else:
        #         translate_labels.append('')
        # uspto_suzuki_dataset[col[0]+col[-1]] = translate_labels
        class_dict[col] = len(translate_dict)
        translate_dict_collect[col] = translate_dict

    all_translate_dict = {}
    cum_labels = 0
    for col in ['catalyst1', 'solvent1', 'solvent2', 'reagent1', 'reagent2']:
        col_class_dict = translate_dict_collect[col]
        for k, v in col_class_dict.items():
            all_translate_dict[k] = v + cum_labels
        cum_labels += len(col_class_dict)
    
    with open(os.path.join(source_data_path, 'USPTO_suzuki_final', 'USPTO_suzuki_condition_translate_dict.json'), 'w', encoding='utf-8') as f:
        json.dump(all_translate_dict, f)
    
    for col in ['catalyst1', 'solvent1', 'solvent2', 'reagent1', 'reagent2']:
        uspto_suzuki_dataset[col] = uspto_suzuki_dataset[col].apply(lambda x:[all_translate_dict[j] for j in x])
    
    labels = list(class_dict.keys())
    class_num = sum(list(class_dict.values()))
    
    dataset_info['labels'] = labels
    dataset_info['class_dict'] = class_dict
    dataset_info['class_num'] = class_num + 5 + 1
    
    with open(os.path.join(source_data_path, 'USPTO_suzuki_final', 'USPTO_suzuki_condition_relgat_datasetInfo.json'), 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f)
    print(uspto_suzuki_dataset.head())
    uspto_suzuki_dataset.to_csv(os.path.join(source_data_path, 'USPTO_suzuki_final', 'USPTO_suzuki_condition_relgat_translated.csv'), index=False, sep=';')
    
    
    
    
    