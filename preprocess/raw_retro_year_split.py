import pandas as pd
import os
import json

with open('/Mounts/rbg-storage1/users/yujieq/textreact/preprocess/uspto_script/patent_info.json') as f:
    patent_info = json.load(f)

train_df = pd.read_csv('data/USPTO_50K/train.csv')
valid_df = pd.read_csv('data/USPTO_50K/valid.csv')
test_df = pd.read_csv('data/USPTO_50K/test.csv')
train_df_proc = pd.read_csv('data/USPTO_50K/matched1/train.csv')
valid_df_proc = pd.read_csv('data/USPTO_50K/matched1/valid.csv')
test_df_proc = pd.read_csv('data/USPTO_50K/matched1/test.csv')

df = pd.concat([train_df, valid_df, test_df]).reindex()
df_proc = pd.concat([train_df_proc, valid_df_proc, test_df_proc]).reindex()

train_idx = []
valid_idx = []
test_idx = []
train_bad = 0
valid_bad = 0
test_bad = 0
unk = 0
for i, (proc_rxn_id, source, rxn_id) in enumerate(zip(df_proc['id'], df_proc['source'], df['id'])):
    p = proc_rxn_id.split('_')[0]
    if p != 'unk':
        bad = source != rxn_id
        if p in patent_info:
            year = patent_info[p]['year']
        else:
            year = -1
        if year < 2012:
            train_idx.append(i)
            train_bad += int(bad)
        elif year in [2012, 2013]:
            valid_idx.append(i)
            valid_bad += int(bad)
        else:
            test_idx.append(i)
            test_bad += int(bad)
    else:
        unk += 1
print(train_bad, valid_bad, test_bad)
print(unk)

os.makedirs('data_USPTO_50K_year_raw_alt/', exist_ok=True)
train_df = df.iloc[train_idx].reindex()
train_df.to_csv('data_USPTO_50K_year_raw_alt/train.csv', index=False)
valid_df = df.iloc[valid_idx].reindex()
valid_df.to_csv('data_USPTO_50K_year_raw_alt/valid.csv', index=False)
test_df = df.iloc[test_idx].reindex()
test_df.to_csv('data_USPTO_50K_year_raw_alt/test.csv', index=False)
