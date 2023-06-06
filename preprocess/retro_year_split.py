import pandas as pd
import os
import json

with open('preprocess/uspto_script/patent_info.json') as f:
    patent_info = json.load(f)

train_df = pd.read_csv('data/USPTO_50K/matched1/train.csv')
valid_df = pd.read_csv('data/USPTO_50K/matched1/valid.csv')
test_df = pd.read_csv('data/USPTO_50K/matched1/test.csv')

df = pd.concat([train_df, valid_df, test_df]).reindex()

train_idx = []
valid_idx = []
test_idx = []
for i, rxn_id in enumerate(df['id']):
    p = rxn_id.split('_')[0]
    if p in patent_info:
        year = patent_info[p]['year']
    else:
        year = -1
    if year < 2012:
        train_idx.append(i)
    elif year in [2012, 2013]:
        valid_idx.append(i)
    else:
        test_idx.append(i)

os.makedirs('data/USPTO_50K_year/', exist_ok=True)
train_df = df.iloc[train_idx].reindex()
train_df.to_csv('data/USPTO_50K_year/train.csv', index=False)
valid_df = df.iloc[valid_idx].reindex()
valid_df.to_csv('data/USPTO_50K_year/valid.csv', index=False)
test_df = df.iloc[test_idx].reindex()
test_df.to_csv('data/USPTO_50K_year/test.csv', index=False)
