import os
import json
import pandas as pd

# Dedup

# corpus_df = pd.read_csv('uspto_script/uspto_rxn_corpus.csv')
# text_to_corpus_id = {}
# id_to_corpus_id = {}
# dedup_flags = [False] * len(corpus_df)
# for i, (idx, text) in enumerate(zip(corpus_df['id'], corpus_df['paragraph_text'])):
#     if text not in text_to_corpus_id:
#         text_to_corpus_id[text] = idx
#         dedup_flags[i] = True
#     id_to_corpus_id[idx] = text_to_corpus_id[text]
#
# dedup_df = corpus_df[dedup_flags]
# dedup_df.to_csv('uspto_script/uspto_rxn_corpus_dedup.csv', index=False)
#
# with open('id_to_corpus_id.json', 'w') as f:
#     json.dump(id_to_corpus_id, f)


# Add corpus id
with open('id_to_corpus_id.json') as f:
    id_to_corpus_id = json.load(f)

# data_path = '../data/USPTO_condition/'
# data_path = '../data/USPTO_condition_year/'
# for file in ['USPTO_condition_train.csv', 'USPTO_condition_val.csv', 'USPTO_condition_test.csv']:

# data_path = '../data/USPTO_50K/matched1/'
data_path = '../data/USPTO_50K_year/'
for file in ['train.csv', 'valid.csv', 'test.csv']:
    path = os.path.join(data_path, file)
    print(path)
    df = pd.read_csv(path)
    corpus_id = [id_to_corpus_id.get(idx, idx) for idx in df['id']]
    df['corpus_id'] = corpus_id
    cols = ['id', 'corpus_id']
    for col in df.columns:
        if col not in cols:
            cols.append(col)
    df = df[cols]
    df.to_csv(path, index=False)
