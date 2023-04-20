import json
import pandas as pd


corpus_df = pd.read_csv('uspto_rxn_corpus.csv')

grant_df = corpus_df[corpus_df['patent_type'] == 'grant']
grant_df.to_csv('USPTO_rxn_grant_corpus.csv', index=False)
