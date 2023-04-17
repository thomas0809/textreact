import os
import json
import logging
import multiprocessing
import numpy as np
import pandas as pd
from tqdm import tqdm

import rdkit
from rdkit import Chem
rdkit.RDLogger.DisableLog('rdApp.*')
import rdkit.Chem.rdChemReactions as rdChemReactions
import rdkit.DataStructs as DataStructs


BASE = 'data/USPTO_50K'


def canonical_rxn_smiles(rxn_smiles):
    try:
        reactants, reagents, products = rxn_smiles.split(">")
        mols_r = Chem.MolFromSmiles(reactants)
        mols_p = Chem.MolFromSmiles(products)
        [a.ClearProp('molAtomMapNumber') for a in mols_r.GetAtoms()]
        [a.ClearProp('molAtomMapNumber') for a in mols_p.GetAtoms()]
        cano_smi_r = Chem.MolToSmiles(mols_r, isomericSmiles=True, canonical=True)
        cano_smi_p = Chem.MolToSmiles(mols_p, isomericSmiles=True, canonical=True)
        return cano_smi_r + '>>' + cano_smi_p, True
    except:
        return rxn_smiles, False


def reaction_fingerprint(smiles):
    rxn = rdChemReactions.ReactionFromSmarts(smiles)
    fp = rdChemReactions.CreateDifferenceFingerprintForReaction(rxn)
    return fp


def reaction_similarity(smiles1=None, smiles2=None, fp1=None, fp2=None):
    if fp1 is None and smiles1:
        fp1 = reaction_fingerprint(smiles1)
    if fp2 is None and smiles2:
        fp2 = reaction_fingerprint(smiles2)
    assert fp1 is not None
    assert fp2 is not None
    return DataStructs.TanimotoSimilarity(fp1, fp2)


# Canonical SMILES

# for split in ['train', 'valid', 'test']:
#     df = pd.read_csv(os.path.join(BASE, f'{split}.csv'))
#     invalid_count = 0
#     for i, row in df.iterrows():
#         try:
#             reactants, reagents, products = row["rxn_smiles"].split(">")
#             mols_r = Chem.MolFromSmiles(reactants)
#             mols_p = Chem.MolFromSmiles(products)
#             if mols_r is None or mols_p is None:
#                 invalid_count += 1
#                 continue

#             [a.ClearProp('molAtomMapNumber') for a in mols_r.GetAtoms()]
#             [a.ClearProp('molAtomMapNumber') for a in mols_p.GetAtoms()]

#             cano_smi_r = Chem.MolToSmiles(mols_r, isomericSmiles=True, canonical=True)
#             cano_smi_p = Chem.MolToSmiles(mols_p, isomericSmiles=True, canonical=True)

#             df.loc[i, 'reactant_smiles'] = cano_smi_r
#             df.loc[i, 'product_smiles'] = cano_smi_p
#         except Exception as e:
#             logging.info(e)
#             logging.info(row["rxn_smiles"].split(">"))
#             invalid_count += 1

#     logging.info(f"Invalid count: {invalid_count}")
#     df.drop(columns='rxn_smiles', inplace=True)
#     df.to_csv(os.path.join(BASE, f'processed/{split}.csv'), index=False)


# Match id

# corpus_df = pd.read_csv('data/USPTO_rxn_condition.csv')
# with multiprocessing.Pool(32) as p:
#     results = p.map(canonical_rxn_smiles, corpus_df['rxn_smiles'], chunksize=128)
#     canonical_rxn, success = zip(*results)
#
# print(np.mean(success))
# corpus_df['canonical_rxn'] = canonical_rxn
# corpus_df.to_csv('data/USPTO_rxn_condition_canon.csv', index=False)

corpus_df = pd.read_csv('preprocess/uspto_script/uspto_rxn_condition_remapped_and_reassign_condition_role.csv')
rxn_smiles_to_id = {}
for i, row in tqdm(corpus_df.iterrows()):
    rxn_smiles_to_id[row['canonical_rxn']] = row['id']

# fulldata_df = pd.read_csv('preprocess/uspto_script/uspto_rxn_condition.csv')

for split in ['train', 'valid', 'test']:
    df = pd.read_csv(f'data/USPTO_50K/processed/{split}.csv')
    cnt = 0
    nomatch_cnt = 0
    matched_ids = []
    f = open('tmp.txt', 'w')
    for i, row in tqdm(df.iterrows()):
        rxn_smiles = row['reactant_smiles'] + '>>' + row['product_smiles']
        if rxn_smiles in rxn_smiles_to_id:
            rxn_id = rxn_smiles_to_id[rxn_smiles]
            cnt += 1
        else:
            patent_df = corpus_df.loc[corpus_df['source'] == row['id']]
            f.write(row['id'] + '\n')
            rxn_id = f'unk_{split}_{i}'
            if len(patent_df) == 0:
                nomatch_cnt += 1
                f.write('No match\n\n')
            else:
                fp = reaction_fingerprint(rxn_smiles)
                # patent_rxn_smiles = [canonical_rxn_smiles(smiles)[0] for smiles in patent_df['rxn_smiles']]
                patent_rxn_smiles = patent_df['canonical_rxn'].tolist()
                similarities = [reaction_similarity(fp1=fp, smiles2=smiles) for smiles in patent_rxn_smiles]
                nearest_idx = np.argmax(similarities)
                nearest_row = patent_df.iloc[nearest_idx]
                if similarities[nearest_idx] > 0.9:
                    rxn_id = nearest_row['id']
                    cnt += 1
                f.write(rxn_smiles + '\n')
                f.write(patent_rxn_smiles[nearest_idx] + '\n')
                f.write(json.dumps(similarities) + '\n')
                f.write(f'{similarities[nearest_idx]}\n')
                f.write('\n')
            f.flush()
        matched_ids.append(rxn_id)
    f.close()
    df['matched_id'] = matched_ids
    df.to_csv(f'data/USPTO_50K/matched/{split}.csv', index=False)
    print(cnt, nomatch_cnt, len(df))
