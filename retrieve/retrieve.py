import pandas as pd
import numpy as np
import json
import rdkit
import rdkit.Chem as Chem
import rdkit.Chem.rdChemReactions as rdChemReactions
import rdkit.DataStructs as DataStructs
from tqdm import tqdm
import multiprocessing


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


def reaction_similarity_fp_smiles(fp, smiles):
    return reaction_similarity(fp1=fp, smiles2=smiles)


def compute_reaction_similarities(test_smiles, train_smiles_list, num_workers=64):
    test_fp = reaction_fingerprint(test_smiles)
    with multiprocessing.Pool(num_workers) as p:
        similarities = p.starmap(
            reaction_similarity_fp_smiles,
            [(test_fp, smiles) for smiles in train_smiles_list],
            chunksize=128
        )
    return similarities


if __name__ == '__main__':

    train_df = pd.read_csv('data/USPTO_condition_train.csv')
    val_df = pd.read_csv('data/USPTO_condition_val.csv')
    test_df = pd.read_csv('data/USPTO_condition_test.csv')

    train_smiles_list = train_df['canonical_rxn']

    results = {}
    # with open('test_nn.json') as f:
    #     results = json.load(f)

    for i, test_row in tqdm(test_df.iterrows()):
        if str(i) in results:
            continue
        test_smiles = test_row['canonical_rxn']
        similarities = compute_reaction_similarities(test_smiles, train_smiles_list, num_workers=64)
        ranks = np.argsort(similarities)[::-1][:100].tolist()
        results[i] = {
            'rank': ranks,
            'similarity': [similarities[j] for j in ranks]
        }
        if i + 1 == 100:
            break

    with open('test_nn.json', 'w') as f:
        json.dump(results, f)
