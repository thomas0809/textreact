import os
import argparse
import pandas as pd
import numpy as np
import json
import rdkit
import rdkit.Chem as Chem
rdkit.RDLogger.DisableLog('rdApp.*')
import rdkit.Chem.AllChem as AllChem
import rdkit.Chem.rdChemReactions as rdChemReactions
import rdkit.DataStructs as DataStructs
from tqdm import tqdm
import multiprocessing
import faiss
import time


def reaction_fingerprint(smiles):
    rxn = rdChemReactions.ReactionFromSmarts(smiles)
    fp = rdChemReactions.CreateDifferenceFingerprintForReaction(rxn)
    return fp


def reaction_fingerprint_array(smiles):
    fp = reaction_fingerprint(smiles)
    array = np.array([x for x in fp])
    return array


def compute_reaction_fingerprints(smiles_list, num_workers=64):
    with multiprocessing.Pool(num_workers) as p:
        fps = p.map(reaction_fingerprint_array, smiles_list, chunksize=128)
    return np.array(fps)


def morgan_fingerprint(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        array = np.zeros((0,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, array)
    except:
        return morgan_fingerprint('C')
    return array


def compute_molecule_fingerprints(smiles_list, num_workers=64):
    with multiprocessing.Pool(num_workers) as p:
        fps = p.map(morgan_fingerprint, smiles_list, chunksize=64)
    return np.array(fps)


def compare_condition(row1, row2):
    for field in ['catalyst1', 'solvent1', 'solvent2', 'reagent1', 'reagent2']:
        if type(row1[field]) is not str and type(row2[field]) is not str:
            continue
        if row1[field] != row2[field]:
            return False
    return True


def index_and_search(train_fps, query_fps):
    print('Faiss build index')
    d = train_fps.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(train_fps)

    print('Faiss nearest neighbor search')
    start = time.time()
    k = 20
    distance, rank = index.search(query_fps, k)
    end = time.time()
    print(f"{end - start:.2f} s")
    return rank


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=None, required=True)
    parser.add_argument('--train_file', type=str, default=None, required=True)
    parser.add_argument('--valid_file', type=str, default=None, required=True)
    parser.add_argument('--test_file', type=str, default=None, required=True)
    parser.add_argument('--field', type=str, default='canonical_rxn')
    parser.add_argument('--before', type=int, default=-1)
    parser.add_argument('--output_path', type=str, default=None, required=True)
    args = parser.parse_args()

    train_df = pd.read_csv(os.path.join(args.data_path, args.train_file), keep_default_na=False)
    val_df = pd.read_csv(os.path.join(args.data_path, args.valid_file), keep_default_na=False)
    test_df = pd.read_csv(os.path.join(args.data_path, args.test_file), keep_default_na=False)

    if args.field == 'canonical_rxn':
        print('Reaction fingerprint')
        fingerprint_fn = compute_reaction_fingerprints
    else:
        print('Molecule fingerprint')
        fingerprint_fn = compute_molecule_fingerprints

    train_fp_file = os.path.join(args.output_path, 'train_fp.pkl')
    if not os.path.exists(train_fp_file):
        if args.before != -1:
            train_df = train_df[train_df['year'] < args.before].reset_index(drop=True)
        train_fps = fingerprint_fn(train_df[args.field])
        os.makedirs(args.output_path, exist_ok=True)
        with open(train_fp_file, 'wb') as f:
            np.save(f, train_fps)
    else:
        with open(train_fp_file, 'rb') as f:
            train_fps = np.load(f)

    train_id = train_df['id']

    # query_fps, query_id = train_fps, train_df['id']
    # rank = index_and_search(train_fps, query_fps)
    # result = [{'id': query_id[i], 'nn': [train_id[n] for n in nn]} for i, nn in enumerate(rank)]
    # with open(os.path.join(args.output_path, 'train.json'), 'w') as f:
    #     json.dump(result, f)

    query_fps, query_id = fingerprint_fn(val_df[args.field]), val_df['id']
    rank = index_and_search(train_fps, query_fps)
    result = [{'id': query_id[i], 'nn': [train_id[n] for n in nn]} for i, nn in enumerate(rank)]
    with open(os.path.join(args.output_path, 'val.json'), 'w') as f:
        json.dump(result, f)

    query_fps, query_id = fingerprint_fn(test_df[args.field]), test_df['id']
    rank = index_and_search(train_fps, query_fps)
    result = [{'id': query_id[i], 'nn': [train_id[n] for n in nn]} for i, nn in enumerate(rank)]
    with open(os.path.join(args.output_path, 'test.json'), 'w') as f:
        json.dump(result, f)

    if args.field == 'canonical_rxn':
        cnt = {x: 0 for x in [1, 3, 5, 10, 15]}
        for i, nn in enumerate(rank):
            test_row = test_df.iloc[i]
            train_rows = [train_df.iloc[n] for n in nn]
            hit_map = [compare_condition(test_row, train_row) for train_row in train_rows]
            for x in cnt:
                cnt[x] += np.any(hit_map[:x])

        print(cnt, len(test_df))
        for x in cnt:
            print(f"Top-{x}: {cnt[x] / len(test_df):.4f}", end='  ')
        print()
