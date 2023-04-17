import numpy as np
import multiprocessing

import rdkit
from rdkit import Chem
rdkit.RDLogger.DisableLog('rdApp.*')

from .dataset import CONDITION_COLS


def evaluate_reaction_condition(prediction, data_df):
    cnt = {x: 0 for x in [1, 3, 5, 10, 15]}
    for i, output in prediction.items():
        label = data_df.loc[i, CONDITION_COLS].tolist()
        hit_map = [pred == label for pred in output['prediction']]
        for x in cnt:
            cnt[x] += np.any(hit_map[:x])
    num_example = len(data_df)
    accuracy = {x: cnt[x] / num_example for x in cnt}
    return accuracy


def canonical_smiles(smiles):
    try:
        canon_smiles = Chem.CanonSmiles(smiles)
    except:
        canon_smiles = smiles
    return canon_smiles


def _compare_pred_and_gold(pred, gold):
    pred = [canonical_smiles(smiles) for smiles in pred]
    for i, smiles in enumerate(pred):
        if smiles == gold:
            return i
    return 100000


def evaluate_retrosynthesis(prediction, data_df, num_workers=16):
    num_example = len(data_df)
    with multiprocessing.Pool(num_workers) as p:
        gold_list = p.map(canonical_smiles, data_df['reactant_smiles'])
        pred_list = [prediction[i]['prediction'] for i in range(num_example)]
        indices = p.starmap(_compare_pred_and_gold, [(p, g) for p, g in zip(pred_list, gold_list)])
    accuracy = {}
    for x in [1, 2, 3, 5, 10, 20]:
        accuracy[x] = sum([idx < x for idx in indices]) / num_example
    return accuracy
