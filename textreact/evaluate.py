import os
from itertools import repeat
import numpy as np
import pandas as pd
import multiprocessing

import rdkit
from rdkit import Chem
rdkit.RDLogger.DisableLog('rdApp.*')

from .dataset import CONDITION_COLS
from .template_decoder import get_pred_smiles_from_templates


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


def evaluate_retrosynthesis(prediction, data_df, top_k, template_based=False, template_path=None, num_workers=16):
    num_example = len(data_df)
    with multiprocessing.Pool(num_workers) as p:
        gold_list = p.map(canonical_smiles, data_df['reactant_smiles'])
        if template_based:
            pred_prob_list = [[(*prediction, score)
                for prediction, score in zip(prediction[i]['prediction'], prediction[i]['score'])]
                for i in range(num_example)]
            atom_templates = pd.read_csv(os.path.join(template_path, 'atom_templates.csv'))
            bond_templates = pd.read_csv(os.path.join(template_path, 'bond_templates.csv'))
            template_infos = pd.read_csv(os.path.join(template_path, 'template_infos.csv'))
            atom_templates = {atom_templates['Class'][i]: atom_templates['Template'][i] for i in atom_templates.index}
            bond_templates = {bond_templates['Class'][i]: bond_templates['Template'][i] for i in bond_templates.index}
            template_infos = {template_infos['Template'][i]: {
                                                                 'edit_site': eval(template_infos['edit_site'][i]),
                                                                 'change_H': eval(template_infos['change_H'][i]),
                                                                 'change_C': eval(template_infos['change_C'][i]),
                                                                 'change_S': eval(template_infos['change_S'][i])
                                                             } for i in template_infos.index}
            pred_list = p.starmap(get_pred_smiles_from_templates,
                                  zip(pred_prob_list, data_df['product_smiles'],
                                      repeat(atom_templates), repeat(bond_templates), repeat(template_infos), repeat(top_k)))
        else:
            pred_list = [prediction[i]['prediction'] for i in range(num_example)]
        indices = p.starmap(_compare_pred_and_gold, [(p, g) for p, g in zip(pred_list, gold_list)])
    accuracy = {}
    for x in [1, 2, 3, 5, 10, 20]:
        accuracy[x] = sum([idx < x for idx in indices]) / num_example
    return accuracy
