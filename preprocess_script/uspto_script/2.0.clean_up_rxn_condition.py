import re
from collections import OrderedDict
from joblib import Parallel, delayed
import pandas as pd
import multiprocessing
import os
import argparse
import torch
from tqdm import tqdm
from utils import canonicalize_smiles, get_writer
from rxnmapper import RXNMapper


# rxns = ['[CH2:1]([O:8][C:9]([NH:11][C:12]1([C:15](O)=[O:16])[CH2:14][CH2:13]1)=[O:10])[C:2]1[CH:7]=[CH:6][CH:5]=[CH:4][CH:3]=1.[H]1[BH2][H][BH2]1.C([O-])([O-])=O.[K+].[K+]>O1CCCC1>[CH2:1]([O:8][C:9]([NH:11][C:12]1([CH2:15][OH:16])[CH2:13][CH2:14]1)=[O:10])[C:2]1[CH:3]=[CH:4][CH:5]=[CH:6][CH:7]=1']
# results = rxn_mapper.get_attention_guided_atom_maps(rxns)

debug = False


def remap_and_reassign_condition_role(org_rxn, org_solvent, org_catalyst, org_reagent):
    if org_rxn.split('>') == 1:
        return None
    if '|' in org_rxn:
        rxn, frag = org_rxn.split(' ')
    else:
        rxn, frag = org_rxn, ''

    org_solvent, org_catalyst, org_reagent = [
        canonicalize_smiles(x) for x in [org_solvent, org_catalyst, org_reagent]
    ]
    try:
        results = rxn_mapper.get_attention_guided_atom_maps([rxn])[0]
    except Exception as e:
        print('\n'+rxn+'\n')
        print(e)
        return None

    remapped_rxn = results['mapped_rxn']
    confidence = results['confidence']

    new_precursors, new_products = remapped_rxn.split('>>')

    pt = re.compile(r':(\d+)]')
    new_react_list = []
    new_reag_list = []
    for precursor in new_precursors.split('.'):
        if re.findall(pt, precursor):
            new_react_list.append(precursor)    # 有原子映射-->反应物
        else:
            new_reag_list.append(precursor)     # 无原子映射-->试剂

    new_reactants = '.'.join(new_react_list)
    react_maps = sorted(re.findall(pt, new_reactants))
    prod_maps = sorted(re.findall(pt, new_products))
    if react_maps != prod_maps:
        return None
    new_reagent_list = []
    c_list = org_catalyst.split('.')
    s_list = org_solvent.split('.')
    r_list = org_reagent.split('.')
    for r in new_reag_list:
        if (r not in c_list + s_list) and (r not in r_list):
            new_reagent_list.append(r)
    new_reagent_list += [x for x in r_list if x != '']
    catalyst = org_catalyst
    solvent = org_solvent
    reagent = '.'.join(new_reagent_list)
    can_react = canonicalize_smiles(new_reactants, clear_map=True)
    can_prod = canonicalize_smiles(new_products, clear_map=True)
    can_rxn = '{}>>{}'.format(can_react, can_prod)     
    results = OrderedDict()
    results['remapped_rxn'] = remapped_rxn                     # remapped_rxn中包含有反应条件 
    results['frag'] = frag
    results['confidence'] = confidence
    results['can_rxn'] = can_rxn                               # can_rxn中无反应条件，只有原子参与贡献的反应物和产物 
    results['catalyst'] = catalyst
    results['solvent'] = solvent
    results['reagent'] = reagent

    return results


def run_tasks(task):
    idx, rxn, solvent, catalyst, reagent, source = task
    if pd.isna(solvent):
        solvent = ''
    if pd.isna(catalyst):
        catalyst = ''
    if pd.isna(reagent):
        reagent = ''
    results = remap_and_reassign_condition_role(
        rxn, solvent, catalyst, reagent
    )

    # remapped_rxn = results['remapped_rxn']
    # frag = results['frag']
    # confidence = results['confidence']
    # can_rxn = results['can_rxn']
    # catalyst = results['catalyst']
    # solvent = results['solvent']
    # reagent = results['reagent']

    return idx, results, source


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--split_group', type=int, default=4)
    parser.add_argument('--group', type=int, default=1)
    args = parser.parse_args()

    assert args.group <= args.split_group-1

    print('Debug:', debug)
    print('Split group: {}'.format(args.split_group))
    print('Group number: {}'.format(args.group))
    print('GPU index: {}'.format(args.gpu))

    device = torch.device('cuda:{}'.format(args.gpu)
                          if args.gpu >= 0 else 'cpu')
    rxn_mapper = RXNMapper(device=device)
    source_data_path = '../../dataset/source_dataset/'
    rxn_condition_fname = 'uspto_rxn_condition.csv'
    new_database_fpath = os.path.join(
        source_data_path, 'uspto_rxn_condition_remapped_and_reassign_condition_role_group_{}.csv'.format(args.group))
    # n_core = 14

    # pool = multiprocessing.Pool(n_core)
    if debug:
        database = pd.read_csv(os.path.join(
            source_data_path, rxn_condition_fname), nrows=10001)
    else:
        database = pd.read_csv(os.path.join(
            source_data_path, rxn_condition_fname))
    print('All data number: {}'.format(len(database)))

    group_size = len(database) // args.split_group

    if args.group >= args.split_group-1:
        database = database.iloc[args.group * group_size:]
    else:
        database = database.iloc[args.group *
                                 group_size:(args.group+1) * group_size]
    
    print('Caculate index {} to {}'.format(database.index.min(), database.index.max()))

    # rxn_smiles = database['rxn_smiles'].tolist()

    # tasks = [(idx, rxn, database.iloc[idx].solvent, database.iloc[idx].catalyst, database.iloc[idx].reagent, database.iloc[idx].source)
    #          for idx, rxn in tqdm(enumerate(rxn_smiles), total=len(rxn_smiles))]
    header = [
        'remapped_rxn',
        'fragment',
        'confidence',
        'canonical_rxn',
        'catalyst',
        'solvent',
        'reagent',
        'source',
        'org_rxn'
    ]
    fout, writer = get_writer(
        new_database_fpath, header=header
    )
    all_results = []
    for row in tqdm(database.itertuples(), total=len(database)):
        task = (row.index, row.rxn_smiles, row.solvent, row.catalyst, row.reagent, row.source)
        try:
            run_results = run_tasks(task)
            idx, results, source = run_results
            if results:
                results['source'] = source
                results['org_rxn'] = task[1]
                row = list(results.values())
                assert len(row) == len(header)
                writer.writerow(row)
                fout.flush()
        except Exception as e:
            print(e)
            pass
    # for results in tqdm(pool.imap_unordered(run_tasks, tasks), total=len(tasks)):
    #     all_results.append(results)
    # all_results = Parallel(n_jobs=n_core, verbose=1)(
    #     delayed(run_tasks)(task) for task in tqdm(tasks))
    fout.close()
    new_database = pd.read_csv(new_database_fpath)
    reset_header = [
        'source',
        'org_rxn',
        'fragment',
        'remapped_rxn',
        'confidence',
        'canonical_rxn',
        'catalyst',
        'solvent',
        'reagent',
    ]
    new_database = new_database[reset_header]
    new_database.to_csv(new_database_fpath, index=False)
    print('Done!')
