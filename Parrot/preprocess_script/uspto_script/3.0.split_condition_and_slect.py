from collections import defaultdict, namedtuple
import os
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from utils import MolRemover, get_mol_charge, list_of_metal_atoms, mol_charge_class
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
# if any(metal in cat for metal in list_of_metals):
# 	metal_cat_dict[key] = True


if __name__ == '__main__':
    debug = False
    unknown_check = False
    split_token = '分'
    print('Debug:', debug)
    remove_threshold = 100
    source_data_path = '../../dataset/source_dataset/'
    duplicate_removal_fname = 'uspto_rxn_condition_remapped_and_reassign_condition_role_rm_duplicate.csv'
    freq_info_path = os.path.join(source_data_path, 'freq_info')

    if debug:
        database = pd.read_csv(os.path.join(
            source_data_path, duplicate_removal_fname), nrows=10000)

    else:
        database = pd.read_csv(os.path.join(
            source_data_path, duplicate_removal_fname))

    # 去除条件频率少于阈值以下的数据
    print('Remove data with less than remove_threshold...')
    print(f'remove_threshold = {remove_threshold}')
    print('data number before remove: {}'.format(len(database)))
    condition_roles = ['catalyst', 'solvent', 'reagent']
    remove_index = pd.isna(database.index)
    for role in condition_roles:
        df = pd.read_csv(os.path.join(freq_info_path, f'{role}_freq.csv'))
        remove_compund = df[df['freq_cnt'] < remove_threshold]['smiles']
        remove_index = remove_index | database[role].isin(remove_compund)
    database_remove_below_threshold = database.loc[~remove_index].reset_index(drop=True)
    print('data number after remove: {}'.format(
        len(database_remove_below_threshold)))
    # database_remove_below_threshold.to_csv
    remover = MolRemover(defnFilename='../reagent_Ionic_compound.txt')
    reagent2index_dict = defaultdict(list)
    for idx, reagent in tqdm(enumerate(database_remove_below_threshold.reagent.tolist()), total=len(database_remove_below_threshold)):
        reagent2index_dict[reagent].append(idx)

    if unknown_check:
        # 检查带有离子的试剂，删去非电中性的组合
        unknown_combination = defaultdict(int)
        reagent2single_reagent_dict = {}
        for reagent in tqdm(reagent2index_dict):
            if pd.isna(reagent):
                reagent_namedtuple = namedtuple('reagent', ['known', 'unknown'])
                reagent2single_reagent_dict[reagent] = reagent_namedtuple(
                    [], [])
                continue
            reagent_mol = Chem.MolFromSmiles(reagent)
            reagent_mol_after_rm, remove_mol = remover.StripMolWithDeleted(
                reagent_mol, onlyFrags=False)
            reagent_smiles_after_rm = Chem.MolToSmiles(reagent_mol_after_rm)
            reagent_smiles_after_rm_list = reagent_smiles_after_rm.split('.')
            reagent_mols_after_rm = [Chem.MolFromSmiles(
                x) for x in reagent_smiles_after_rm_list]
            known_ionic = [Chem.MolToSmiles(x) for x in remove_mol]
            _unknown = []
            reagent_charge_neutral = []
            for mol in reagent_mols_after_rm:
                mol_charge_flag, mol_neutralization = get_mol_charge(mol)
                if mol_charge_flag != mol_charge_class[2]:
                    smi = Chem.MolToSmiles(mol)
                    if smi != '':
                        _unknown.append(smi)
                else:
                    smi = Chem.MolToSmiles(mol)
                    if smi != '':
                        reagent_charge_neutral.append(smi)
            reagent_namedtuple = namedtuple('reagent', ['known', 'unknown'])
            _known = reagent_charge_neutral + known_ionic
            # if '' in _known:
            #     print(_known)
            # if '' in _unknown:
            #     print(_unknown)
            if _unknown:
                unknown_combination['.'.join(_unknown)] += 1
            reagent2single_reagent_dict[reagent] = reagent_namedtuple(
                _known, _unknown)
        unknown_combination = list(unknown_combination.items())
        unknown_combination.sort(key=lambda x:x[1], reverse=True)
        print('Unknown reagent data count:', sum([x[1] for x in unknown_combination]))
        with open('../reagent_unknown.txt', 'w', encoding='utf-8') as f:
            for line in unknown_combination:
                f.write('{},{}\n'.format(line[0], line[1]))
    else:
        block_unknown_combination = pd.read_csv('../reagent_unknown.txt', header=None)
        block_unknown_combination.columns = ['smiles', 'cnt']
        print('Will block {} reagent combination, a total of {} data will be deleted.'.format(len(block_unknown_combination), block_unknown_combination['cnt'].sum()))

        reagent2single_reagent_dict = defaultdict(list)
        for reagent in tqdm(reagent2index_dict):
            if pd.isna(reagent):
                reagent2single_reagent_dict[reagent].append('')
                continue
            reagent_mol = Chem.MolFromSmiles(reagent)
            reagent_mol_after_rm, remove_mol = remover.StripMolWithDeleted(
                reagent_mol, onlyFrags=False)
            reagent_smiles_after_rm = Chem.MolToSmiles(reagent_mol_after_rm)
            reagent_smiles_after_rm_list = reagent_smiles_after_rm.split('.')
            reagent_mols_after_rm = [Chem.MolFromSmiles(
                x) for x in reagent_smiles_after_rm_list]
            known_ionic = [Chem.MolToSmiles(x) for x in remove_mol]
            _unknown = []
            reagent_charge_neutral = []
            for mol in reagent_mols_after_rm:
                mol_charge_flag, mol_neutralization = get_mol_charge(mol)
                if mol_charge_flag != mol_charge_class[2]:
                    smi = Chem.MolToSmiles(mol)
                    if smi != '':
                        _unknown.append(smi)
                else:
                    smi = Chem.MolToSmiles(mol)
                    if smi != '':
                        reagent_charge_neutral.append(smi)
            if _unknown:
                _unknown_smiles = '.'.join(_unknown)
                assert _unknown_smiles in block_unknown_combination['smiles'].tolist()
            _known = reagent_charge_neutral + known_ionic
            reagent2single_reagent_dict[reagent] += _known
    unknown_drop_index = []
    for reagent in reagent2single_reagent_dict:
        if not reagent2single_reagent_dict[reagent]:
            unknown_drop_index.extend(reagent2index_dict[reagent])
    reagent2single_reagent_dict = {k:v for k,v in reagent2single_reagent_dict.items() if v}
    database_remove_below_threshold = database_remove_below_threshold.drop(unknown_drop_index)
    database_remove_below_threshold = database_remove_below_threshold.reset_index(drop=True)
    reagent2index_dict = defaultdict(list)
    for idx, reagent in tqdm(enumerate(database_remove_below_threshold.reagent.tolist()), total=len(database_remove_below_threshold)):
        reagent2index_dict[reagent].append(idx)

    # 按照 https://pubs.acs.org/doi/10.1021/acscentsci.8b00357 论文所提到的将catalyst>1, solvent>2, reagent>2 的数据排除
    print('Exceeding one catalyst, two solvents, or two reagents --> remove')
    remove_index_for_excess = pd.isna(database_remove_below_threshold.index)
    # reagent > 2 remove!
    for reagent in reagent2index_dict:
        if len(reagent2single_reagent_dict[reagent]) > 2:
            remove_idx = reagent2index_dict[reagent]
            for _idx in remove_idx:
                remove_index_for_excess[_idx] = True
    for _idx, catalyst in enumerate(database_remove_below_threshold.catalyst.tolist()):
        if not pd.isna(catalyst):
            if len(catalyst.split('.')) > 1:
                remove_index_for_excess[_idx] = True
    for _idx, solvent in enumerate(database_remove_below_threshold.solvent.tolist()):
        if not pd.isna(solvent):
            if len(solvent.split('.')) > 2:
                remove_index_for_excess[_idx] = True
    database_remove_below_threshold = database_remove_below_threshold.loc[~remove_index_for_excess].reset_index(drop=True)
    reagent2index_dict = defaultdict(list)
    for idx, reagent in tqdm(enumerate(database_remove_below_threshold.reagent.tolist()), total=len(database_remove_below_threshold)):
        reagent2index_dict[reagent].append(idx)
    print('Spliting conditions...')
    database_remove_below_threshold['catalyst_split'] = database_remove_below_threshold['catalyst']
    database_remove_below_threshold['solvent_split'] = ['']*len(database_remove_below_threshold)
    database_remove_below_threshold['reagent_split'] = ['']*len(database_remove_below_threshold)
    for reagent in reagent2index_dict:
        write_index = reagent2index_dict[reagent]
        write_value = split_token.join(reagent2single_reagent_dict[reagent])
        database_remove_below_threshold.loc[write_index, 'reagent_split'] = write_value

    solvent_split = []
    for x in database_remove_below_threshold['solvent'].tolist():
        if pd.isna(x): 
            solvent_split.append('')
            continue
        solvent_split.append(split_token.join(x.split('.')))
    database_remove_below_threshold['solvent_split'] = solvent_split
    database_remove_below_threshold.to_csv(os.path.join(source_data_path, 'uspto_rxn_condition_remapped_and_reassign_condition_role_rm_duplicate_rm_excess.csv'), index=False)
    print('Remaining data in the end:',len(database_remove_below_threshold))
    print('Unique canonical reaction:', len(set(database_remove_below_threshold['canonical_rxn'])))
    print('Unique remapped reaction:', len(set(database_remove_below_threshold['remapped_rxn'])))
    print('Unique catalyst:', len(set(database_remove_below_threshold['catalyst'])))
    print('Unique solvent:', len(set(database_remove_below_threshold['solvent'])))
    print('Unique reagent:', len(set(database_remove_below_threshold['reagent'])))
    print('Done!')
