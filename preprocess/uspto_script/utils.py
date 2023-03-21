from collections import defaultdict, namedtuple
import csv
import json
import os
import pickle
from rdkit import Chem
import pandas as pd
from rdkit.Chem.SaltRemover import SaltRemover, InputFormat

# atapted from https://github.com/Coughy1991/Reaction_condition_recommendation/blob/64f151e302abcb87e0a14088e08e11b4cea8d5ab/scripts/prepare_data_cont_2_rgt_2_slv_1_cat_temp_deploy.py#L658-L690
list_of_metal_atoms = ['Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag',
                       'Cd', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg',
                       # 'Cn',
                       'Ln',
                       'Ce',
                       'Pr',
                       'Nd',
                       'Pm',
                       'Sm',
                       'Eu',
                       'Gd',
                       'Tb',
                       'Dy',
                       'Ho',
                       'Er',
                       'Tm',
                       'Yb',
                       'Lu',
                       'Ac',
                       'Th',
                       'Pa',
                       'U',
                       'Np',
                       'Am',
                       'Cm',
                       'Bk',
                       'Cf',
                       'Es',
                       'Fm',
                       'Md',
                       'No',
                       'Lr',
                       ]

mol_charge_class = [
    'Postive',
    'Negative',
    'Neutral'
]

class MolRemover(SaltRemover):
    def __init__(self, defnFilename=None, defnData=None, defnFormat=InputFormat.SMARTS):
        super().__init__(defnFilename, defnData, defnFormat)

    def _StripMol(self, mol, dontRemoveEverything=False, onlyFrags=False):

        def _applyPattern(m, salt, notEverything, onlyFrags=onlyFrags):
            nAts = m.GetNumAtoms()
            if not nAts:
                return m
            res = m

            t = Chem.DeleteSubstructs(res, salt, onlyFrags)
            if not t or (notEverything and t.GetNumAtoms() == 0):
                return res
            res = t
            while res.GetNumAtoms() and nAts > res.GetNumAtoms():
                nAts = res.GetNumAtoms()
                t = Chem.DeleteSubstructs(res, salt, True)
                if notEverything and t.GetNumAtoms() == 0:
                    break
                res = t
            return res

        StrippedMol = namedtuple('StrippedMol', ['mol', 'deleted'])
        deleted = []
        if dontRemoveEverything and len(Chem.GetMolFrags(mol)) <= 1:
            return StrippedMol(mol, deleted)
        modified = False
        natoms = mol.GetNumAtoms()
        for salt in self.salts:
            mol = _applyPattern(mol, salt, dontRemoveEverything, onlyFrags)
            if natoms != mol.GetNumAtoms():
                natoms = mol.GetNumAtoms()
                modified = True
                deleted.append(salt)
                if dontRemoveEverything and len(Chem.GetMolFrags(mol)) <= 1:
                    break
        if modified and mol.GetNumAtoms() > 0:
            Chem.SanitizeMol(mol)
        return StrippedMol(mol, deleted)

    def StripMolWithDeleted(self, mol, dontRemoveEverything=False, onlyFrags=False):
        return self._StripMol(mol, dontRemoveEverything, onlyFrags=onlyFrags)

def pickle2json(fname):
    name, ext = os.path.splitext(fname)
    with open(fname, 'rb') as f:
        data = pickle.load(f)

    with open(name + '.json', 'w', encoding='utf-8') as f:
        json.dump(data, f)


def read_json(fpath):
    with open(fpath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def read_pickle(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)


def canonicalize_smiles(smi, clear_map=False):
    if pd.isna(smi):
        return ''
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        if clear_map:
            [atom.ClearProp('molAtomMapNumber') for atom in mol.GetAtoms()]
        return Chem.MolToSmiles(mol)
    else:
        return ''


def covert_to_series(smi, none_pandding):
    if pd.isna(smi):
        return pd.Series([none_pandding])
    elif smi == '':
        return pd.Series([none_pandding])
    else:
        return pd.Series(smi.split('.'))


def get_writer(output_name, header):
    # output_name = os.path.join(cmd_args.save_dir, fname)
    fout = open(output_name, 'w')
    writer = csv.writer(fout)
    writer.writerow(header)
    return fout, writer


def calculate_frequency(input_list, report=True):
    output_dict = defaultdict(int)
    for x in input_list:
        output_dict[x] += 1

    output_items = list(output_dict.items())
    output_items.sort(key=lambda x: x[1], reverse=True)

    if report:
        report_threshold = [10000, 5000, 1000, 500, 100, 50, 1]
        for t in report_threshold:
            t_list = [x for x in output_items if x[1] > t]
            print('Frequency >={} : {}'.format(t, len(t_list)))

    return output_items


def get_mol_charge(mol):
    mol_neutralization = None
    positive = []
    negative = []
    for atom in mol.GetAtoms():
        charge = atom.GetFormalCharge()
        if charge > 0:
            positive.append(charge)
        elif charge < 0:
            negative.append(charge)
    if len(positive) == 0 and len(negative) == 0:
        mol_charge_flag = mol_charge_class[2]
        mol_neutralization = False
    elif len(positive) != 0 and len(negative) == 0:
        mol_charge_flag = mol_charge_class[0]
        mol_neutralization = False
    elif len(positive) == 0 and len(negative) != 0:
        mol_charge_flag = mol_charge_class[1]
        mol_neutralization = False
    elif len(positive) != 0 and len(negative) != 0:
        mol_charge = sum(positive) + sum(negative)
        if mol_charge > 0:
            mol_charge_flag = mol_charge_class[0]
        elif mol_charge < 0:
            mol_charge_flag = mol_charge_class[1]
        else:
            mol_charge_flag = mol_charge_class[2]
        mol_neutralization = True
    return mol_charge_flag, mol_neutralization





if __name__ == '__main__':
    # smi = 'CC(C)C[Al+]CC(C)C.O=C(O)CC(O)(CC(=O)O)C(=O)O.[H-]'
    # smi = 'CCN(CC)CC.CN(C)C(On1nnc2ccccc21)=[N+](C)C.F[B-](F)(F)F'
    smi = 'O.[Al+3].[H-].[H-].[H-].[H-].[Li+].[Na+].[OH-]'
    mol = Chem.MolFromSmiles(smi)
    # print(get_mol_charge(mol))
    # from rdkit.Chem.SaltRemover import SaltRemover
    remover = MolRemover(defnFilename='../reagent_Ionic_compound.txt')
    remover.StripMolWithDeleted(
        mol, dontRemoveEverything=False, onlyFrags=False)
