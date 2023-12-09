import csv
import logging
from copy import deepcopy
import os
import pandas as pd
import re
from abc import ABC, abstractmethod
from collections import defaultdict
# from models.localretro_model.Extract_from_train_data import get_full_template
# from models.localretro_model.LocalTemplate.template_extractor import extract_from_reaction
# from models.localretro_model.Run_preprocessing import get_edit_site_retro
from typing import Dict, List
from rdkit import Chem
from rdkit.Chem import AllChem
from template_extraction.template_extractor import extract_from_reaction
from template_extraction.template_extract_utils import get_bonds_from_smiles

import sys
sys.path.append('../')
from textreact.tokenizer import BasicSmilesTokenizer

log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s" 
logging.basicConfig(format = log_format, level = logging.INFO)
 
logger = logging.getLogger()


ATOM_REGEX = re.compile(r"\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p")


def get_full_template(template, H_change, Charge_change, Chiral_change):
    H_code = ''.join([str(H_change[k+1]) for k in range(len(H_change))])
    Charge_code = ''.join([str(Charge_change[k+1]) for k in range(len(Charge_change))])
    Chiral_code = ''.join([str(Chiral_change[k+1]) for k in range(len(Chiral_change))])
    if Chiral_code == '':
        return '_'.join([template, H_code, Charge_code])
    else:
        return '_'.join([template, H_code, Charge_code, Chiral_code])


def canonicalize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    for a in mol.GetAtoms():
        a.SetAtomMapNum(0)
    canon_smile = Chem.MolToSmiles(mol)
    canon_perm = eval(mol.GetProp("_smilesAtomOutputOrder"))
    # atomidx2stridx = [i
    #         for i, token in enumerate(BasicSmilesTokenizer().tokenize(canon_smile))
    #         if ATOM_REGEX.fullmatch(token) is not None]
    # atomidx2canonstridx = [0 for _ in range(len(canon_perm))]
    # for canon_idx, orig_idx in enumerate(canon_perm):
    #     atomidx2canonstridx[orig_idx] = atomidx2stridx[canon_idx]
    atomidx2canonidx = [None for _ in range(len(canon_perm))]
    for canon_idx, orig_idx in enumerate(canon_perm):
        atomidx2canonidx[orig_idx] = canon_idx
    return canon_smile, atomidx2canonidx


class Processor(ABC):
    """Base class for processor"""

    @abstractmethod
    def __init__(self,
                 model_name: str,
                 model_args,                                        # let's enforce everything to be passed in args
                 model_config: Dict[str, any],                      # or config
                 data_name: str,
                 raw_data_files: List[str],
                 processed_data_path: str):
        self.model_name = model_name
        self.model_args = model_args
        self.model_config = model_config
        self.data_name = data_name
        self.raw_data_files = raw_data_files
        self.processed_data_path = processed_data_path

        os.makedirs(self.processed_data_path, exist_ok=True)

        self.check_count = 100

    def check_data_format(self) -> None:
        """Check that all files exists and the data format is correct for the first few lines"""
        logger.info(f"Checking the first {self.check_count} entries for each file")
        for fn in self.raw_data_files:
            if not fn:
                continue
            assert os.path.exists(fn), f"{fn} does not exist!"

            with open(fn, "r") as csv_file:
                csv_reader = csv.DictReader(csv_file)
                for i, row in enumerate(csv_reader):
                    if i > self.check_count:  # check the first few rows
                        break

                    assert (c in row for c in ["class", "rxn_smiles"]), \
                        f"Error processing file {fn} line {i}, ensure columns 'class' and " \
                        f"'rxn_smiles' is included!"

                    reactants, reagents, products = row["rxn_smiles"].split(">")
                    Chem.MolFromSmiles(reactants)  # simply ensures that SMILES can be parsed
                    Chem.MolFromSmiles(products)  # simply ensures that SMILES can be parsed

        logger.info("Data format check passed")

    @abstractmethod
    def preprocess(self) -> None:
        """Actual file-based preprocessing"""
        pass

# Adpated from Extract_from_train_data.py
class LocalRetroProcessor(Processor):
    """Class for LocalRetro Preprocessing"""

    def __init__(self,
                 model_name: str,
                 model_args,
                 model_config: Dict[str, any],
                 data_name: str,
                 raw_data_files: List[str],
                 processed_data_path: str,
                 num_cores: int = None):
        super().__init__(model_name=model_name,
                         model_args=model_args,
                         model_config=model_config,
                         data_name=data_name,
                         raw_data_files=raw_data_files,
                         processed_data_path=processed_data_path)
        self.train_file, self.val_file, self.test_file = raw_data_files
        self.num_cores = num_cores
        self.setting = {'verbose': False, 'use_stereo': True, 'use_symbol': True,
                        'max_unmap': 5, 'retro': True, 'remote': True, 'least_atom_num': 2,
                        "max_edit_n": 8, "min_template_n": 1}
        self.RXNHASCLASS = False

    def preprocess(self) -> None:
        """Actual file-based preprocessing"""
        self.extract_templates()
        self.match_templates()

    def extract_templates(self):
        """Adapted from Extract_from_train_data.py"""
        if all(os.path.exists(os.path.join(self.processed_data_path, fn))
               for fn in ["template_infos.csv", "atom_templates.csv", "bond_templates.csv"]):
            logger.info(f"Extracted templates found at {self.processed_data_path}, skipping extraction.")
            return

        logger.info(f"Extracting templates from {self.train_file}")

        with open(self.train_file, "r") as csv_file:
            csv_reader = csv.DictReader(csv_file)
            rxns = [row["rxn_smiles"].strip() for row in csv_reader]

        TemplateEdits = {}
        TemplateCs, TemplateHs, TemplateSs = {}, {}, {}
        TemplateFreq, templates_A, templates_B = defaultdict(int), defaultdict(int), defaultdict(int)

        for i, rxn in enumerate(rxns):
            try:
                rxn = {'reactants': rxn.split('>')[0], 'products': rxn.split('>')[-1], '_id': i}
                result = extract_from_reaction(rxn, self.setting)
                if 'reactants' not in result or 'reaction_smarts' not in result.keys():
                    logger.info(f'\ntemplate problem: id: {i}')
                    continue
                reactant = result['reactants']
                template = result['reaction_smarts']
                edits = result['edits']
                H_change = result['H_change']
                Charge_change = result['Charge_change']
                Chiral_change = result["Chiral_change"] if self.setting["use_stereo"] else {}

                template_H = get_full_template(template, H_change, Charge_change, Chiral_change)
                if template_H not in TemplateHs.keys():
                    TemplateEdits[template_H] = {edit_type: edits[edit_type][2] for edit_type in edits}
                    TemplateHs[template_H] = H_change
                    TemplateCs[template_H] = Charge_change
                    TemplateSs[template_H] = Chiral_change

                TemplateFreq[template_H] += 1
                for edit_type, bonds in edits.items():
                    bonds = bonds[0]
                    if len(bonds) > 0:
                        if edit_type in ['A', 'R']:
                            templates_A[template_H] += 1
                        else:
                            templates_B[template_H] += 1

            except Exception as e:
                logger.info(i, e)

            if i % 1000 == 0:
                logger.info(f'\r i = {i}, # of template: {len(TemplateFreq)}, '
                             f'# of atom template: {len(templates_A)}, '
                             f'# of bond template: {len(templates_B)}')
        logger.info('\n total # of template: %s' % len(TemplateFreq))

        derived_templates = {'atom': templates_A, 'bond': templates_B}

        ofn = os.path.join(self.processed_data_path, "template_infos.csv")
        TemplateInfos = pd.DataFrame(
            {'Template': k,
             'edit_site': TemplateEdits[k],
             'change_H': TemplateHs[k],
             'change_C': TemplateCs[k],
             'change_S': TemplateSs[k],
             'Frequency': TemplateFreq[k]} for k in TemplateHs.keys())
        TemplateInfos.to_csv(ofn)

        for k, local_templates in derived_templates.items():
            ofn = os.path.join(self.processed_data_path, f"{k}_templates.csv")
            with open(ofn, "w") as of:
                writer = csv.writer(of)
                header = ['Template', 'Frequency', 'Class']
                writer.writerow(header)

                sorted_tuples = sorted(local_templates.items(), key=lambda item: item[1])
                for i, (template, template_freq) in enumerate(sorted_tuples):
                    writer.writerow([template, template_freq, i + 1])

    def match_templates(self):
        """Adapted from Run_preprocessing.py"""
        # load_templates()
        template_dicts = {}

        for site in ['atom', 'bond']:
            fn = os.path.join(self.processed_data_path, f"{site}_templates.csv")
            with open(fn, "r") as csv_file:
                csv_reader = csv.DictReader(csv_file)
                template_dict = {row["Template"].strip(): int(row["Class"]) for row in csv_reader}
                logger.info(f'loaded {len(template_dict)} {site} templates')
                template_dicts[site] = template_dict

        fn = os.path.join(self.processed_data_path, "template_infos.csv")
        with open(fn, "r") as csv_file:
            csv_reader = csv.DictReader(csv_file)
            template_infos = {
                row["Template"]: {
                    "edit_site": eval(row["edit_site"]),
                    "frequency": int(row["Frequency"])
                } for row in csv_reader
            }
        logger.info('loaded total %s templates' % len(template_infos))

        # labeling_dataset()
        dfs = {}
        for phase, fn in [("train", self.train_file),
                          ("val", self.val_file),
                          ("test", self.test_file)]:
            with open(fn, "r") as csv_file:
                csv_reader = csv.DictReader(csv_file)
                rxns = [row["rxn_smiles"].strip() for row in csv_reader]
            reactants, products, reagents = [], [], []
            labels, frequency = [], []
            product_canon_smiles = []
            # product_atomidx2canonstridxs = []
            product_atomidx2canonidxs = []
            product_canon_bondss = []
            success = 0
            num_canon_smiles_mismatch = 0

            for i, rxn in enumerate(rxns):
                reactant, _, product = rxn.split(">")
                reagent = ''
                rxn_labels = []

                # get canonical permutation of atoms
                product_canon_smile, product_atomidx2canonidx = canonicalize_smiles(product)
                product_canon_bonds = get_bonds_from_smiles(product_canon_smile)

                # parse reaction and store results
                try:
                    rxn = {'reactants': reactant, 'products': product, '_id': i}
                    result = extract_from_reaction(rxn, self.setting)

                    template = result['reaction_smarts']
                    reactant = result['reactants']
                    product = result['products']
                    extracted_product_canon_smile, product_atomidx2canonidx = canonicalize_smiles(product)
                    num_canon_smiles_mismatch += int(extracted_product_canon_smile != product_canon_smile)
                    reagent = '.'.join(result['necessary_reagent'])
                    edits = {edit_type: edit_bond[0] for edit_type, edit_bond in result['edits'].items()}
                    H_change, Charge_change, Chiral_change = \
                        result['H_change'], result['Charge_change'], result['Chiral_change']
                    template_H = get_full_template(template, H_change, Charge_change, Chiral_change)

                    if template_H not in template_infos.keys():
                        reactants.append(reactant)
                        products.append(product)
                        reagents.append(reagent)
                        labels.append(rxn_labels)
                        frequency.append(0)
                        product_canon_smiles.append(product_canon_smile)
                        # product_atomidx2canonstridxs.append(product_atomidx2canonstridx)
                        product_atomidx2canonidxs.append(product_atomidx2canonidx)
                        product_canon_bondss.append(product_canon_bonds)
                        continue

                except Exception as e:
                    logger.info(i, e)
                    reactants.append(reactant)
                    products.append(product)
                    reagents.append(reagent)
                    labels.append(rxn_labels)
                    frequency.append(0)
                    product_canon_smiles.append(product_canon_smile)
                    # product_atomidx2canonstridxs.append(product_atomidx2canonstridx)
                    product_atomidx2canonidxs.append(product_atomidx2canonidx)
                    product_canon_bondss.append(product_canon_bonds)
                    continue

                edit_n = 0
                for edit_type in edits:
                    if edit_type == 'C':
                        edit_n += len(edits[edit_type]) / 2
                    else:
                        edit_n += len(edits[edit_type])

                if edit_n <= self.setting['max_edit_n']:
                    try:
                        success += 1
                        for edit_type, edit in edits.items():
                            for e in edit:
                                if edit_type in ['A', 'R']:
                                    rxn_labels.append(
                                        ('a', e, template_dicts['atom'][template_H]))
                                else:
                                    rxn_labels.append(
                                        ('b', e, template_dicts['bond'][template_H]))
                        reactants.append(reactant)
                        products.append(product)
                        reagents.append(reagent)
                        labels.append(rxn_labels)
                        frequency.append(template_infos[template_H]['frequency'])
                        product_canon_smiles.append(product_canon_smile)
                        # product_atomidx2canonstridxs.append(product_atomidx2canonstridx)
                        product_atomidx2canonidxs.append(product_atomidx2canonidx)
                        product_canon_bondss.append(product_canon_bonds)

                    except Exception as e:
                        logger.info(i, e)
                        reactants.append(reactant)
                        products.append(product)
                        reagents.append(reagent)
                        labels.append(rxn_labels)
                        frequency.append(0)
                        product_canon_smiles.append(product_canon_smile)
                        # product_atomidx2canonstridxs.append(product_atomidx2canonstridx)
                        product_atomidx2canonidxs.append(product_atomidx2canonidx)
                        product_canon_bondss.append(product_canon_bonds)
                        continue

                    if i % 1000 == 0:
                        logger.info(f'\r Processing {self.data_name} {phase} data..., '
                                     f'success {success} data ({i}/{len(rxns)})')
                else:
                    logger.info(f'\nReaction # {i} has too many edits ({edit_n})...may be wrong mapping!')
                    reactants.append(reactant)
                    products.append(product)
                    reagents.append(reagent)
                    labels.append(rxn_labels)
                    frequency.append(0)
                    product_canon_smiles.append(product_canon_smile)
                    # product_atomidx2canonstridxs.append(product_atomidx2canonstridx)
                    product_atomidx2canonidxs.append(product_atomidx2canonidx)
                    product_canon_bondss.append(product_canon_bonds)

            logger.info(f'\nDerived templates cover {success / len(rxns): .3f} of {phase} data reactions')
            logger.info(f'\nNumber of canonical smiles mismatches: {num_canon_smiles_mismatch} / {len(rxns)}')
            ofn = os.path.join(self.processed_data_path, f"preprocessed_{phase}.csv")
            dfs[phase] = pd.DataFrame(
                {'Reactants': reactants,
                 'Products': products,
                 'Reagents': reagents,
                 'Labels': labels,
                 'Frequency': frequency,
                 'ProductCanonSmiles': product_canon_smiles,
                 # 'ProductAtomIdx2CanonStrIdx': product_atomidx2canonstridxs,
                 'ProductAtomIdx2CanonIdx': product_atomidx2canonidxs,
                 'ProductCanonBonds': product_canon_bondss})
            dfs[phase].to_csv(ofn)

        # make_simulate_output()
        df = dfs["test"]
        ofn = os.path.join(self.processed_data_path, "simulate_output.txt")
        with open(ofn, 'w') as of:
            of.write('Test_id\tReactant\tProduct\t%s\n' % '\t'.join(
                [f'Edit {i + 1}\tProba {i + 1}' for i in range(self.setting['max_edit_n'])]))
            for i in df.index:
                labels = []
                for y in df['Labels'][i]:
                    if y != 0:
                        labels.append(y)
                if not labels:
                    labels = [(0, 0)]
                string_labels = '\t'.join([f'{l}\t{1.0}' for l in labels])
                of.write('%s\t%s\t%s\t%s\n' % (i, df['Reactants'][i], df['Products'][i], string_labels))

        # combine_preprocessed_data()
        dfs["train"]['Split'] = ['train'] * len(dfs["train"])
        dfs["val"]['Split'] = ['val'] * len(dfs["val"])
        dfs["test"]['Split'] = ['test'] * len(dfs["test"])
        all_valid = dfs["train"]._append(dfs["val"], ignore_index=True)
        all_valid = all_valid._append(dfs["test"], ignore_index=True)
        all_valid['Mask'] = [int(f >= self.setting['min_template_n']) for f in all_valid['Frequency']]
        ofn = os.path.join(self.processed_data_path, "labeled_data.csv")
        all_valid.to_csv(ofn, index=None)
        logger.info(f'Valid data size: {len(all_valid)}')


if __name__ == "__main__":
    from argparse import Namespace

    model_name = "localretro"
    model_args = Namespace()
    model_config = {}
    data_name = "USPTO_50K_year"
    raw_data_files = ["train.csv", "valid.csv", "test.csv"]
    raw_data_files = [os.path.join("../data_USPTO_50K_year_raw/", file_name) for file_name in raw_data_files]
    processed_data_path = "../data_template/USPTO_50K_year/"
    preprocessor = LocalRetroProcessor(model_name, model_args, model_config, data_name, raw_data_files, processed_data_path)
    preprocessor.check_data_format()
    preprocessor.preprocess()
