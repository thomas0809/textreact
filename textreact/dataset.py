import copy
import json
import os.path
import random
import pickle
import logging
import numpy as np
import pandas as pd
import rdkit.Chem as Chem
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import DefaultDataCollator, DataCollatorWithPadding


CONDITION_COLS = ['catalyst1', 'solvent1', 'solvent2', 'reagent1', 'reagent2']


class ReactionConditionDataset(Dataset):

    def __init__(self, args, data_file, enc_tokenizer, dec_tokenizer, split='train'):
        super().__init__()
        self.args = args
        self.enc_tokenizer = enc_tokenizer
        self.dec_tokenizer = dec_tokenizer
        self.data_df = pd.read_csv(data_file, keep_default_na=False)
        self.corpus = None
        self.neighbors = None
        if args.debug:
            self.data_df = self.data_df[:1000]
        self.name = split
        self.split = split
        self.collator = ReactionConditionCollator(args, enc_tokenizer, return_label=(split != 'test'))

    def __len__(self):
        return len(self.data_df)

    def load_corpus(self, corpus, nn_file):
        self.corpus = corpus
        with open(nn_file) as f:
            self.neighbors = json.load(f)

    def __getitem__(self, idx):
        row = self.data_df.iloc[idx]
        rxn_smiles = row['canonical_rxn']
        # Data augmentation: shuffle the rxn smiles
        if self.split == 'train' and self.args.shuffle_smiles:
            rxn_smiles = random_shuffle_reaction_smiles(rxn_smiles)
        # Collect text from nearest neighbors
        if self.args.num_neighbors > 0:
            neighbors_texts = [self.corpus[rxn_id] for rxn_id in self.neighbors[idx]['nn']]
            if self.split == 'train':
                if random.random() < 0.2:
                    selected_texts = neighbors_texts[:self.args.num_neighbors]
                else:
                    selected_texts = random.sample(neighbors_texts, k=self.args.num_neighbors)
            else:
                if self.args.skip_gold_neighbor:
                    neighbors_texts = [self.corpus[rxn_id]
                                       for rxn_id in self.neighbors[idx]['nn'] if rxn_id != row['id']]
                selected_texts = neighbors_texts[:self.args.num_neighbors]
            # nn_text = ' '.join(selected_texts)
            nn_text = ''
            for i, text in enumerate(selected_texts):
                nn_text += f' ({i}) ' + text
        else:
            nn_text = None
        # Encoder input
        enc_input = self.enc_tokenizer(
            rxn_smiles, text_pair=nn_text, truncation=False, return_token_type_ids=False, verbose=False)
        # Truncate
        max_length = self.args.max_length
        enc_input = {k: v[:max_length] for k, v in enc_input.items()}
        # MLM
        additional = {}
        if self.args.mlm and self.split == 'train':
            origin_ids = copy.deepcopy(enc_input['input_ids'])
            input_ids = copy.deepcopy(enc_input['input_ids'])
            input_len = len(input_ids)
            mlm_labels = [-100] * input_len
            num_tokens_to_mask = int(len(input_ids) * self.args.mlm_ratio)
            for i in range(100):
                k = np.random.poisson(lam=3)
                if k == 0 or k > min(10, input_len) or k > num_tokens_to_mask:
                    continue
                start = random.randrange(input_len - k)
                end = start + k
                span_ids = origin_ids[start:end]
                input_ids = input_ids[:start] + [self.enc_tokenizer.mask_token_id] * k + input_ids[end:]
                mlm_labels = mlm_labels[:start] + span_ids + mlm_labels[end:]
                num_tokens_to_mask -= k
                if num_tokens_to_mask < 0:
                    break
            input_ids, position_ids, mlm_labels = self._reorder_masked_sequence(input_ids, mlm_labels)
            enc_input['input_ids'] = input_ids
            enc_input['position_ids'] = position_ids
            additional['mlm_labels'] = mlm_labels
        # Decoder input
        if self.split != 'test':
            conditions = row[CONDITION_COLS].tolist()
            dec_input = self.dec_tokenizer(conditions, return_token_type_ids=False)
        else:
            dec_input = {}
        return idx, enc_input, dec_input, additional

    def _reorder_masked_sequence(self, input_ids, mlm_labels):
        position_ids_masked, position_ids_unmasked = [], []
        input_ids_masked, input_ids_unmasked = [], []
        mlm_labels_masked = []
        mask_token_id = self.enc_tokenizer.mask_token_id
        for i in range(len(input_ids)):
            if input_ids[i] == mask_token_id:
                input_ids_masked.append(input_ids[i])
                mlm_labels_masked.append(mlm_labels[i])
                position_ids_masked.append(i)
            else:
                input_ids_unmasked.append(input_ids[i])
                position_ids_unmasked.append(i)
        return input_ids_masked + input_ids_unmasked, position_ids_masked + position_ids_unmasked, mlm_labels_masked


class ReactionConditionCollator:

    def __init__(self, args, enc_tokenizer, return_label=True):
        self.args = args
        self.enc_tokenizer=enc_tokenizer
        self.return_label = return_label

    def __call__(self, features):
        indices = [feat[0] for feat in features]
        enc_inputs = [feat[1] for feat in features]
        dec_inputs = [feat[2] for feat in features]
        additionals = [feat[3] for feat in features]
        batch_in = {
            'input_ids': self.pad_sequences([enc_input['input_ids'] for enc_input in enc_inputs],
                                            self.enc_tokenizer.pad_token_id),
            'attention_mask': self.pad_sequences([enc_input['attention_mask'] for enc_input in enc_inputs], 0)
        }
        if 'position_ids' in enc_inputs[0]:
            batch_in['position_ids'] = self.pad_sequences([enc_input['position_ids'] for enc_input in enc_inputs], 0)
        if self.return_label:
            batch_in['decoder_input_ids'] = torch.tensor([dec_input['input_ids'] for dec_input in dec_inputs])
            batch_in['decoder_attention_mask'] = torch.tensor([dec_input['attention_mask'] for dec_input in dec_inputs])
        batch_out = {}
        if 'mlm_labels' in additionals[0]:
            batch_out['mlm_labels'] = self.pad_sequences(
                [additional['mlm_labels'] for additional in additionals], pad_id=-100)
        return indices, batch_in, batch_out

    def pad_sequences(self, sequences, pad_id, max_length=None, return_tensor=True):
        if max_length is None:
            max_length = max([len(seq) for seq in sequences])
        output = []
        for seq in sequences:
            if len(seq) < max_length:
                output.append(seq + [pad_id] * (max_length - len(seq)))
            else:
                output.append(seq[:max_length])
        if return_tensor:
            return torch.tensor(output)
        else:
            return output


def read_corpus(corpus_file, cache_path=None):
    if cache_path:
        cache_file = os.path.join(cache_path, corpus_file.replace('.csv', '.pkl'))
        if os.path.exists(cache_file):
            logging.info(f' Load corpus from: {cache_file}')
            with open(cache_file, 'rb') as f:
                corpus = pickle.load(f)
            return corpus
    corpus_df = pd.read_csv(corpus_file, keep_default_na=False)
    corpus = {}
    for i, row in corpus_df.iterrows():
        if len(row['heading_text']) > 0:
            corpus[row['id']] = row['heading_text'] + '. ' + row['paragraph_text']
        else:
            corpus[row['id']] = row['paragraph_text']
    if cache_path:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        logging.info(f' Save corpus to: {cache_file}')
        with open(cache_file, 'wb') as f:
            pickle.dump(corpus, f)
    return corpus


def generate_train_label_corpus(train_file):
    """Use rxn smiles and labels in the training data as the corpus, instead of text."""
    train_df = pd.read_csv(train_file, keep_default_na=False)
    corpus = {}
    for i, row in train_df.iterrows():
        condition = ''
        for col in CONDITION_COLS:
            if len(row[col]) > 0:
                if condition == '':
                    condition += row[col]
                else:
                    condition += '.' + row[col]
        rxn_smiles = row['canonical_rxn']
        corpus[row['id']] = rxn_smiles.replace('>>', f'>{condition}>')
    return corpus


def random_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        new_smiles = Chem.MolToSmiles(mol, doRandom=True, canonical=False)
        return new_smiles
    except:
        return smiles


def random_shuffle_reaction_smiles(rxn_smiles, p=0.8):
    if random.random() > p:
        return rxn_smiles
    reactant_str, product_str = rxn_smiles.split('>>')
    reactants = [random_smiles(smiles) for smiles in reactant_str.split('.')]
    products = [random_smiles(smiles) for smiles in product_str.split('.')]
    random.shuffle(reactants)
    random.shuffle(products)
    return '.'.join(reactants) + '>>' + '.'.join(products)
