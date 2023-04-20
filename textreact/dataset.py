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


CONDITION_COLS = ['catalyst1', 'solvent1', 'solvent2', 'reagent1', 'reagent2']


class BaseDataset(Dataset):

    def __init__(self, args, data_file, enc_tokenizer, dec_tokenizer, split='train'):
        super().__init__()
        self.args = args
        self.enc_tokenizer = enc_tokenizer
        self.dec_tokenizer = dec_tokenizer
        self.data_df = pd.read_csv(data_file, keep_default_na=False)
        self.indices = self.data_df[args.id_field].tolist()
        self.corpus = None
        self.neighbors = None
        self.skip_gold_neighbor = False
        self.name = split
        self.split = split
        self.collator = DataCollator(args, enc_tokenizer, dec_tokenizer, return_label=(split != 'test'))

    def __len__(self):
        return len(self.data_df)

    def load_corpus(self, corpus, nn_file):
        self.corpus = corpus
        with open(nn_file) as f:
            nn_data = json.load(f)
            self.neighbors = {ex['id']: ex['nn'] for ex in nn_data}

    def deduplicate_neighbors(self, neighbors_ids):
        output = []
        for i in neighbors_ids:
            flag = False
            for j in output:
                if self.corpus[i] == self.corpus[j]:
                    flag = True
                    break
            if not flag:
                output.append(i)
        return output

    def get_neighbor_text(self, idx, return_list=False):
        rxn_id = self.indices[idx]
        neighbors_ids = self.neighbors[rxn_id]
        if self.split == 'train':
            if self.args.use_gold_neighbor:
                if rxn_id in neighbors_ids:
                    neighbors_ids.remove(rxn_id)
                if rxn_id in self.corpus:
                    neighbors_ids = [rxn_id] + neighbors_ids
            neighbors_ids = self.deduplicate_neighbors(neighbors_ids)
            neighbors_texts = [self.corpus[i] for i in neighbors_ids[:self.args.max_num_neighbors]]
            if random.random() < self.args.random_neighbor_ratio:
                selected_texts = random.sample(neighbors_texts, k=self.args.num_neighbors)
            else:
                selected_texts = neighbors_texts[:self.args.num_neighbors]
        else:
            if self.skip_gold_neighbor and rxn_id in self.corpus:
                gold_text = self.corpus[rxn_id]
                neighbors_ids = [i for i in neighbors_ids if self.corpus[i] != gold_text]
            neighbors_ids = self.deduplicate_neighbors(neighbors_ids)
            selected_texts = [self.corpus[i] for i in neighbors_ids[:self.args.num_neighbors]]
        return selected_texts if return_list \
            else ''.join([f' ({i}) {text}' for i, text in enumerate(selected_texts)])

    def apply_mlm(self, enc_input, output):
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
        output['mlm_labels'] = mlm_labels
        return enc_input, output

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

    def prepare_encoder_input(self, idx):
        raise NotImplementedError

    def prepare_decoder_input(self, idx):
        raise NotImplementedError

    def __getitem__(self, idx, debug=False):
        enc_input, dec_input, outputs = {}, {}, {}
        # Encoder input
        enc_input = self.prepare_encoder_input(idx)
        enc_input = {k: v[:self.args.max_length] for k, v in enc_input.items()}
        # MLM
        if self.args.mlm and self.split == 'train' and not debug:
            enc_input, outputs = self.apply_mlm(enc_input, outputs)
        # Decoder input
        dec_input = self.prepare_decoder_input(idx)
        dec_input = {k: v[:self.args.max_dec_length] for k, v in dec_input.items()}
        # Merge
        inputs = enc_input.copy()
        inputs.update({f'decoder_{k}': v for k, v in dec_input.items()})
        return idx, inputs, outputs

    def print_example(self, idx=0):
        idx, inputs, outputs = self.__getitem__(idx, debug=True)
        # print(inputs)
        print(self.enc_tokenizer.decode(inputs['input_ids']))
        print(self.dec_tokenizer.decode(inputs['decoder_input_ids']))


class ReactionConditionDataset(BaseDataset):

    def prepare_encoder_input(self, idx):
        row = self.data_df.iloc[idx]
        rxn_smiles = row['canonical_rxn']
        # Data augmentation
        if self.split == 'train' and self.args.shuffle_smiles:
            rxn_smiles = random_shuffle_reaction_smiles(rxn_smiles)
        nn_text = self.get_neighbor_text(idx) if self.args.num_neighbors > 0 else None
        enc_input = self.enc_tokenizer(
            rxn_smiles, text_pair=nn_text, truncation=False, return_token_type_ids=False, verbose=False)
        return enc_input

    def prepare_decoder_input(self, idx):
        row = self.data_df.iloc[idx]
        dec_input = {}
        if self.split != 'test':
            conditions = row[CONDITION_COLS].tolist()
            dec_input = self.dec_tokenizer(conditions, return_token_type_ids=False)
        return dec_input


class RetrosynthesisDataset(BaseDataset):

    def __len__(self):
        if self.split == 'test' and self.args.test_each_neighbor:
            return len(self.data_df) * self.args.test_num_neighbors
        return len(self.data_df)

    def get_neighbor_text(self, idx, return_list=False):
        if self.split == 'test' and self.args.test_each_neighbor:
            rxn_id = self.indices[idx // self.args.test_num_neighbors]
            neighbors_ids = self.neighbors[rxn_id]
            nn_id = idx % self.args.test_num_neighbors
            selected_texts = [self.corpus[i] for i in neighbors_ids[nn_id:nn_id + self.args.num_neighbors]]
            return selected_texts if return_list \
                else ''.join([f' ({i}) {text}' for i, text in enumerate(selected_texts)])
        return super().get_neighbor_text(idx, return_list)

    def prepare_encoder_input(self, idx):
        if self.split == 'test' and self.args.test_each_neighbor:
            row = self.data_df.iloc[idx // self.args.test_num_neighbors]
        else:
            row = self.data_df.iloc[idx]
        product_smiles = row['product_smiles']
        # Data augmentation
        if self.split == 'train' and self.args.shuffle_smiles:
            product_smiles = random_smiles(product_smiles)
        nn_text = self.get_neighbor_text(idx) if self.args.num_neighbors > 0 else None
        enc_input = self.enc_tokenizer(
            product_smiles, text_pair=nn_text, truncation=False, return_token_type_ids=False, verbose=False)
        return enc_input

    def prepare_decoder_input(self, idx):
        dec_input = {}
        if self.split != 'test':
            row = self.data_df.iloc[idx]
            dec_input = self.dec_tokenizer(
                row['reactant_smiles'], truncation=False,  return_token_type_ids=False, verbose=False)
        return dec_input


class DataCollator:

    def __init__(self, args, enc_tokenizer, dec_tokenizer, return_label=True):
        self.args = args
        self.enc_tokenizer = enc_tokenizer
        self.dec_tokenizer = dec_tokenizer
        self.return_label = return_label

    def __call__(self, features):
        indices = [feat[0] for feat in features]
        inputs = [feat[1] for feat in features]
        outputs = [feat[2] for feat in features]
        batch_in = {
            'input_ids': self.pad_sequences([feat['input_ids'] for feat in inputs], self.enc_tokenizer.pad_token_id),
            'attention_mask': self.pad_sequences([feat['attention_mask'] for feat in inputs], 0)
        }
        if 'position_ids' in inputs[0]:
            batch_in['position_ids'] = self.pad_sequences([feat['position_ids'] for feat in inputs], 0)
        if self.return_label:
            batch_in['decoder_input_ids'] = self.pad_sequences([feat['decoder_input_ids'] for feat in inputs],
                                                               self.dec_tokenizer.pad_token_id)
            batch_in['decoder_attention_mask'] = self.pad_sequences([feat['decoder_attention_mask'] for feat in inputs],
                                                                    0)
        batch_out = {}
        if 'mlm_labels' in outputs[0]:
            batch_out['mlm_labels'] = self.pad_sequences([feat['mlm_labels'] for feat in outputs], pad_id=-100)
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
