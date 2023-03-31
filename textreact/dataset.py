import json
import random
import pandas as pd
import rdkit.Chem as Chem
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
                selected_texts = neighbors_texts[:self.args.num_neighbors]
            nn_text = ' '.join(selected_texts)
        else:
            nn_text = None
        enc_input = self.enc_tokenizer(
            rxn_smiles, text_pair=nn_text, truncation=False, return_token_type_ids=False, verbose=False)
        max_length = self.args.max_length
        enc_input = {k: v[:max_length] for k, v in enc_input.items()}
        if self.split == 'test':
            return idx, enc_input
        conditions = row[CONDITION_COLS].tolist()
        dec_input = self.dec_tokenizer(conditions, return_token_type_ids=False)
        return idx, enc_input, dec_input


class ReactionConditionCollator:

    def __init__(self, args, enc_tokenizer, return_label=True):
        self.args = args
        self.enc_collator = DataCollatorWithPadding(tokenizer=enc_tokenizer, padding='longest')
        self.return_label = return_label
        if return_label:
            self.dec_collator = DefaultDataCollator()

    def __call__(self, features):
        indices = [feat[0] for feat in features]
        batch = self.enc_collator([feat[1] for feat in features])
        if self.return_label:
            dec_batch = self.dec_collator([feat[2] for feat in features])
            for key, value in dec_batch.items():
                batch['decoder_' + key] = value
        return indices, batch


def read_corpus(corpus_file):
    corpus_df = pd.read_csv(corpus_file, keep_default_na=False)
    corpus = {}
    for i, row in corpus_df.iterrows():
        if len(row['heading_text']) > 0:
            corpus[row['id']] = row['heading_text'] + '. ' + row['paragraph_text']
        else:
            corpus[row['id']] = row['paragraph_text']
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
