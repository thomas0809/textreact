import collections
import os
import re
from typing import List
from transformers import PreTrainedTokenizer, AutoTokenizer, BertTokenizer


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")
        vocab[token] = index
    return vocab


class ReactionConditionTokenizer(PreTrainedTokenizer):

    def __init__(self, vocab_file):
        super().__init__(
            pad_token='[PAD]',
            bos_token='[BOS]',
            eos_token='[EOS]',
            mask_token='[MASK]',
            unk_token='[UNK]',
            sep_token='[SEP]'
        )
        self.vocab = load_vocab(vocab_file)
        self.ids_to_tokens = {ids: tok for tok, ids in self.vocab.items()}

    def __len__(self):
        return len(self.vocab)

    def __call__(self, conditions, **kwargs):
        tokens = self.convert_tokens_to_ids(conditions)
        return self.prepare_for_model(tokens, **kwargs)

    def _decode(self, token_ids, skip_special_tokens=False, **kwargs):
        tokens = self.convert_ids_to_tokens(token_ids, skip_special_tokens=skip_special_tokens)
        return tokens

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        token_ids = token_ids_0
        return [self.bos_token_id] + token_ids + [self.eos_token_id]

    def create_token_type_ids_from_sequences(self, token_ids_0, token_ids_1=None):
        token_ids = token_ids_0
        return [0] * (len(token_ids) + 2)

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.ids_to_tokens.get(index, self.unk_token)


SMI_REGEX_PATTERN = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#" \
                    r"|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"


class SmilesTokenizer(BertTokenizer):
    """
    Creates the SmilesTokenizer class. The tokenizer heavily inherits from the BertTokenizer
    implementation found in Huggingface's transformers library. It runs a WordPiece tokenization
    algorithm over SMILES strings using the tokenisation SMILES regex developed by Schwaller et. al.
    Please see https://github.com/huggingface/transformers and https://github.com/rxn4chemistry/rxnfp for more details.

    This class requires huggingface's transformers and tokenizers libraries to be installed.
    """

    def __init__(self, vocab_file: str = '', **kwargs):
        """Constructs a SmilesTokenizer.
        Parameters
        ----------
        vocab_file: str
            Path to a SMILES character per line vocabulary file.
            Default vocab file is found in deepchem/feat/tests/data/vocab_smiles.txt
        """

        super().__init__(vocab_file, bos_token='[CLS]', eos_token='[SEP]', **kwargs)

        if not os.path.isfile(vocab_file):
            raise ValueError("Can't find a vocab file at path '{}'.".format(vocab_file))
        self.vocab = load_vocab(vocab_file)
        self.highest_unused_index = max(
            [i for i, v in enumerate(self.vocab.keys()) if v.startswith("[unused")])
        self.ids_to_tokens = collections.OrderedDict(
            [(ids, tok) for tok, ids in self.vocab.items()])
        self.basic_tokenizer = BasicSmilesTokenizer()

    @property
    def vocab_size(self):
        return len(self.vocab)

    @property
    def vocab_list(self):
        return list(self.vocab.keys())

    def _tokenize(self, text: str):
        """
        Tokenize a string into a list of tokens.
        Parameters
        ----------
        text: str
            Input string sequence to be tokenized.
        """
        split_tokens = [token for token in self.basic_tokenizer.tokenize(text)]
        return split_tokens

    def _convert_token_to_id(self, token):
        """
        Converts a token (str/unicode) in an id using the vocab.
        Parameters
        ----------
        token: str
            String token from a larger sequence to be converted to a numerical id.
        """
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """
        Converts an index (integer) in a token (string/unicode) using the vocab.
        Parameters
        ----------
        index: int
            Integer index to be converted back to a string-based token as part of a larger sequence.
        """
        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens: List[str]):
        """ Converts a sequence of tokens (string) in a single string.
        Parameters
        ----------
        tokens: List[str]
            List of tokens for a given string sequence.
        Returns
        -------
        out_string: str
            Single string from combined tokens.
        """
        out_string: str = "".join(tokens).replace(" ##", "").strip()
        return out_string

    def add_special_tokens_ids_single_sequence(self, token_ids: List[int]):
        """
        Adds special tokens to the a sequence for sequence classification tasks.
        A BERT sequence has the following format: [CLS] X [SEP]
        Parameters
        ----------
        token_ids: list[int]
            list of tokenized input ids. Can be obtained using the encode or encode_plus methods.
        """
        return [self.cls_token_id] + token_ids + [self.sep_token_id]

    def add_special_tokens_single_sequence(self, tokens: List[str]):
        """
        Adds special tokens to the a sequence for sequence classification tasks.
        A BERT sequence has the following format: [CLS] X [SEP]
        Parameters
        ----------
        tokens: List[str]
            List of tokens for a given string sequence.
        """
        return [self.cls_token] + tokens + [self.sep_token]

    def add_special_tokens_ids_sequence_pair(self, token_ids_0: List[int],
                                             token_ids_1: List[int]) -> List[int]:
        """
        Adds special tokens to a sequence pair for sequence classification tasks.
        A BERT sequence pair has the following format: [CLS] A [SEP] B [SEP]
        Parameters
        ----------
        token_ids_0: List[int]
            List of ids for the first string sequence in the sequence pair (A).
        token_ids_1: List[int]
            List of tokens for the second string sequence in the sequence pair (B).
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        return cls + token_ids_0 + sep + token_ids_1 + sep

    def add_padding_tokens(self,
                           token_ids: List[int],
                           length: int,
                           right: bool = True) -> List[int]:
        """
        Adds padding tokens to return a sequence of length max_length.
        By default padding tokens are added to the right of the sequence.
        Parameters
        ----------
        token_ids: list[int]
            list of tokenized input ids. Can be obtained using the encode or encode_plus methods.
        length: int
        right: bool (True by default)
        Returns
        ----------
        token_ids :
            list of tokenized input ids. Can be obtained using the encode or encode_plus methods.
        padding: int
            Integer to be added as padding token
        """
        padding = [self.pad_token_id] * (length - len(token_ids))

        if right:
            return token_ids + padding
        else:
            return padding + token_ids


class BasicSmilesTokenizer(object):
    """
    Run basic SMILES tokenization using a regex pattern developed by Schwaller et. al. This tokenizer is to be used
    when a tokenizer that does not require the transformers library by HuggingFace is required.
    """

    def __init__(self, regex_pattern: str = SMI_REGEX_PATTERN):
        """ Constructs a BasicSMILESTokenizer. """
        self.regex_pattern = regex_pattern
        self.regex = re.compile(self.regex_pattern)

    def tokenize(self, text):
        """ Basic Tokenization of a SMILES. """
        tokens = [token for token in self.regex.findall(text)]
        return tokens


class SmilesTextTokenizer(PreTrainedTokenizer):

    def __init__(self, text_tokenizer, smiles_tokenizer=None):
        super().__init__(
            pad_token=text_tokenizer.pad_token,
            mask_token=text_tokenizer.mask_token)
        if smiles_tokenizer is None:
            self.separate = False
            self.smiles_tokenizer = text_tokenizer
        else:
            self.separate = True
            self.smiles_tokenizer = smiles_tokenizer
        self.text_tokenizer = text_tokenizer

    @property
    def smiles_offset(self):
        return len(self.text_tokenizer) if self.separate is not None else 0

    def __len__(self):
        return len(self.text_tokenizer) + self.smiles_offset

    def __call__(self, text, text_pair, **kwargs):
        result = self.smiles_tokenizer(text, **kwargs)
        if self.separate:
            result['input_ids'] = [v + self.smiles_offset for v in result['input_ids']]
        if isinstance(text_pair, str):
            result_pair = self.text_tokenizer(text_pair, **kwargs)
            for key in result:
                result[key] = result[key] + result_pair[key][1:]  # skip the CLS token
        elif isinstance(text_pair, list):
            for t in text_pair:
                result_pair = self.text_tokenizer(t, **kwargs)
                for key in result:
                    result[key] = result[key] + result_pair[key][1:]  # skip the CLS token
        return result

    def _convert_id_to_token(self, index):
        if index < len(self.text_tokenizer):
            return self.text_tokenizer.convert_ids_to_tokens(index)
        else:
            return self.smiles_tokenizer.convert_ids_to_tokens(index - len(self.text_tokenizer))

    def _convert_token_to_id(self, token):
        return self.text_tokenizer.convert_tokens_to_ids(token)


def get_tokenizers(args):
    # Encoder
    if args.encoder_tokenizer == 'smiles':
        enc_tokenizer = SmilesTokenizer(args.vocab_file)
    elif args.encoder_tokenizer == 'text':
        text_tokenizer = AutoTokenizer.from_pretrained(args.encoder, use_fast=False)
        enc_tokenizer = SmilesTextTokenizer(text_tokenizer)
    elif args.encoder_tokenizer == 'smiles_text':
        smiles_tokenizer = SmilesTokenizer(args.vocab_file)
        text_tokenizer = AutoTokenizer.from_pretrained(args.encoder, use_fast=False)
        enc_tokenizer = SmilesTextTokenizer(text_tokenizer, smiles_tokenizer)
    else:
        raise ValueError
    # Decoder
    if args.task == 'condition':
        dec_tokenizer = ReactionConditionTokenizer(args.vocab_file)
    elif args.task == 'retro':
        dec_tokenizer = SmilesTokenizer(args.vocab_file)
    else:
        raise ValueError
    return enc_tokenizer, dec_tokenizer
