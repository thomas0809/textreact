from transformers import PreTrainedTokenizer, AutoTokenizer


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = {}
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


# class ReactionInputTokenizer(AutoTokenizer):
#
#     def truncate_sequences(self, ids, pair_ids=None, num_tokens_to_remove=0, truncation_strategy="longest_first", stride=0):
#         if num_tokens_to_remove > 0:
#             if pair_ids is not None:
#                 num = max(len(pair_ids), num_tokens_to_remove)
#                 pair_ids = pair_ids[:-num]
#                 num_tokens_to_remove -= num
#             if num_tokens_to_remove > 0:
#                 ids = ids[:-num_tokens_to_remove]
#         return ids, pair_ids, []
