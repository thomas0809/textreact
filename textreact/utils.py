import json
import os

import torch
import torch.nn as nn
import logging


metric_to_mode = {
    'val_loss': 'min',
    'val_acc': 'max'
}


def expand_position_embeddings(encoder, max_length):
    if encoder.config.model_type in ['bert', 'longformer', 'roberta']:
        if max_length <= encoder.config.max_position_embeddings:
            return
        embeddings = encoder.embeddings
        config = encoder.config
        old_emb = embeddings.position_embeddings.weight.data.clone()
        config.max_position_embeddings = max_length
        embeddings.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        embeddings.position_embeddings.weight.data[:old_emb.size(0)] = old_emb
        embeddings.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        embeddings.register_buffer(
            "token_type_ids", torch.zeros((1, config.max_position_embeddings), dtype=torch.long), persistent=False)
    else:
        raise NotImplementedError


def expand_word_embeddings(encoder, vocab_size):
    if vocab_size <= encoder.config.vocab_size:
        return
    embeddings = encoder.embeddings
    config = encoder.config
    old_emb = embeddings.word_embeddings.weight.data.clone()
    config.vocab_size = vocab_size
    embeddings.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
    embeddings.word_embeddings.weight.data[:old_emb.size(0)] = old_emb


# def share_embedding(model):
#     if model.decoder.config.model_type == 'roberta':
#         model.decoder.roberta.embeddings.word_embeddings.weight = model.encoder.embeddings.word_embeddings.weight
#         model.decoder.lm_head.decoder.weight = model.encoder.embeddings.word_embeddings.weight
#     else:
#         raise NotImplementedError


def clear_path(path, trainer):
    for file in os.listdir(path):
        if file.endswith('.ckpt'):
            filepath = os.path.join(path, file)
            logging.info(f' Remove checkpoint {filepath}')
            trainer.strategy.remove_checkpoint(filepath)


def gather_prediction_each_neighbor(prediction, num_neighbors):
    results = {}
    for i, pred in sorted(prediction.items()):
        idx = i // num_neighbors
        if i % num_neighbors == 0:
            results[idx] = pred
        else:
            for key in results[idx]:
                results[idx][key] += pred[key]
    return results
