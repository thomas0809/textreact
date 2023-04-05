import json
import os

import torch
import torch.nn as nn
import logging


metric_to_mode = {
    'val_loss': 'min',
    'val_acc': 'max'
}


def expand_max_length(encoder, max_length):
    if encoder.config.model_type in ['bert', 'longformer']:
        if max_length <= encoder.config.max_position_embeddings:
            return
        embeddings = encoder.embeddings
        config = encoder.config
        old_emb = embeddings.position_embeddings.weight.data.clone()
        config.max_position_embeddings = 1024
        embeddings.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        embeddings.position_embeddings.weight.data[:old_emb.size(0)] = old_emb
        embeddings.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        embeddings.register_buffer(
            "token_type_ids", torch.zeros((1, config.max_position_embeddings), dtype=torch.long), persistent=False)
    else:
        logging.warning(f' Cannot expand the max_length of {encoder.config.model_type} models')


def clear_path(path, trainer):
    for file in os.listdir(path):
        if file.endswith('.ckpt'):
            filepath = os.path.join(path, file)
            logging.info(f' Remove checkpoint {filepath}')
            trainer.remove_checkpoint(filepath)
