import json
import os

import numpy as np
import torch
import torch.nn as nn
import logging

from rdkit import Chem


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


""" Adapted from localretro_model/get_edit.py """

def get_id_template(a, class_n, num_atoms, edit_type, python_numbers=False):
    edit_idx = a // class_n
    template = a % class_n
    if edit_type == 'b':
        edit_idx = (edit_idx // num_atoms, edit_idx % num_atoms)
    if python_numbers:
        edit_idx = edit_idx.item() if edit_type == 'a' else (edit_idx[0].item(), edit_idx[1].item())
        template = template.item()
    return edit_idx, template

def output2edit(out, top_num, edit_type, bonds=None):
    num_atoms, class_n = out.shape[-2:]
    readout = out.cpu().detach().numpy()
    readout = readout.reshape(-1)
    output_rank = np.flip(np.argsort(readout))
    filtered_output_rank = []
    for r in output_rank:
        edit_idx, template = get_id_template(r, class_n, num_atoms, edit_type)
        if (bonds is None or edit_idx in bonds) and template != 0:
            filtered_output_rank.append(r)
            if len(filtered_output_rank) == top_num:
                break
    selected_edit = [get_id_template(a, class_n, num_atoms, edit_type, python_numbers=True) for a in filtered_output_rank]
    selected_proba = [readout[a].item() for a in filtered_output_rank]
     
    return selected_edit, selected_proba
    
def combined_edit(atom_out, bond_out, bonds, top_num=None):
    edit_id_a, edit_proba_a = output2edit(atom_out, top_num, edit_type='a')
    edit_id_b, edit_proba_b = output2edit(bond_out, top_num, edit_type='b', bonds=bonds)
    edit_id_c = edit_id_a + edit_id_b
    edit_type_c = ['a'] * len(edit_proba_a) + ['b'] * len(edit_proba_b)
    edit_proba_c = edit_proba_a + edit_proba_b
    edit_rank_c = np.flip(np.argsort(edit_proba_c))
    if top_num is not None:
        edit_rank_c = edit_rank_c[:top_num]
    edit_preds_c = [(edit_type_c[r], *edit_id_c[r]) for r in edit_rank_c]
    # edit_id_c = [edit_id_c[r] for r in edit_rank_c]
    edit_proba_c = [edit_proba_c[r] for r in edit_rank_c]
    
    # return edit_type_c, edit_id_c, edit_proba_c
    return edit_preds_c, edit_proba_c
