from collections import defaultdict
from functools import partial
import json
import logging
import os
import pickle
import uuid

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import torch
import seaborn as sns
from IPython.core.display import display, HTML, Javascript
from torch.utils.data import Dataset
from bertviz.util import format_attention, num_layers
import pkg_resources
from tqdm import tqdm
from simpletransformers.classification.classification_utils import preprocess_data_multiprocessing, preprocess_data
from multiprocessing import Pool
from typing import List, Tuple
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizer
from rdkit import Chem
from rdkit.Chem.Draw import SimilarityMaps
from matplotlib import pyplot as plt
import pandas as pd
from rxnfp.tokenization import RegexTokenizer

from preprocess_script.uspto_script.utils import canonicalize_smiles

logger = logging.getLogger(__name__)

BAD_TOKS = ["[CLS]", "[SEP]"]  # Default Bad Tokens
CONDITION_TYPE = ['c1', 's1', 's2', 'r1', 'r2']
def caonicalize_rxn_smiles(rxn_smiles):
    try:
        react, _, prod = rxn_smiles.split('>')
        react, prod = [canonicalize_smiles(x) for x in [react, prod]]
        if '' in [react, prod]:
            return ''
        return f'{react}>>{prod}'
    except:
        return ''


def get_output_results(input_rxn_smiles, pred_conditions, pred_temperatures, output_dataframe=True):
    output_results = []
    output_df = pd.DataFrame()
    for idx, one_pred in enumerate(pred_conditions):
        conditions, scores = zip(*one_pred)
        one_df = pd.DataFrame(conditions)
        one_df.columns = [
            'catalyst1', 'solvent1', 'solvent2', 'reagent1', 'reagent2'
        ]
        one_df['scores'] = scores
        one_df['rxn_smiles'] = [input_rxn_smiles[idx]
                                ] + [''] * (len(conditions) - 1)
        one_df['top-k'] = [f'top-{x+1}' for x in range(len(conditions)) ]
        one_df = one_df[[
            'rxn_smiles', 'top-k', 'catalyst1', 'solvent1', 'solvent2', 'reagent1',
            'reagent2', 'scores'
        ]]
        if pred_temperatures:
            one_df['temperatures'] = [pred_temperatures[idx]
                                      ] + [''] * (len(conditions) - 1)
            one_df = one_df[[
                'rxn_smiles', 'top-k', 'catalyst1', 'solvent1', 'solvent2', 'reagent1',
                'reagent2', 'temperatures', 'scores'
            ]]
        one_df = one_df.round(5)
        output_df = output_df.append(one_df)
        output_results.append(one_df)
    if output_dataframe:
        output_df = output_df.reset_index(drop=True)
        return output_df
    else:
        return output_results

def load_dataset(dataset_root, database_fname, use_temperature=False):
    csv_fpath = os.path.abspath(os.path.join(dataset_root, database_fname))
    print('Reading database csv from {}...'.format(csv_fpath))
    database = pd.read_csv(csv_fpath)

    all_idx_mapping_data_fpath = os.path.join(
        dataset_root, '{}_alldata_idx.pkl'.format(database_fname.split('.')[0]))
    print('Reading index-condition mapping data from {}'.format(all_idx_mapping_data_fpath))
    with open(all_idx_mapping_data_fpath, 'rb') as f:
        all_idx2data, all_data2idx = pickle.load(f)

    all_condition_labels_fpath = os.path.join(
        dataset_root, '{}_condition_labels.pkl'.format(database_fname.split('.')[0]))
    if not os.path.exists(all_condition_labels_fpath):
        condition_cols = ['catalyst1', 'solvent1',
                          'solvent2', 'reagent1', 'reagent2']
        all_condition_labels = []
        for _, row in tqdm(database[condition_cols].iterrows(), total=len(database)):
            row.loc[pd.isna(row)] = ''
            row = list(row)
            row = ['[BOS]'] + row + ['[EOS]']
            all_condition_labels.append([all_data2idx[x] for x in row])
        assert(len(database) == len(all_condition_labels))
    else:
        with open(all_condition_labels_fpath, 'rb') as f:
            all_condition_labels = pickle.load(f)

    assert len(database) == len(all_condition_labels)

    if use_temperature:
        print('Collecting tempertures data ...')
        all_condition_labels_array = np.array(all_condition_labels)
        all_condition_labels = np.concatenate(
            [
                all_condition_labels_array,
                database['temperature'].values.reshape(-1, 1)
            ],
            axis=1).tolist()
    database['condition_labels'] = all_condition_labels
    condition_label_mapping = (all_idx2data, all_data2idx)
    return database, condition_label_mapping

def load_test_dataset(dataset_root, database_fname, condition_label_mapping, use_temperature=False):
    csv_fpath = os.path.abspath(os.path.join(dataset_root, database_fname))
    print('Reading test dataset csv from {}...'.format(csv_fpath))
    database = pd.read_csv(csv_fpath)


    all_idx2data, all_data2idx = condition_label_mapping
    condition_cols = ['catalyst1', 'solvent1',
                        'solvent2', 'reagent1', 'reagent2']
    all_condition_labels = []
    for _, row in tqdm(database[condition_cols].iterrows(), total=len(database)):
        row.loc[pd.isna(row)] = ''
        row = list(row)
        row = ['[BOS]'] + row + ['[EOS]']
        all_condition_labels.append([all_data2idx[x] for x in row])
    assert(len(database) == len(all_condition_labels))

    if use_temperature:
        print('Collecting tempertures data ...')
        all_condition_labels_array = np.array(all_condition_labels)
        all_condition_labels = np.concatenate(
            [
                all_condition_labels_array,
                database['temperature'].values.reshape(-1, 1)
            ],
            axis=1).tolist()
    database['condition_labels'] = all_condition_labels
    condition_label_mapping = (all_idx2data, all_data2idx)
    return database, condition_label_mapping

def inference_load(dataset_root, database_fname, use_temperature):
    csv_fpath = os.path.abspath(os.path.join(dataset_root, database_fname))
    all_idx_mapping_data_fpath = os.path.join(
        dataset_root, '{}_alldata_idx.pkl'.format(database_fname.split('.')[0]))
    print('Reading index-condition mapping data from {}'.format(all_idx_mapping_data_fpath))
    with open(all_idx_mapping_data_fpath, 'rb') as f:
        all_idx2data, all_data2idx = pickle.load(f)

    condition_label_mapping = (all_idx2data, all_data2idx)
    return condition_label_mapping


def print_args(config_file):
    print('#'*30)
    with open(config_file, 'r', encoding='utf-8') as f:
        print(''.join([x for x in f.readlines()]))
    print('#'*30)
    print()


def condition_bert_head_view(
    attention=None,
    src_tokens=None,
    tgt_tokens=None,
    html_action: str = 'view',    # view or return
):
    attention = [x.unsqueeze(0) for x in attention]
    attn_data = []
    include_layers = list(range(num_layers(attention)))
    attention = format_attention(attention, include_layers)
    attn_data.append(
        {
            'name': None,
            'attn': attention.tolist(),
            'left_text': src_tokens,
            'right_text': tgt_tokens
        }
    )

    vis_id = 'bertviz-%s' % (uuid.uuid4().hex)
    select_html = ""
    vis_html = f"""      
        <div id="{vis_id}" style="font-family:'Helvetica Neue', Helvetica, Arial, sans-serif;">
            <span style="user-select:none">
                Layer: <select id="layer"></select>
                {select_html}
            </span>
            <div id='vis'></div>
        </div>
    """
    params = {
        'attention': attn_data,
        'default_filter': "0",
        'root_div_id': vis_id,
        'layer': None,
        'heads': None,
        'include_layers': include_layers
    }

    pkg_path = pkg_resources.resource_filename("bertviz", ".")
    if html_action == 'view':
        display(HTML(
            '<script src="https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.6/require.min.js"></script>'))
        display(HTML(vis_html))
        __location__ = os.path.realpath(
            os.path.join(os.getcwd(), pkg_path))
        vis_js = open(os.path.join(__location__, 'head_view.js')
                      ).read().replace("PYTHON_PARAMS", json.dumps(params))
        display(Javascript(vis_js))
    elif html_action == 'return':
        html1 = HTML(
            '<script src="https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.6/require.min.js"></script>')
        html2 = HTML(vis_html)
        __location__ = os.path.dirname(os.path.abspath(__file__))
        vis_js = open(os.path.join(__location__, 'data', 'bert_condition_data',
                      'head_view.js')).read().replace("PYTHON_PARAMS", json.dumps(params))
        html3 = Javascript(vis_js)
        script = '\n<script type="text/javascript">\n' + html3.data + '\n</script>\n'
        head_html = HTML(html1.data + html2.data + script)
        return head_html


def condition_bert_head_heatmap(attention, src_tokens, tgt_tokens, fig_save_path=None, split_map=True, figsize=(46, 20)):
    print('src_tokens', src_tokens)
    print('tgt_tokens', tgt_tokens)
    sns.set(style='ticks', font_scale=1.5)
    
    attention = [x.unsqueeze(0) for x in attention]

    assert isinstance(attention, tuple) or isinstance(attention, list)
    assert isinstance(attention[0], torch.Tensor)

    assert attention[0].size(2) == len(src_tokens)
    assert attention[0].size(3) == len(tgt_tokens)

    if split_map:

        for i, layer_attention in enumerate(attention):
            layer_attention = layer_attention.squeeze()
            for head_idx in range(layer_attention.size(0)):
                slice_attention = layer_attention[head_idx].transpose(0, 1)
                slice_attention = slice_attention.to(torch.device('cpu'))
                df = pd.DataFrame(slice_attention.detach().numpy(),
                                index=tgt_tokens, columns=src_tokens)
                # data_list.append(df)
                x_major_locator = MultipleLocator(1)
                fig, axes = plt.subplots(nrows=1,
                                ncols=1, figsize=figsize)
                sns.heatmap(
                    df,
                    # cmap="rainbow",
                    # ax=axes[i][head_idx],
                    cbar=True,
                    # cbar_ax=None if i else cbar_ax
                )
                axes.set_title(
                    'Layer {}, Head {}'.format(i+1, head_idx+1), fontsize=30)
                axes.xaxis.tick_top()
                axes.xaxis.set_major_locator(x_major_locator)
                axes.set_xticklabels([' '] + src_tokens + [' '])
                plt.xticks(fontsize=8, rotation=90)
                plt.yticks(fontsize=8)
                # display(fig)
                if fig_save_path:
                    fig.savefig('{}_layer{}_head{}.svg'.format(fig_save_path.replace('.svg', ''), i+1, head_idx+1), format="svg", bbox_inches="tight")

    else:
        fig, axes = plt.subplots(nrows=len(attention),
                                ncols=attention[0].size(1), figsize=figsize)
        # data_list = []
        cbar_ax = fig.add_axes([1.02, .3, .03, .4])

        x_major_locator = MultipleLocator(1)
        for i, layer_attention in enumerate(attention):
            layer_attention = layer_attention.squeeze()
            for head_idx in range(layer_attention.size(0)):
                slice_attention = layer_attention[head_idx].transpose(0, 1)
                slice_attention = slice_attention.to(torch.device('cpu'))
                df = pd.DataFrame(slice_attention.detach().numpy(),
                                index=tgt_tokens, columns=src_tokens)
                # data_list.append(df)
                sns.heatmap(
                    df,
                    # cmap="rainbow",
                    ax=axes[i][head_idx],
                    cbar=i == 0,
                    cbar_ax=None if i else cbar_ax
                )
                axes[i][head_idx].set_title(
                    'Layer {}, Head {}'.format(i+1, head_idx+1), fontsize=30)
                axes[i][head_idx].xaxis.tick_top()
                axes[i][head_idx].xaxis.set_major_locator(x_major_locator)
                axes[i][head_idx].set_xticklabels([' '] + src_tokens + [' '])
        # plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

        # plt.show()
        # display(fig)
        if fig_save_path:
            fig.savefig(fig_save_path, format="svg", bbox_inches="tight")


def build_classification_dataset(
    data, tokenizer, args, mode, multi_label, output_mode, no_cache
):
    cached_features_file = os.path.join(
        args.cache_dir,
        "cached_{}_{}_{}_{}_{}".format(
            mode,
            args.model_type,
            args.max_seq_length,
            len(args.labels_list),
            len(data),
        ),
    )

    if os.path.exists(cached_features_file) and (
        (not args.reprocess_input_data and not args.no_cache)
        or (mode == "dev" and args.use_cached_eval_features and not args.no_cache)
    ):
        data = torch.load(cached_features_file)
        logger.info(f" Features loaded from cache at {cached_features_file}")
        examples, labels = data
    else:
        logger.info(" Converting to features started. Cache is not used.")

        if len(data) == 3:
            # Sentence pair task
            text_a, text_b, labels = data
        else:
            text_a, labels = data
            text_b = None

        # If labels_map is defined, then labels need to be replaced with ints
        if args.labels_map and not args.regression:
            if multi_label:
                labels = [[args.labels_map[l] for l in label]
                          for label in labels]
            else:
                labels = [args.labels_map[label] for label in labels]

        if (mode == "train" and args.use_multiprocessing) or (
            mode == "dev" and args.use_multiprocessing_for_evaluation
        ):
            if args.multiprocessing_chunksize == -1:
                chunksize = max(len(data) // (args.process_count * 2), 500)
            else:
                chunksize = args.multiprocessing_chunksize

            if text_b is not None:
                data = [
                    (
                        text_a[i: i + chunksize],
                        text_b[i: i + chunksize],
                        tokenizer,
                        args.max_seq_length,
                    )
                    for i in range(0, len(text_a), chunksize)
                ]
            else:
                data = [
                    (text_a[i: i + chunksize], None,
                     tokenizer, args.max_seq_length)
                    for i in range(0, len(text_a), chunksize)
                ]

            with Pool(args.process_count) as p:
                examples = list(
                    tqdm(
                        p.imap(preprocess_data_multiprocessing, data),
                        total=len(text_a),
                        disable=args.silent,
                    )
                )

            examples = {
                key: torch.cat([example[key] for example in examples])
                for key in examples[0]
            }
        else:
            examples = preprocess_data(
                text_a, text_b, labels, tokenizer, args.max_seq_length
            )
        if not args.use_temperature:
            if output_mode == "classification":
                labels = torch.tensor(labels, dtype=torch.long)
            elif output_mode == "regression":
                labels = torch.tensor(labels, dtype=torch.float)
            data = (examples, labels)
        else:
            labels = torch.tensor(labels)
            condition_labels = labels[:, :-1].long()
            temperature = labels[:, -1:].float()

            data = (examples, (condition_labels, temperature))

        if not args.no_cache and not no_cache:
            logger.info(" Saving features into cached file %s",
                        cached_features_file)
            torch.save(data, cached_features_file)

    return data


class ConditionWithTempDataset(Dataset):
    def __init__(self, data, tokenizer, args, mode, multi_label, output_mode, no_cache):
        self.examples, self.labels = build_classification_dataset(
            data, tokenizer, args, mode, multi_label, output_mode, no_cache
        )

    def __len__(self):
        return len(self.examples["input_ids"])

    def __getitem__(self, index):
        return (
            {key: self.examples[key][index] for key in self.examples},
            (self.labels[0][index], self.labels[1][index]),
        )


def encode(data):
    tokenizer, line = data
    return tokenizer.encode(line)


def encode_sliding_window(data):
    tokenizer, line, max_seq_length, special_tokens_count, stride, no_padding = data

    tokens = tokenizer.tokenize(line)
    stride = int(max_seq_length * stride)
    token_sets = []
    if len(tokens) > max_seq_length - special_tokens_count:
        token_sets = [
            tokens[i: i + max_seq_length - special_tokens_count]
            for i in range(0, len(tokens), stride)
        ]
    else:
        token_sets.append(tokens)

    features = []
    if not no_padding:
        sep_token = tokenizer.sep_token_id
        cls_token = tokenizer.cls_token_id
        pad_token = tokenizer.pad_token_id

        for tokens in token_sets:
            tokens = [cls_token] + tokens + [sep_token]

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            padding_length = max_seq_length - len(input_ids)
            input_ids = input_ids + ([pad_token] * padding_length)

            assert len(input_ids) == max_seq_length

            features.append(input_ids)
    else:
        for tokens in token_sets:
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            features.append(input_ids)

    return features


def preprocess_batch_for_hf_dataset(dataset, tokenizer, max_seq_length):
    return tokenizer(
        text=dataset["text"],
        truncation=True,
        padding="max_length",
        max_length=max_seq_length,
    )


def load_hf_dataset(data, tokenizer, args):
    dataset = load_dataset(
        "text",
        data_files=data,
        download_mode="force_redownload"
        if args.reprocess_input_data
        else "reuse_dataset_if_exists",
    )

    dataset = dataset.map(
        lambda x: preprocess_batch_for_hf_dataset(
            x, tokenizer=tokenizer, max_seq_length=args.max_seq_length
        ),
        batched=True,
    )

    dataset.set_format(type="pt", columns=["input_ids"])

    if isinstance(data, str):
        # This is not necessarily a train dataset. The datasets library insists on calling it train.
        return dataset["train"]
    else:
        return dataset


class SimpleCenterDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        args,
        file_path,
        mode,
        block_size=512,
        special_tokens_count=2,
        sliding_window=False,
    ):
        assert os.path.isfile(file_path)
        block_size = block_size - special_tokens_count
        directory, filename = os.path.split(file_path)
        rxn_center_file_path = file_path.replace('rxn', 'rxn_center')
        cached_features_file = os.path.join(
            args.cache_dir,
            args.model_type + "_cached_lm_" + str(block_size) + "_" + filename,
        )

        if os.path.exists(cached_features_file) and (
            (not args.reprocess_input_data and not args.no_cache)
            or (mode == "dev" and args.use_cached_eval_features and not args.no_cache)
        ):
            logger.info(" Loading features from cached file %s",
                        cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
        else:
            logger.info(
                " Creating features from dataset file at %s", args.cache_dir)

            if sliding_window:
                no_padding = (
                    True if args.model_type in [
                        "gpt2", "openai-gpt"] else False
                )
                with open(file_path, encoding="utf-8") as f:
                    lines = [
                        (
                            tokenizer,
                            line,
                            args.max_seq_length,
                            special_tokens_count,
                            args.stride,
                            no_padding,
                        )
                        for line in f.read().splitlines()
                        if (len(line) > 0 and not line.isspace())
                    ]

                if (mode == "train" and args.use_multiprocessing) or (
                    mode == "dev" and args.use_multiprocessing_for_evaluation
                ):
                    if args.multiprocessing_chunksize == -1:
                        chunksize = max(
                            len(lines) // (args.process_count * 2), 500)
                    else:
                        chunksize = args.multiprocessing_chunksize

                    with Pool(args.process_count) as p:
                        self.examples = list(
                            tqdm(
                                p.imap(
                                    encode_sliding_window, lines, chunksize=chunksize
                                ),
                                total=len(lines),
                                # disable=silent,
                            )
                        )
                else:
                    self.examples = [encode_sliding_window(
                        line) for line in lines]

                self.examples = [
                    example for example_set in self.examples for example in example_set
                ]
            else:
                with open(file_path, encoding="utf-8") as f:
                    lines = [
                        (tokenizer, line)
                        for line in f.read().splitlines()
                        if (len(line) > 0 and not line.isspace())
                    ]

                with open(rxn_center_file_path, encoding="utf-8") as fc:
                    center_lines = [
                        (tokenizer, line)
                        for line in fc.read().splitlines()
                        if (len(line) > 0 and not line.isspace())
                    ]

                # is_rxn_center_tokens = []
                # for (_, line), center_line in zip(lines, center_lines):
                #     tokenlize_line = tokenizer.tokenize(line)
                #     tokenlize_center_line = tokenizer.tokenize(center_line)
                #     if len(tokenlize_line) == len(tokenlize_center_line): # 应该是数量相等的，但是还是做了个判断，以防万一
                #         one_line_center_mark = []
                #         for rxn_token, c_rxn_token in zip(tokenlize_line, tokenlize_center_line):
                #             if rxn_token != c_rxn_token:
                #                 one_line_center_mark.append(1)
                #             else:
                #                 one_line_center_mark.append(0)

                #     else:
                #         is_rxn_center_tokens.append([0]*len(tokenlize_line))

                if args.use_multiprocessing:
                    if args.multiprocessing_chunksize == -1:
                        chunksize = max(
                            len(lines) // (args.process_count * 2), 500)
                    else:
                        chunksize = args.multiprocessing_chunksize

                    with Pool(args.process_count) as p:
                        self.examples = list(
                            tqdm(
                                p.imap(encode, lines, chunksize=chunksize),
                                total=len(lines),
                                # disable=silent,
                            )
                        )
                    with Pool(args.process_count) as p:
                        self.examples_with_center = list(
                            tqdm(
                                p.imap(encode, center_lines,
                                       chunksize=chunksize),
                                total=len(center_lines),
                                # disable=silent,
                            )
                        )
                else:
                    self.examples = [encode(line) for line in lines]
                    self.examples_with_center = [
                        encode(line) for line in center_lines]

                self.examples = [
                    token for tokens in self.examples for token in tokens]
                self.examples_with_center = [
                    token for tokens in self.examples_with_center for token in tokens]
                if len(self.examples) > block_size:
                    self.examples = [
                        tokenizer.build_inputs_with_special_tokens(
                            self.examples[i: i + block_size]
                        )
                        for i in tqdm(
                            range(0, len(self.examples) -
                                  block_size + 1, block_size)
                        )
                    ]
                    self.examples_with_center = [
                        tokenizer.build_inputs_with_special_tokens(
                            self.examples_with_center[i: i + block_size]
                        )
                        for i in tqdm(
                            range(0, len(self.examples_with_center) -
                                  block_size + 1, block_size)
                        )
                    ]
                else:
                    self.examples = [
                        tokenizer.build_inputs_with_special_tokens(
                            self.examples)
                    ]
                    self.examples_with_center = [
                        tokenizer.build_inputs_with_special_tokens(
                            self.examples_with_center)
                    ]
            self.is_rxn_center_tokens = (torch.tensor(
                self.examples) != torch.tensor(self.examples_with_center)).long()
            logger.info(" Saving features into cached file %s",
                        cached_features_file)
            with open(cached_features_file, "wb") as handle:
                pickle.dump(self.examples, handle,
                            protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item], dtype=torch.long), self.is_rxn_center_tokens[item]

class SimpleCenterIdxDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        args,
        dataset_df,
        mode,
        block_size=512,
        special_tokens_count=2,
        sliding_window=False,
    ):

        block_size = block_size - special_tokens_count
      
        if sliding_window:
            no_padding = (
                True if args.model_type in [
                    "gpt2", "openai-gpt"] else False
            )
            with open(dataset_df, encoding="utf-8") as f:
                lines = [
                    (
                        tokenizer,
                        line,
                        args.max_seq_length,
                        special_tokens_count,
                        args.stride,
                        no_padding,
                    )
                    for line in f.read().splitlines()
                    if (len(line) > 0 and not line.isspace())
                ]

            if (mode == "train" and args.use_multiprocessing) or (
                mode == "dev" and args.use_multiprocessing_for_evaluation
            ):
                if args.multiprocessing_chunksize == -1:
                    chunksize = max(
                        len(lines) // (args.process_count * 2), 500)
                else:
                    chunksize = args.multiprocessing_chunksize

                with Pool(args.process_count) as p:
                    self.examples = list(
                        tqdm(
                            p.imap(
                                encode_sliding_window, lines, chunksize=chunksize
                            ),
                            total=len(lines),
                            # disable=silent,
                        )
                    )
            else:
                self.examples = [encode_sliding_window(
                    line) for line in lines]

            self.examples = [
                example for example_set in self.examples for example in example_set
            ]
        else:
            # with open(dataset_df, encoding="utf-8") as f:
            #     lines = [
            #         (tokenizer, line)
            #         for line in f.read().splitlines()
            #         if (len(line) > 0 and not line.isspace())
            #     ]
            text_lines = dataset_df['text'].tolist()
            lines = [
                (tokenizer, line)
                for line in text_lines
                if (len(line) > 0 and not line.isspace())
            ]
            
            def pad_masks(mask_label):
                assert isinstance(mask_label, torch.Tensor)
                pad_tensor = torch.zeros(1).bool()
                return torch.cat([pad_tensor, mask_label, pad_tensor], dim=-1)

            mask_labels = [torch.from_numpy(np.array(x)) for x in dataset_df['labels'].tolist()]
            mask_labels_pad = [pad_masks(x) for x in mask_labels]

            if args.use_multiprocessing:
                if args.multiprocessing_chunksize == -1:
                    chunksize = max(
                        len(lines) // (args.process_count * 2), 500)
                else:
                    chunksize = args.multiprocessing_chunksize

                with Pool(args.process_count) as p:
                    self.examples = list(
                        tqdm(
                            p.imap(encode, lines, chunksize=chunksize),
                            total=len(lines),
                            # disable=silent,
                        )
                    )

            else:
                self.examples = [encode(line) for line in lines]

            mask_labels_pad = torch.cat(mask_labels_pad, dim=-1)
            self.examples = [
                token for tokens in self.examples for token in tokens]




            def check_data(tokens_ids, mask_labels):
                flag = True
                for tokens, mask_label in zip(tokens_ids, mask_labels):
                    if len(tokens) != len(mask_label):
                        flag = False
                        return flag
                return flag
            
            if len(self.examples) > block_size:
                self.examples = [
                    tokenizer.build_inputs_with_special_tokens(
                        self.examples[i: i + block_size]
                    )
                    for i in tqdm(
                        range(0, len(self.examples) -
                                block_size + 1, block_size)
                    )
                ]
                mask_labels_pad = [
                    pad_masks(mask_labels_pad[i: i + block_size])
                    for i in tqdm(
                            range(0, len(mask_labels_pad) -
                                    block_size + 1, block_size)
                    )
                ]
            else:
                self.examples = [
                    tokenizer.build_inputs_with_special_tokens(
                        self.examples)
                ]
                
                mask_labels_pad = [
                    pad_masks(mask_labels_pad)
                ]
            
            assert check_data(self.examples, mask_labels_pad)
            
            self.is_rxn_center_tokens = mask_labels_pad

            

        # self.is_rxn_center_tokens = (torch.tensor(
        #     self.examples) != torch.tensor(self.examples_with_center)).long()



    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item], dtype=torch.long), self.is_rxn_center_tokens[item]


def mask_tokens_with_rxn(
    inputs, tokenizer: PreTrainedTokenizer, args
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original."""
    if isinstance(inputs, torch.Tensor):
        inputs = inputs
    elif isinstance(inputs, tuple):
        inputs, is_center_marks = inputs
    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked language modeling."
            "Set 'mlm' to False in args if you want to use this tokenizer."
        )

    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training
    # (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, args.mlm_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
        for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(
        torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0
    )
    # We sample a few tokens in each sequence for masked reaction center modeling training
    # (with probability args.mrc_probability defaults to 0.5)
    probability_matrix.masked_fill_(
        is_center_marks.bool(), value=args.mrc_probability
    )
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    if args.model_type == "electra":
        # For ELECTRA, we replace all masked input tokens with tokenizer.mask_token
        inputs[masked_indices] = tokenizer.convert_tokens_to_ids(
            tokenizer.mask_token)
    else:
        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape, 0.8)
                            ).bool() & masked_indices
        )
        inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(
            tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(
            len(tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels

#  Adapted from  https://github.com/rxn4chemistry/rxnmapper/blob/master/rxnmapper/smiles_utils.py
def is_atom(token: str, special_tokens: List[str] = BAD_TOKS) -> bool:
    """

    
    Determine whether a token is an atom.

    Args:
        token: Token fed into the transformer model
        special_tokens: List of tokens to consider as non-atoms (often introduced by tokenizer)

    Returns:
        bool: True if atom, False if not
    """
    bad_toks = set(special_tokens)
    normal_atom = token[0].isalpha() or token[0] == "["
    is_bad = token in bad_toks
    return (not is_bad) and normal_atom


def get_mask_for_tokens(tokens: List[str],
                        special_tokens: List[str] = []) -> List[int]:
    """Return a mask for a tokenized smiles, where atom tokens
    are converted to 1 and other tokens to 0.

    e.g. c1ccncc1 would give [1, 0, 1, 1, 1, 1, 1, 0]

    Args:
        smiles: Smiles string of reaction
        special_tokens: Any special tokens to explicitly not call an atom. E.g. "[CLS]" or "[SEP]"

    Returns:
        Binary mask as a list where non-zero elements represent atoms
    """
    check_atom = partial(is_atom, special_tokens=special_tokens)

    atom_token_mask = [1 if check_atom(t) else 0 for t in tokens]
    return atom_token_mask


def number_tokens(tokens: List[str],
                  special_tokens: List[str] = BAD_TOKS) -> List[int]:
    """Map list of tokens to a list of numbered atoms

    Args:
        tokens: Tokenized SMILES
        special_tokens: List of tokens to not consider as atoms

    Example:
        >>> number_tokens(['[CLS]', 'C', '.', 'C', 'C', 'C', 'C', 'C', 'C','[SEP]'])
                #=> [-1, 0, -1, 1, 2, 3, 4, 5, 6, -1]
    """
    atom_num = 0
    isatm = partial(is_atom, special_tokens=special_tokens)

    def check_atom(t):
        if isatm(t):
            nonlocal atom_num
            ind = atom_num
            atom_num = atom_num + 1
            return ind
        return -1

    out = [check_atom(t) for t in tokens]

    return out

def visualize_atom_attention(smiles, weights):
    mol = Chem.MolFromSmiles(smiles)
    fig = SimilarityMaps.GetSimilarityMapFromWeights(mol, weights, colorMap=plt.get_cmap('RdBu'), alpha=0,
                                                        size=(150, 150))
    return fig


def identify_attention_token_idx_for_rxn_component(src_tokens):
    N_tokens = len(src_tokens)
    try:
        split_ind = src_tokens.index(
            ">>"
        )  # Index that separates products from reactants
        _product_inds = slice(split_ind + 1, N_tokens)
        _reactant_inds = slice(0, split_ind)
    except ValueError:
        raise ValueError(
            "rxn smiles is not a complete reaction. Can't find the '>>' to separate the products"
        )
    atom_token_mask = torch.tensor(
            get_mask_for_tokens(src_tokens, ["[CLS]", "[SEP]"])
        ).bool()
    token2atom = torch.tensor(number_tokens(src_tokens))
    atom2token = {
            k: v for k, v in zip(token2atom.tolist(), range(len(token2atom)))
        }


    _reactants_token_idx = torch.tensor([atom2token[x.item()] for x in token2atom[_reactant_inds][atom_token_mask[_reactant_inds]]])   # atom idx 顺序
    _product_token_idx = torch.tensor([atom2token[x.item()] for x in token2atom[_product_inds][atom_token_mask[_product_inds]]]) # atom idx + reactants atom number
    
    return _reactants_token_idx, _product_token_idx, atom_token_mask

def viz_attention_reaction(attention_weights, src_tokens, tgt_tokens, rxn_smiles, mean_attn=False):
    
    _reactants_token_idx, _product_token_idx, atom_token_mask = identify_attention_token_idx_for_rxn_component(src_tokens=src_tokens)
    
    if mean_attn:
        attention_weights = attention_weights.mean(1).unsqueeze(1).mean(0).unsqueeze(0)
    
    for layer_idx, layer_attentions in enumerate(attention_weights):
        for head_idx, head_attentions in enumerate(layer_attentions):
            if not mean_attn:
                print('Layer index: {}, Head index: {}'.format(layer_idx, head_idx))
            else:
                print('Average weight:')
            selected_attentions = torch.zeros_like(head_attentions)
            # softmax_attentions[atom_token_mask] = torch.softmax(head_attentions[atom_token_mask], dim=0)
            selected_attentions[atom_token_mask] = head_attentions[atom_token_mask]
            reactants_attentions = selected_attentions[_reactants_token_idx].T
            products_attentions = selected_attentions[_product_token_idx].T
            for tgt_token_idx in range(selected_attentions.size(1)):
                reactants_att = reactants_attentions[tgt_token_idx, :].tolist()
                products_att = products_attentions[tgt_token_idx, :].tolist()
                if tgt_tokens[tgt_token_idx]=='': continue
                print('Predicted Condition: {}'.format(tgt_tokens[tgt_token_idx]))
                react_fig = visualize_atom_attention(rxn_smiles.split('>>')[0], weights=reactants_att)
                print('Reactants:')
                display(react_fig)
                print('Products:')
                prod_fig = visualize_atom_attention(rxn_smiles.split('>>')[1], weights=products_att)
                display(prod_fig)
                print('####################################################################')

def get_dummy_atom_masks(mol):
    # mol from smarts
    # 获取原子的mask，* --> False, 其余为True
    masks = []
    for atom in mol.GetAtoms():
        if atom.GetSmarts() == '*':
            masks.append(False)
        else:
            masks.append(True)
    return torch.tensor(masks).bool()


def get_subgraph_condtion_attention_score(mol, mol_attentions, subgraph_smiles, subgraph_mols, subgraph_dummy_atom_masks, conditions, condition_masks):
    
    
    
    one_subgraph_condition_pair_score = defaultdict(list)
    conditions_rm_empty = np.asanyarray(conditions)[condition_masks.numpy()].tolist()
    condition_type_rm_empty = np.asanyarray(CONDITION_TYPE)[condition_masks.numpy()].tolist()
    for subgraph_smi, subgraph_mol, dummy_atom_masks, in zip(subgraph_smiles, subgraph_mols, subgraph_dummy_atom_masks):
        matches = mol.GetSubstructMatches(subgraph_mol)
        if matches:
            matches_indices = torch.tensor(matches)[:, dummy_atom_masks]
            for matches_idx in matches_indices:
                pair_scores = mol_attentions[condition_masks, :][:, matches_idx].mean(1).tolist()
                assert len(conditions_rm_empty) == len(pair_scores) == len(condition_type_rm_empty)
                for condition, condition_type, score in zip(conditions_rm_empty, condition_type_rm_empty, pair_scores):
                    one_subgraph_condition_pair_score['{}[PAIR]{}[TYPE]{}'.format(subgraph_smi, condition, condition_type)].append(score)
        
    return one_subgraph_condition_pair_score

def merge_score_dict(overall_score_dict, one_score_dict):
    for key in one_score_dict:
        overall_score_dict[key] += one_score_dict[key]
    return overall_score_dict

def analyze_subgraph_attention_with_condition(input_tokens_list, rxn_smiles, predicted_conditions, attention_weights, subgraph_smiles):

    # 分析时忽略空的Condition-->'' 以及不考虑匹配中假原子'*'的attention 
    assert len(input_tokens_list) == len(rxn_smiles) == len(predicted_conditions) == len(attention_weights)
    
    subgraph_mols = [Chem.MolFromSmarts(x) for x in subgraph_smiles]
    subgraph_dummy_atom_masks = [get_dummy_atom_masks(mol) for mol in subgraph_mols]
    subgraph_condition_pair_score = defaultdict(list)
    for src_tokens, rxn, one_conditions, one_attentions in tqdm(
        zip(
        input_tokens_list,
        rxn_smiles,
        predicted_conditions,
        attention_weights
    ),
        desc='Analysing Function Group-Conditions Attentions',
        total=len(input_tokens_list)):
        
        condition_masks = torch.from_numpy(np.asanyarray(one_conditions) != '')
        
        _reactants_token_idx, _product_token_idx, atom_token_mask = identify_attention_token_idx_for_rxn_component(src_tokens=src_tokens)
        reduced_one_attentions = one_attentions.mean(1).mean(0)
        selected_attentions = torch.zeros_like(reduced_one_attentions)
        selected_attentions[atom_token_mask] = reduced_one_attentions[atom_token_mask]
        reactants_attentions = selected_attentions[_reactants_token_idx].T
        products_attentions = selected_attentions[_product_token_idx].T
        
        react, prod = rxn.split('>>')
        
        react_mol = Chem.MolFromSmiles(react)
        prod_mol = Chem.MolFromSmiles(prod)
        
        react_one_subgraph_condition_pair_score = get_subgraph_condtion_attention_score(react_mol, reactants_attentions, subgraph_smiles, subgraph_mols, subgraph_dummy_atom_masks, one_conditions, condition_masks)
        subgraph_condition_pair_score = merge_score_dict(subgraph_condition_pair_score, react_one_subgraph_condition_pair_score)
        
        prod_one_subgraph_condition_pair_score = get_subgraph_condtion_attention_score(prod_mol, products_attentions, subgraph_smiles, subgraph_mols, subgraph_dummy_atom_masks, one_conditions, condition_masks)
        subgraph_condition_pair_score = merge_score_dict(subgraph_condition_pair_score, prod_one_subgraph_condition_pair_score)
    
    
    
    pairs = [x.split('[PAIR]') for x in list(subgraph_condition_pair_score.keys())]
    subgraph_list = list(set([x[0] for x in pairs]))
    subgraph_list.sort()
    condition_with_type_list = list(set([x[1] for x in pairs]))
    condition_with_type_list.sort()
    score_map = pd.DataFrame(np.zeros((len(subgraph_list), len(condition_with_type_list))))
    score_map.index = subgraph_list
    score_map.columns = condition_with_type_list
    conditions_list, conditions_types = zip(*[x.split('[TYPE]') for x in condition_with_type_list])
    # condition   
    for pair in  subgraph_condition_pair_score:
        subgraph_smi, condition_with_type = pair.split('[PAIR]')
        score_map.loc[subgraph_smi, condition_with_type] = np.array(subgraph_condition_pair_score[pair]).mean()
    
    score_map_condition_type_dict = {}
    for c_type in CONDITION_TYPE:
        type_indices = np.asanyarray(conditions_types) == c_type
        type_score_map = score_map.iloc[:, type_indices]
        type_columns = np.asanyarray(conditions_list)[type_indices].tolist()
        type_score_map.columns = type_columns
        score_map_condition_type_dict[c_type] = type_score_map
    
    return  score_map_condition_type_dict


def generate_vocab(rxn_smiles, vocab_path):
    general_vocab = [
        '[PAD]',
        '[unused1]',
        '[unused2]',
        '[unused3]',
        '[unused4]',
        '[unused5]',
        '[unused6]',
        '[unused7]',
        '[unused8]',
        '[unused9]',
        '[unused10]',
        '[UNK]',
        '[CLS]',
        '[SEP]',
        '[MASK]',
    ]
    vocab = set(general_vocab)
    basic_tokenizer = RegexTokenizer()
    for rxn in tqdm(rxn_smiles):
        tokens = basic_tokenizer.tokenize(rxn)
        for token in tokens:
            vocab.add(token)
    print('A total of {} vacabs were obtained.'.format(len(vocab)))
    print('Write vocabs to {}'.format(os.path.abspath(vocab_path)))

    with open(vocab_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(list(vocab)))