import os
import math
import json
import copy
import random
import argparse
import collections
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, LightningDataModule
from pytorch_lightning.strategies.ddp import DDPStrategy
from transformers import get_scheduler, EncoderDecoderModel, EncoderDecoderConfig, AutoTokenizer, AutoConfig, AutoModel

from textreact.tokenizer import get_tokenizers
from textreact.model import get_model, get_mlm_head
from textreact.dataset import ReactionConditionDataset, RetrosynthesisDataset, read_corpus, generate_train_label_corpus
from textreact.evaluate import evaluate_reaction_condition, evaluate_retrosynthesis
import textreact.utils as utils


def get_args(notebook=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='condition')
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_valid', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--precision', type=str, default='32')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--print_freq', type=int, default=200)
    parser.add_argument('--debug', action='store_true')
    # Model
    parser.add_argument('--template_based', action='store_true')
    parser.add_argument('--unattend_nonbonds', action='store_true')
    parser.add_argument('--encoder', type=str, default=None)
    parser.add_argument('--decoder', type=str, default=None)
    parser.add_argument('--encoder_pretrained', action='store_true')
    parser.add_argument('--decoder_pretrained', action='store_true')
    parser.add_argument('--share_embedding', action='store_true')
    parser.add_argument('--encoder_tokenizer', type=str, default='text')
    # Data
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--template_path', type=str, default=None)
    parser.add_argument('--train_file', type=str, default=None)
    parser.add_argument('--valid_file', type=str, default=None)
    parser.add_argument('--test_file', type=str, default=None)
    parser.add_argument('--vocab_file', type=str, default=None)
    parser.add_argument('--corpus_file', type=str, default=None)
    parser.add_argument('--train_label_corpus', action='store_true')
    parser.add_argument('--cache_path', type=str, default=None)
    parser.add_argument('--nn_path', type=str, default=None)
    parser.add_argument('--train_nn_file', type=str, default=None)
    parser.add_argument('--valid_nn_file', type=str, default=None)
    parser.add_argument('--test_nn_file', type=str, default=None)
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--max_dec_length', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--shuffle_smiles', action='store_true')
    parser.add_argument('--no_smiles', action='store_true')
    parser.add_argument('--num_neighbors', type=int, default=-1)
    parser.add_argument('--use_gold_neighbor', action='store_true')
    parser.add_argument('--max_num_neighbors', type=int, default=10)
    parser.add_argument('--random_neighbor_ratio', type=float, default=0.8)
    parser.add_argument('--mlm', action='store_true')
    parser.add_argument('--mlm_ratio', type=float, default=0.15)
    parser.add_argument('--mlm_layer', type=str, default='linear')
    parser.add_argument('--mlm_lambda', type=float, default=1)
    # Training
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--max_grad_norm', type=float, default=5.)
    parser.add_argument('--scheduler', type=str, choices=['cosine', 'constant'], default='cosine')
    parser.add_argument('--warmup_ratio', type=float, default=0)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--load_ckpt', type=str, default='best.ckpt')
    parser.add_argument('--eval_per_epoch', type=int, default=1)
    parser.add_argument('--val_metric', type=str, default='val_acc')
    parser.add_argument('--save_path', type=str, default='output/')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--num_train_example', type=int, default=None)
    parser.add_argument('--label_smoothing', type=float, default=0.0)
    # Inference
    parser.add_argument('--test_batch_size', type=int, default=64)
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument('--test_each_neighbor', action='store_true')
    parser.add_argument('--test_num_neighbors', type=int, default=1)

    args = parser.parse_args([]) if notebook else parser.parse_args()

    return args


class ReactionConditionRecommender(LightningModule):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.enc_tokenizer, self.dec_tokenizer = get_tokenizers(args)
        self.model = get_model(args, self.enc_tokenizer, self.dec_tokenizer)
        if args.mlm:
            self.mlm_head = get_mlm_head(args, self.model)
        self.validation_outputs = collections.defaultdict(dict)
        self.test_outputs = collections.defaultdict(dict)

    def compute_loss(self, logits, batch, reduction='mean'):
        if self.args.template_based:
            atom_logits, bond_logits = logits
            batch_size, max_len, atom_vocab_size = atom_logits.size()
            bond_vocab_size = bond_logits.size()[-1]
            atom_template_loss = F.cross_entropy(input=atom_logits.reshape(-1, atom_vocab_size),
                                                 target=batch['decoder_atom_template_labels'].reshape(-1),
                                                 reduction=reduction)
            bond_template_loss = F.cross_entropy(input=bond_logits.reshape(-1, bond_vocab_size),
                                                 target=batch['decoder_bond_template_labels'].reshape(-1),
                                                 reduction=reduction)
            if reduction == 'none':
                atom_template_loss = atom_template_loss.view(batch_size, -1).mean(dim=1)
                bond_template_loss = bond_template_loss.view(batch_size, -1).mean(dim=1)
            loss = atom_template_loss + bond_template_loss
        else:
            batch_size, max_len, vocab_size = logits.size()
            labels = batch['decoder_input_ids'][:, 1:]
            loss = F.cross_entropy(input=logits[:, :-1].reshape(-1, vocab_size), target=labels.reshape(-1),
                                   ignore_index=self.dec_tokenizer.pad_token_id, reduction=reduction)
            if reduction == 'none':
                loss = loss.view(batch_size, -1).mean(dim=1)
        return loss

    def compute_acc(self, logits, batch, reduction='mean'):
        # This accuracy is equivalent to greedy search accuracy
        if self.args.template_based:
            atom_logits_batch, bond_logits_batch = logits
            atom_probs_batch = F.softmax(atom_logits_batch, dim=-1)
            bond_probs_batch = F.softmax(bond_logits_batch, dim=-1)
            atom_probs_batch[batch['decoder_atom_template_labels'] == -100] = 0
            bond_probs_batch[batch['decoder_bond_template_labels'] == -100] = 0
            acc = []
            for atom_probs, bond_probs, bonds, raw_template_labels in zip(
                    atom_probs_batch, bond_probs_batch, batch['bonds'], batch['decoder_raw_template_labels']):
                edit_pred = utils.combined_edit(atom_probs, bond_probs, bonds, 1)[0][0]
                acc.append(float(edit_pred in raw_template_labels) / max(len(raw_template_labels), 1))
            acc = torch.tensor(acc)
        else:
            preds = logits.argmax(dim=-1)[:, :-1]
            labels = batch['decoder_input_ids'][:, 1:]
            acc = torch.logical_or(preds.eq(labels), labels.eq(self.dec_tokenizer.pad_token_id)).all(dim=-1)
        if reduction == 'mean':
            acc = acc.mean()
        return acc

    def compute_mlm_loss(self, encoder_last_hidden_state, labels):
        batch_size, trunc_len = labels.size()
        trunc_hidden_state = encoder_last_hidden_state[:, :trunc_len].contiguous()
        logits = self.mlm_head(trunc_hidden_state)
        return F.cross_entropy(input=logits.view(batch_size * trunc_len, -1), target=labels.view(-1))

    def training_step(self, batch, batch_idx):
        indices, batch_in, batch_out = batch
        output = self.model(**batch_in)
        loss = self.compute_loss(output.logits, batch_in)
        self.log('train_loss', loss)
        total_loss = loss
        if self.args.mlm:
            mlm_loss = self.compute_mlm_loss(output.encoder_last_hidden_state, batch_out['mlm_labels'])
            total_loss += mlm_loss * self.args.mlm_lambda
            self.log('mlm_loss', mlm_loss)
            self.log('total_loss', total_loss)
        return total_loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        indices, batch_in, batch_out = batch
        output = self.model(**batch_in)
        if self.args.val_metric == 'val_loss':
            scores = self.compute_loss(output.logits, batch_in, reduction='none').tolist()
        elif self.args.val_metric == 'val_acc':
            scores = self.compute_acc(output.logits, batch_in, reduction='none').tolist()
        else:
            raise ValueError
        for idx, score in zip(indices, scores):
            self.validation_outputs[dataloader_idx][idx] = score
        return output

    def on_validation_epoch_end(self):
        for dataloader_idx in self.validation_outputs:
            validation_outputs = self.gather_outputs(self.validation_outputs[dataloader_idx])
            val_score = np.mean([v for v in validation_outputs.values()])
            metric_name = self.args.val_metric if dataloader_idx == 0 else f'{self.args.val_metric}/{dataloader_idx}'
            self.log(metric_name, val_score, prog_bar=True, rank_zero_only=True)
        self.validation_outputs.clear()

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        indices, batch_in, batch_out = batch
        num_beams = self.args.num_beams
        if self.args.template_based:
            atom_logits_batch, bond_logits_batch = self.model(**batch_in).logits
            atom_probs_batch = F.softmax(atom_logits_batch, dim=-1)
            bond_probs_batch = F.softmax(bond_logits_batch, dim=-1)
            atom_probs_batch[batch_in['decoder_atom_template_labels'] == -100] = 0
            bond_probs_batch[batch_in['decoder_bond_template_labels'] == -100] = 0
            acc = []
            for idx, atom_probs, bond_probs, bonds, raw_template_labels in zip(
                    indices, atom_probs_batch, bond_probs_batch, batch_in['bonds'], batch_in['decoder_raw_template_labels']):
                edit_pred, edit_prob = utils.combined_edit(atom_probs, bond_probs, bonds, top_num=500)
                self.test_outputs[dataloader_idx][idx] = {
                    'prediction': edit_pred,
                    'score': edit_prob,
                    'raw_template_labels': raw_template_labels,
                    'top1_template_match': edit_pred[0] in raw_template_labels
                }
        else:
            output = self.model.generate(
                **batch_in, num_beams=num_beams, num_return_sequences=num_beams,
                max_length=self.args.max_dec_length, length_penalty=0,
                bos_token_id=self.dec_tokenizer.bos_token_id, eos_token_id=self.dec_tokenizer.eos_token_id,
                pad_token_id=self.dec_tokenizer.pad_token_id,
                return_dict_in_generate=True, output_scores=True)
            predictions = self.dec_tokenizer.batch_decode(output.sequences, skip_special_tokens=True)
            if 'sequences_scores' in predictions:
                scores = output.sequences_scores.tolist()
            else:
                scores = [0] * len(predictions)
            for i, idx in enumerate(indices):
                self.test_outputs[dataloader_idx][idx] = {
                    'prediction': predictions[i * num_beams: (i + 1) * num_beams],
                    'score': scores[i * num_beams: (i + 1) * num_beams]
                }
        return

    def on_test_epoch_end(self):
        for dataloader_idx in self.test_outputs:
            test_outputs = self.gather_outputs(self.test_outputs[dataloader_idx])
            if self.args.test_each_neighbor:
                test_outputs = utils.gather_prediction_each_neighbor(test_outputs, self.args.test_num_neighbors)
            if self.trainer.is_global_zero:
                # Save prediction
                with open(os.path.join(self.args.save_path,
                                       f'prediction_{self.test_dataset.name}_{dataloader_idx}.json'), 'w') as f:
                    json.dump(test_outputs, f)
                # Evaluate
                if self.args.task == 'condition':
                    accuracy = evaluate_reaction_condition(test_outputs, self.test_dataset.data_df)
                elif self.args.task == 'retro':
                    accuracy = evaluate_retrosynthesis(test_outputs, self.test_dataset.data_df, self.args.num_beams,
                                                       template_based=self.args.template_based,
                                                       template_path=self.args.template_path)
                else:
                    accuracy = []
                self.print(self.ckpt_path)
                self.print(json.dumps(accuracy))
        self.test_outputs.clear()

    def gather_outputs(self, outputs):
        if self.trainer.num_devices > 1:
            gathered = [{} for i in range(self.trainer.num_devices)]
            dist.all_gather_object(gathered, outputs)
            gathered_outputs = {}
            for outputs in gathered:
                gathered_outputs.update(outputs)
        else:
            gathered_outputs = outputs
        return gathered_outputs

    def configure_optimizers(self):
        num_training_steps = self.trainer.num_training_steps
        self.print(f'Num training steps: {num_training_steps}')
        num_warmup_steps = int(num_training_steps * self.args.warmup_ratio)
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        scheduler = get_scheduler(self.args.scheduler, optimizer, num_warmup_steps, num_training_steps)
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'}}


class ReactionConditionDataModule(LightningDataModule):

    DATASET_CLS = {
        'condition': ReactionConditionDataset,
        'retro': RetrosynthesisDataset,
    }

    def __init__(self, args, model):
        super().__init__()
        self.args = args
        self.enc_tokenizer = model.enc_tokenizer
        self.dec_tokenizer = model.dec_tokenizer
        self.train_dataset, self.val_dataset, self.test_dataset = None, None, None

    def prepare_data(self):
        args = self.args
        dataset_cls = self.DATASET_CLS[args.task]
        if args.do_train:
            data_file = os.path.join(args.data_path, args.train_file)
            self.train_dataset = dataset_cls(
                args, data_file, self.enc_tokenizer, self.dec_tokenizer, split='train')
            print(f'Train dataset: {len(self.train_dataset)}')
        if args.do_train or args.do_valid:
            data_file = os.path.join(args.data_path, args.valid_file)
            self.val_dataset = dataset_cls(
                args, data_file, self.enc_tokenizer, self.dec_tokenizer, split='val')
            print(f'Valid dataset: {len(self.val_dataset)}')
        if args.do_test:
            data_file = os.path.join(args.data_path, args.test_file)
            self.test_dataset = dataset_cls(
                args, data_file, self.enc_tokenizer, self.dec_tokenizer, split='test')
            print(f'Test dataset: {len(self.test_dataset)}')
        if args.corpus_file:
            if args.train_label_corpus:
                assert args.task == 'condition'
                corpus = generate_train_label_corpus(os.path.join(args.data_path, args.train_file))
            else:
                corpus = read_corpus(args.corpus_file, args.cache_path)
            if self.train_dataset is not None:
                self.train_dataset.load_corpus(corpus, os.path.join(args.nn_path, args.train_nn_file))
                self.train_dataset.print_example()
            if self.val_dataset is not None:
                self.val_dataset.load_corpus(corpus, os.path.join(args.nn_path, args.valid_nn_file))
            if self.test_dataset is not None:
                self.test_dataset.load_corpus(corpus, os.path.join(args.nn_path, args.test_nn_file))

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.args.batch_size, num_workers=self.args.num_workers,
            collate_fn=self.train_dataset.collator)

    def get_eval_dataloaders(self, dataset):
        args = self.args
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=dataset.collator)
        if args.corpus_file is None:
            return dataloader
        dataset_skip_gold = copy.copy(dataset)
        dataset_skip_gold.skip_gold_neighbor = True
        dataloader_skip_gold = torch.utils.data.DataLoader(
            dataset_skip_gold, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=dataset.collator)
        return [dataloader, dataloader_skip_gold]

    def val_dataloader(self):
        return self.get_eval_dataloaders(self.val_dataset)

    def test_dataloader(self):
        return self.get_eval_dataloaders(self.test_dataset)


def main():
    args = get_args()
    pl.seed_everything(args.seed, workers=True)

    model = ReactionConditionRecommender(args)

    dm = ReactionConditionDataModule(args, model)
    dm.prepare_data()

    checkpoint = pl.callbacks.ModelCheckpoint(
        monitor=args.val_metric, mode=utils.metric_to_mode[args.val_metric], save_top_k=1, filename='best',
        save_last=True, dirpath=args.save_path, auto_insert_metric_name=False)
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')
    if args.do_train and not args.debug:
        project_name = 'TextReact'
        if args.task == 'retro':
            project_name += '_retro'
        logger = pl.loggers.WandbLogger(
            project=project_name, save_dir=args.save_path, name=os.path.basename(args.save_path))
    else:
        logger = None

    trainer = pl.Trainer(
        strategy=DDPStrategy(find_unused_parameters=True),
        accelerator='gpu',
        devices=args.gpus,
        precision=args.precision,
        logger=logger,
        default_root_dir=args.save_path,
        callbacks=[checkpoint, lr_monitor],
        max_epochs=args.epochs,
        gradient_clip_val=args.max_grad_norm,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        check_val_every_n_epoch=args.eval_per_epoch,
        log_every_n_steps=10,
        deterministic=True)

    if args.do_train:
        trainer.num_training_steps = math.ceil(
            len(dm.train_dataset) / (args.batch_size * args.gpus * args.gradient_accumulation_steps)) * args.epochs
        # Load or delete existing checkpoint
        if args.overwrite:
            utils.clear_path(args.save_path, trainer)
            ckpt_path = None
        else:
            ckpt_path = os.path.join(args.save_path, args.load_ckpt)
            ckpt_path = ckpt_path if checkpoint.file_exists(ckpt_path, trainer) else None
        # Train
        trainer.fit(model, datamodule=dm, ckpt_path=ckpt_path)
        best_model_path = checkpoint.best_model_path
    else:
        best_model_path = os.path.join(args.save_path, args.load_ckpt)

    if args.do_valid or args.do_test:
        print('Load model checkpoint:', best_model_path)
        model = ReactionConditionRecommender.load_from_checkpoint(best_model_path, strict=False, args=args)
        model.ckpt_path = best_model_path

    if args.do_valid:
        trainer.validate(model, datamodule=dm)

    if args.do_test:
        model.test_dataset = dm.test_dataset
        trainer.test(model, datamodule=dm)


if __name__ == "__main__":
    main()
