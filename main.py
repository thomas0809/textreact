import os
import math
import json
import yaml
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, LightningDataModule
from pytorch_lightning.strategies.ddp import DDPStrategy
from transformers import get_scheduler, EncoderDecoderModel, EncoderDecoderConfig, AutoTokenizer, AutoConfig, AutoModel
from transformers.models.bert.modeling_bert import BertLMPredictionHead

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
    parser.add_argument('--encoder', type=str, default=None)
    parser.add_argument('--decoder', type=str, default=None)
    parser.add_argument('--encoder_pretrained', action='store_true')
    parser.add_argument('--decoder_pretrained', action='store_true')
    parser.add_argument('--share_embedding', action='store_true')
    parser.add_argument('--encoder_tokenizer', type=str, default='text')
    # Data
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--train_file', type=str, default=None)
    parser.add_argument('--valid_file', type=str, default=None)
    parser.add_argument('--test_file', type=str, default=None)
    parser.add_argument('--vocab_file', type=str, default=None)
    parser.add_argument('--corpus_file', type=str, default=None)
    parser.add_argument('--train_label_corpus', action='store_true')
    parser.add_argument('--cache_path', type=str, default=None)
    parser.add_argument('--id_field', type=str, default='id')
    parser.add_argument('--nn_path', type=str, default=None)
    parser.add_argument('--train_nn_file', type=str, default=None)
    parser.add_argument('--valid_nn_file', type=str, default=None)
    parser.add_argument('--test_nn_file', type=str, default=None)
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--max_dec_length', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--shuffle_smiles', action='store_true')
    parser.add_argument('--num_neighbors', type=int, default=-1)
    parser.add_argument('--skip_gold_neighbor', action='store_true')
    parser.add_argument('--use_gold_neighbor', action='store_true')
    parser.add_argument('--max_num_neighbors', type=int, default=10)
    parser.add_argument('--random_neighbor_ratio', type=float, default=0.8)
    parser.add_argument('--mlm', action='store_true')
    parser.add_argument('--mlm_ratio', type=float, default=0.15)
    parser.add_argument('--mlm_layer', type=str, default='linear')
    # Training
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--max_grad_norm', type=float, default=5.)
    parser.add_argument('--scheduler', type=str, choices=['cosine', 'constant'], default='cosine')
    parser.add_argument('--warmup_ratio', type=float, default=0)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument('--eval_per_epoch', type=int, default=1)
    parser.add_argument('--val_metric', type=str, default='val_acc')
    parser.add_argument('--save_path', type=str, default='output/')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--num_train_example', type=int, default=None)
    parser.add_argument('--label_smoothing', type=float, default=0.0)
    # Inference
    parser.add_argument('--test_batch_size', type=int, default=64)
    parser.add_argument('--num_beams', type=int, default=1)

    args = parser.parse_args([]) if notebook else parser.parse_args()

    return args


class ReactionConditionRecommender(LightningModule):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.enc_tokenizer, self.dec_tokenizer = get_tokenizers(args)
        self.model = get_model(args, self.enc_tokenizer)
        if args.mlm:
            self.mlm_head = get_mlm_head(args, self.model)
        self.validation_outputs = {}
        self.test_outputs = {}

    def compute_loss(self, logits, batch, reduction='mean'):
        batch_size, max_len, vocab_size = logits.size()
        labels = batch['decoder_input_ids'][:, 1:]
        loss = F.cross_entropy(input=logits[:, :-1].reshape(-1, vocab_size), target=labels.reshape(-1),
                               ignore_index=self.dec_tokenizer.pad_token_id, reduction=reduction)
        if reduction == 'none':
            loss = loss.view(batch_size, -1).mean(dim=1)
        return loss

    def compute_acc(self, logits, batch, reduction='mean'):
        # This accuracy is equivalent to greedy search accuracy
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
            total_loss += mlm_loss
            self.log('mlm_loss', mlm_loss)
            self.log('total_loss', total_loss)
        return total_loss

    def validation_step(self, batch, batch_idx):
        indices, batch_in, batch_out = batch
        output = self.model(**batch_in)
        if self.args.val_metric == 'val_loss':
            scores = self.compute_loss(output.logits, batch_in, reduction='none').tolist()
        elif self.args.val_metric == 'val_acc':
            scores = self.compute_acc(output.logits, batch_in, reduction='none').tolist()
        else:
            raise ValueError
        for idx, score in zip(indices, scores):
            self.validation_outputs[idx] = score
        return output

    def on_validation_epoch_end(self):
        validation_outputs = self.gather_outputs(self.validation_outputs)
        val_score = np.mean([v for v in validation_outputs.values()])
        self.log(self.args.val_metric, val_score, prog_bar=True, rank_zero_only=True)
        self.validation_outputs.clear()

    def test_step(self, batch, batch_idx):
        indices, batch_in, batch_out = batch
        num_beams = self.args.num_beams
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
            self.test_outputs[idx] = {
                'prediction': predictions[i * num_beams: (i + 1) * num_beams],
                'score': scores[i * num_beams: (i + 1) * num_beams]
            }
        return

    def on_test_epoch_end(self):
        test_outputs = self.gather_outputs(self.test_outputs)
        if self.trainer.is_global_zero:
            # Save prediction
            with open(os.path.join(self.args.save_path, f'prediction_{self.test_dataset.name}.json'), 'w') as f:
                json.dump(test_outputs, f)
            # Evaluate
            if self.args.task == 'condition':
                accuracy = evaluate_reaction_condition(test_outputs, self.test_dataset.data_df)
            elif self.args.task == 'retro':
                accuracy = evaluate_retrosynthesis(test_outputs, self.test_dataset.data_df)
            else:
                accuracy = []
            self.print(self.args.save_path)
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
                args, data_file, self.enc_tokenizer, self.dec_tokenizer, split='valid')
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

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.args.batch_size, num_workers=self.args.num_workers,
            collate_fn=self.val_dataset.collator)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset, batch_size=self.args.test_batch_size, num_workers=self.args.num_workers,
            collate_fn=self.test_dataset.collator)


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
            ckpt_path = os.path.join(args.save_path, 'best.ckpt')
            ckpt_path = ckpt_path if checkpoint.file_exists(ckpt_path, trainer) else None
        # Train
        trainer.fit(model, datamodule=dm, ckpt_path=ckpt_path)
        best_model_path = checkpoint.best_model_path
    else:
        best_model_path = os.path.join(args.save_path, 'best.ckpt')

    if args.do_valid or args.do_test:
        print('Load model checkpoint:', best_model_path)
        model = ReactionConditionRecommender.load_from_checkpoint(best_model_path, strict=False, args=args)

    if args.do_valid:
        trainer.validate(model, datamodule=dm)

    if args.do_test:
        model.test_dataset = dm.test_dataset
        trainer.test(model, datamodule=dm)


if __name__ == "__main__":
    main()
