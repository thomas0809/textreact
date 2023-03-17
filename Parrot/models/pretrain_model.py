# optional
from tqdm.auto import tqdm, trange
from dotenv import load_dotenv, find_dotenv
from models.utils import SimpleCenterDataset, SimpleCenterIdxDataset, mask_tokens_with_rxn
import os
from typing import Dict, List
import numpy as np
import pandas as pd
import torch
import logging
import random
from rxnfp.models import SmilesLanguageModelingModel
from rxnfp.tokenization import RegexTokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import math
from transformers.optimization import AdamW, Adafactor
from dataclasses import asdict
from torch.utils.tensorboard import SummaryWriter
from transformers.optimization import (
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)
from simpletransformers.language_modeling.language_modeling_utils import (
    SimpleDataset, load_hf_dataset)
from transformers.data.datasets.language_modeling import (
    LineByLineTextDataset,
    TextDataset,
)
try:
    import wandb

    wandb_available = True
except ImportError:
    wandb_available = False
logger = logging.getLogger(__name__)

load_dotenv(find_dotenv())


class RXNCenterModelingModel(SmilesLanguageModelingModel):

    def __init__(self,
                 model_name,
                 model_type='bert',
                 generator_name=None,
                 discriminator_name=None,
                 train_files=None,
                 args=None,
                 use_cuda=True,
                 cuda_device=-1,
                 **kwargs):
        super().__init__(model_type, model_name, generator_name,
                         discriminator_name, train_files, args, use_cuda,
                         cuda_device, **kwargs)

    def train_model(
        self,
        train_df,
        output_dir=None,
        show_running_loss=True,
        args=None,
        eval_df=None,
        verbose=True,
        **kwargs,
    ):

        if args:
            self.args.update_from_dict(args)

        if self.args.silent:
            show_running_loss = False

        if self.args.evaluate_during_training and eval_df is None:
            raise ValueError(
                "evaluate_during_training is enabled but eval_file is not specified."
                " Pass eval_file to model.train_model() if using evaluate_during_training."
            )

        if not output_dir:
            output_dir = self.args.output_dir

        if (os.path.exists(output_dir) and os.listdir(output_dir)
                and not self.args.overwrite_output_dir):
            raise ValueError(
                "Output directory ({}) already exists and is not empty."
                " Set args.overwrite_output_dir = True to overcome.".format(
                    output_dir))

        self._move_model_to_device()

        train_dataset = self.load_and_cache_examples(train_df, verbose=verbose)

        os.makedirs(output_dir, exist_ok=True)

        global_step, training_details = self.train(
            train_dataset,
            output_dir,
            show_running_loss=show_running_loss,
            eval_file=eval_df,
            verbose=verbose,
            **kwargs,
        )

        self.save_model(output_dir, model=self.model)
        if self.args.model_type == "electra":
            self.save_discriminator()
            self.save_generator()
        # model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        # model_to_save.save_pretrained(output_dir)
        # self.tokenizer.save_pretrained(output_dir)
        # torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

        if verbose:
            logger.info(" Training of {} model complete. Saved to {}.".format(
                self.args.model_type, output_dir))

        return global_step, training_details

    def train(self,
              train_dataset,
              output_dir,
              show_running_loss=True,
              eval_file=None,
              verbose=True,
              **kwargs):
        model = self.model
        args = self.args
        tokenizer = self.tokenizer

        def collate(examples: List[torch.Tensor]):
            if isinstance(examples[0], torch.Tensor):

                if tokenizer._pad_token is None:
                    return pad_sequence(examples, batch_first=True)
                return pad_sequence(examples,
                                    batch_first=True,
                                    padding_value=tokenizer.pad_token_id)
            elif isinstance(examples[0], tuple):
                _examples = examples
                examples = [x[0] for x in _examples]
                is_center_marks = [x[1] for x in _examples]
                if tokenizer._pad_token is None:
                    return pad_sequence(
                        examples,
                        batch_first=True), torch.stack(is_center_marks)
                return pad_sequence(
                    examples,
                    batch_first=True,
                    padding_value=tokenizer.pad_token_id), torch.stack(
                        is_center_marks)
            else:
                raise ValueError

        if self.is_world_master():
            tb_writer = SummaryWriter(log_dir=args.tensorboard_dir)
        train_sampler = (RandomSampler(train_dataset) if args.local_rank == -1
                         else DistributedSampler(train_dataset))
        if self.args.use_hf_datasets:
            # Inputs are already padded so default collation is fine
            train_dataloader = DataLoader(train_dataset,
                                          batch_size=args.train_batch_size,
                                          sampler=train_sampler)
        else:
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=args.train_batch_size,
                sampler=train_sampler,
                collate_fn=collate,
            )

        if args.max_steps > 0:
            t_total = args.max_steps
            args.num_train_epochs = (
                args.max_steps //
                (len(train_dataloader) // args.gradient_accumulation_steps) +
                1)
        else:
            t_total = (len(train_dataloader) //
                       args.gradient_accumulation_steps *
                       args.num_train_epochs)

        no_decay = ["bias", "LayerNorm.weight"]

        optimizer_grouped_parameters = []
        custom_parameter_names = set()
        for group in self.args.custom_parameter_groups:
            params = group.pop("params")
            custom_parameter_names.update(params)
            param_group = {**group}
            param_group["params"] = [
                p for n, p in model.named_parameters() if n in params
            ]
            optimizer_grouped_parameters.append(param_group)

        for group in self.args.custom_layer_parameters:
            layer_number = group.pop("layer")
            layer = f"layer.{layer_number}."
            group_d = {**group}
            group_nd = {**group}
            group_nd["weight_decay"] = 0.0
            params_d = []
            params_nd = []
            for n, p in model.named_parameters():
                if n not in custom_parameter_names and layer in n:
                    if any(nd in n for nd in no_decay):
                        params_nd.append(p)
                    else:
                        params_d.append(p)
                    custom_parameter_names.add(n)
            group_d["params"] = params_d
            group_nd["params"] = params_nd

            optimizer_grouped_parameters.append(group_d)
            optimizer_grouped_parameters.append(group_nd)

        if not self.args.train_custom_parameters_only:
            optimizer_grouped_parameters.extend([
                {
                    "params": [
                        p for n, p in model.named_parameters()
                        if n not in custom_parameter_names and not any(
                            nd in n for nd in no_decay)
                    ],
                    "weight_decay":
                    args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in model.named_parameters()
                        if n not in custom_parameter_names and any(
                            nd in n for nd in no_decay)
                    ],
                    "weight_decay":
                    0.0,
                },
            ])

        warmup_steps = math.ceil(t_total * args.warmup_ratio)
        args.warmup_steps = (warmup_steps
                             if args.warmup_steps == 0 else args.warmup_steps)

        if args.optimizer == "AdamW":
            optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=args.learning_rate,
                eps=args.adam_epsilon,
            )
        elif args.optimizer == "Adafactor":
            optimizer = Adafactor(
                optimizer_grouped_parameters,
                lr=args.learning_rate,
                eps=args.adafactor_eps,
                clip_threshold=args.adafactor_clip_threshold,
                decay_rate=args.adafactor_decay_rate,
                beta1=args.adafactor_beta1,
                weight_decay=args.weight_decay,
                scale_parameter=args.adafactor_scale_parameter,
                relative_step=args.adafactor_relative_step,
                warmup_init=args.adafactor_warmup_init,
            )

        else:
            raise ValueError(
                "{} is not a valid optimizer class. Please use one of ('AdamW', 'Adafactor') instead."
                .format(args.optimizer))

        if args.scheduler == "constant_schedule":
            scheduler = get_constant_schedule(optimizer)

        elif args.scheduler == "constant_schedule_with_warmup":
            scheduler = get_constant_schedule_with_warmup(
                optimizer, num_warmup_steps=args.warmup_steps)

        elif args.scheduler == "linear_schedule_with_warmup":
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_steps,
                num_training_steps=t_total,
            )

        elif args.scheduler == "cosine_schedule_with_warmup":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_steps,
                num_training_steps=t_total,
                num_cycles=args.cosine_schedule_num_cycles,
            )

        elif args.scheduler == "cosine_with_hard_restarts_schedule_with_warmup":
            scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_steps,
                num_training_steps=t_total,
                num_cycles=args.cosine_schedule_num_cycles,
            )

        elif args.scheduler == "polynomial_decay_schedule_with_warmup":
            scheduler = get_polynomial_decay_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_steps,
                num_training_steps=t_total,
                lr_end=args.polynomial_decay_schedule_lr_end,
                power=args.polynomial_decay_schedule_power,
            )

        else:
            raise ValueError("{} is not a valid scheduler.".format(
                args.scheduler))

        if (args.model_name and os.path.isfile(
                os.path.join(args.model_name, "optimizer.pt"))
                and os.path.isfile(
                    os.path.join(args.model_name, "scheduler.pt"))):
            # Load in optimizer and scheduler states
            optimizer.load_state_dict(
                torch.load(os.path.join(args.model_name, "optimizer.pt")))
            scheduler.load_state_dict(
                torch.load(os.path.join(args.model_name, "scheduler.pt")))

        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Distributed training
        if args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[args.local_rank],
                output_device=args.local_rank,
                find_unused_parameters=True,
            )

        logger.info(" Training started")

        global_step = 0
        training_progress_scores = None
        tr_loss, logging_loss = 0.0, 0.0
        model.zero_grad()
        train_iterator = trange(int(args.num_train_epochs),
                                desc="Epoch",
                                disable=args.silent,
                                mininterval=0)
        epoch_number = 0
        best_eval_metric = None
        early_stopping_counter = 0
        steps_trained_in_current_epoch = 0
        epochs_trained = 0

        if args.model_name and os.path.exists(args.model_name):
            try:
                # set global_step to gobal_step of last saved checkpoint from model path
                checkpoint_suffix = args.model_name.split("/")[-1].split("-")
                if len(checkpoint_suffix) > 2:
                    checkpoint_suffix = checkpoint_suffix[1]
                else:
                    checkpoint_suffix = checkpoint_suffix[-1]
                global_step = int(checkpoint_suffix)
                epochs_trained = global_step // (
                    len(train_dataloader) // args.gradient_accumulation_steps)
                steps_trained_in_current_epoch = global_step % (
                    len(train_dataloader) // args.gradient_accumulation_steps)

                logger.info(
                    "   Continuing training from checkpoint, will skip to saved global_step"
                )
                logger.info("   Continuing training from epoch %d",
                            epochs_trained)
                logger.info("   Continuing training from global step %d",
                            global_step)
                logger.info(
                    "   Will skip the first %d steps in the current epoch",
                    steps_trained_in_current_epoch,
                )
            except ValueError:
                logger.info("   Starting fine-tuning.")

        if args.evaluate_during_training:
            training_progress_scores = self._create_training_progress_scores(
                **kwargs)

        if args.wandb_project:
            wandb.init(project=args.wandb_project,
                       config={**asdict(args)},
                       **args.wandb_kwargs)
            wandb.run._label(repo="simpletransformers")
            wandb.watch(self.model)
            self.wandb_run_id = wandb.run.id

        if args.fp16:
            from torch.cuda import amp

            scaler = amp.GradScaler()

        for current_epoch in train_iterator:
            model.train()
            if isinstance(train_dataloader, DataLoader) and isinstance(
                    train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(current_epoch)
            if epochs_trained > 0:
                epochs_trained -= 1
                continue
            train_iterator.set_description(
                f"Epoch {epoch_number + 1} of {args.num_train_epochs}")
            batch_iterator = tqdm(
                train_dataloader,
                desc=f"Running Epoch {epoch_number} of {args.num_train_epochs}",
                disable=args.silent,
                mininterval=0,
            )
            for step, batch in enumerate(batch_iterator):
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                if self.args.use_hf_datasets:
                    batch = batch["input_ids"]

                mask_outputs = (mask_tokens_with_rxn(batch, tokenizer, args)
                                if args.mlm else (batch, batch))
                inputs = mask_outputs[0]
                labels = mask_outputs[1]
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                if args.fp16:
                    with amp.autocast():
                        if args.model_type == "longformer":
                            outputs = model(inputs,
                                            attention_mask=None,
                                            labels=labels)
                        else:
                            outputs = (model(inputs, labels=labels) if args.mlm
                                       else model(inputs, labels=labels))
                        # model outputs are always tuple in pytorch-transformers (see doc)
                        if args.model_type == "electra":
                            g_loss = outputs[0]
                            d_loss = outputs[1]
                            loss = g_loss + args.discriminator_loss_weight * d_loss
                        else:
                            loss = outputs[0]
                else:
                    if args.model_type == "longformer":
                        outputs = model(inputs,
                                        attention_mask=None,
                                        labels=labels)
                    else:
                        outputs = (model(inputs, labels=labels) if args.mlm
                                   else model(inputs, labels=labels))
                    # model outputs are always tuple in pytorch-transformers (see doc)
                    if args.model_type == "electra":
                        g_loss = outputs[0]
                        d_loss = outputs[1]
                        loss = g_loss + args.discriminator_loss_weight * d_loss
                    else:
                        loss = outputs[0]
                    # if loss.item() < 1:
                    #     masked = (labels[0] != -100).nonzero()
                    #     print(labels[0][masked])
                    #     preds = outputs[1][0, masked, :].clone().detach().cpu().numpy()
                    #     print(np.argmax(preds, axis=2))

                if args.n_gpu > 1:
                    loss = (
                        loss.mean()
                    )  # mean() to average on multi-gpu parallel training

                current_loss = loss.item()

                if show_running_loss:
                    batch_iterator.set_description(
                        f"Epochs {epoch_number}/{args.num_train_epochs}. Running Loss: {current_loss:9.4f}"
                    )

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                tr_loss += loss.item()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        scaler.unscale_(optimizer)
                    if args.optimizer == "AdamW":
                        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                       args.max_grad_norm)

                    if args.fp16:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1

                    if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                        # Log metrics
                        if self.is_world_master():
                            tb_writer.add_scalar("lr",
                                                 scheduler.get_last_lr()[0],
                                                 global_step)
                            tb_writer.add_scalar(
                                "loss",
                                (tr_loss - logging_loss) / args.logging_steps,
                                global_step,
                            )
                        logging_loss = tr_loss
                        if args.wandb_project or self.is_sweeping:
                            wandb.log({
                                "Training loss": current_loss,
                                "lr": scheduler.get_last_lr()[0],
                                "global_step": global_step,
                            })

                    if args.save_steps > 0 and global_step % args.save_steps == 0:
                        # Save model checkpoint
                        output_dir_current = os.path.join(
                            output_dir, "checkpoint-{}".format(global_step))

                        self.save_model(output_dir_current,
                                        optimizer,
                                        scheduler,
                                        model=model)

                    if args.evaluate_during_training and (
                            args.evaluate_during_training_steps > 0
                            and global_step %
                            args.evaluate_during_training_steps == 0):
                        # Only evaluate when single GPU otherwise metrics may not average well
                        results = self.eval_model(
                            eval_file,
                            verbose=verbose
                            and args.evaluate_during_training_verbose,
                            silent=args.evaluate_during_training_silent,
                            **kwargs,
                        )

                        if self.is_world_master():
                            for key, value in results.items():
                                try:
                                    tb_writer.add_scalar(
                                        "eval_{}".format(key), value,
                                        global_step)
                                except (NotImplementedError, AssertionError):
                                    pass

                        output_dir_current = os.path.join(
                            output_dir, "checkpoint-{}".format(global_step))

                        if args.save_eval_checkpoints:
                            self.save_model(
                                output_dir_current,
                                optimizer,
                                scheduler,
                                model=model,
                                results=results,
                            )

                        training_progress_scores["global_step"].append(
                            global_step)
                        training_progress_scores["train_loss"].append(
                            current_loss)
                        for key in results:
                            training_progress_scores[key].append(results[key])
                        report = pd.DataFrame(training_progress_scores)
                        report.to_csv(
                            os.path.join(args.output_dir,
                                         "training_progress_scores.csv"),
                            index=False,
                        )

                        if args.wandb_project or self.is_sweeping:
                            wandb.log(
                                self._get_last_metrics(
                                    training_progress_scores))

                        if not best_eval_metric:
                            best_eval_metric = results[
                                args.early_stopping_metric]
                            self.save_model(
                                args.best_model_dir,
                                optimizer,
                                scheduler,
                                model=model,
                                results=results,
                            )
                        if best_eval_metric and args.early_stopping_metric_minimize:
                            if (results[args.early_stopping_metric] -
                                    best_eval_metric <
                                    args.early_stopping_delta):
                                best_eval_metric = results[
                                    args.early_stopping_metric]
                                self.save_model(
                                    args.best_model_dir,
                                    optimizer,
                                    scheduler,
                                    model=model,
                                    results=results,
                                )
                                early_stopping_counter = 0
                            else:
                                if args.use_early_stopping:
                                    if (early_stopping_counter <
                                            args.early_stopping_patience):
                                        early_stopping_counter += 1
                                        if verbose:
                                            logger.info(
                                                f" No improvement in {args.early_stopping_metric}"
                                            )
                                            logger.info(
                                                f" Current step: {early_stopping_counter}"
                                            )
                                            logger.info(
                                                f" Early stopping patience: {args.early_stopping_patience}"
                                            )
                                    else:
                                        if verbose:
                                            logger.info(
                                                f" Patience of {args.early_stopping_patience} steps reached."
                                            )
                                            logger.info(
                                                " Training terminated.")
                                            train_iterator.close()
                                        return (
                                            global_step,
                                            tr_loss / global_step if not self.
                                            args.evaluate_during_training else
                                            training_progress_scores,
                                        )
                        else:
                            if (results[args.early_stopping_metric] -
                                    best_eval_metric >
                                    args.early_stopping_delta):
                                best_eval_metric = results[
                                    args.early_stopping_metric]
                                self.save_model(
                                    args.best_model_dir,
                                    optimizer,
                                    scheduler,
                                    model=model,
                                    results=results,
                                )
                                early_stopping_counter = 0
                            else:
                                if args.use_early_stopping:
                                    if (early_stopping_counter <
                                            args.early_stopping_patience):
                                        early_stopping_counter += 1
                                        if verbose:
                                            logger.info(
                                                f" No improvement in {args.early_stopping_metric}"
                                            )
                                            logger.info(
                                                f" Current step: {early_stopping_counter}"
                                            )
                                            logger.info(
                                                f" Early stopping patience: {args.early_stopping_patience}"
                                            )
                                    else:
                                        if verbose:
                                            logger.info(
                                                f" Patience of {args.early_stopping_patience} steps reached."
                                            )
                                            logger.info(
                                                " Training terminated.")
                                            train_iterator.close()
                                        return (
                                            global_step,
                                            tr_loss / global_step if not self.
                                            args.evaluate_during_training else
                                            training_progress_scores,
                                        )
                        model.train()

                if args.max_steps > 0 and global_step > args.max_steps:
                    return (
                        global_step,
                        tr_loss /
                        global_step if not self.args.evaluate_during_training
                        else training_progress_scores,
                    )

            epoch_number += 1
            output_dir_current = os.path.join(
                output_dir,
                "checkpoint-{}-epoch-{}".format(global_step, epoch_number))

            if args.save_model_every_epoch or args.evaluate_during_training:
                os.makedirs(output_dir_current, exist_ok=True)

            if args.save_model_every_epoch:
                self.save_model(output_dir_current,
                                optimizer,
                                scheduler,
                                model=model)

            if args.evaluate_during_training and args.evaluate_each_epoch:
                results = self.eval_model(
                    eval_file,
                    verbose=verbose and args.evaluate_during_training_verbose,
                    silent=args.evaluate_during_training_silent,
                    **kwargs,
                )

                self.save_model(output_dir_current,
                                optimizer,
                                scheduler,
                                results=results)

                training_progress_scores["global_step"].append(global_step)
                training_progress_scores["train_loss"].append(current_loss)
                for key in results:
                    training_progress_scores[key].append(results[key])
                report = pd.DataFrame(training_progress_scores)
                report.to_csv(
                    os.path.join(args.output_dir,
                                 "training_progress_scores.csv"),
                    index=False,
                )

                if args.wandb_project or self.is_sweeping:
                    wandb.log(self._get_last_metrics(training_progress_scores))

                if not best_eval_metric:
                    best_eval_metric = results[args.early_stopping_metric]
                    self.save_model(
                        args.best_model_dir,
                        optimizer,
                        scheduler,
                        model=model,
                        results=results,
                    )
                if best_eval_metric and args.early_stopping_metric_minimize:
                    if (results[args.early_stopping_metric] - best_eval_metric
                            < args.early_stopping_delta):
                        best_eval_metric = results[args.early_stopping_metric]
                        self.save_model(
                            args.best_model_dir,
                            optimizer,
                            scheduler,
                            model=model,
                            results=results,
                        )
                        early_stopping_counter = 0
                    else:
                        if (args.use_early_stopping
                                and args.early_stopping_consider_epochs):
                            if early_stopping_counter < args.early_stopping_patience:
                                early_stopping_counter += 1
                                if verbose:
                                    logger.info(
                                        f" No improvement in {args.early_stopping_metric}"
                                    )
                                    logger.info(
                                        f" Current step: {early_stopping_counter}"
                                    )
                                    logger.info(
                                        f" Early stopping patience: {args.early_stopping_patience}"
                                    )
                            else:
                                if verbose:
                                    logger.info(
                                        f" Patience of {args.early_stopping_patience} steps reached"
                                    )
                                    logger.info(" Training terminated.")
                                    train_iterator.close()
                                return (
                                    global_step,
                                    tr_loss / global_step
                                    if not self.args.evaluate_during_training
                                    else training_progress_scores,
                                )
                else:
                    if (results[args.early_stopping_metric] - best_eval_metric
                            > args.early_stopping_delta):
                        best_eval_metric = results[args.early_stopping_metric]
                        self.save_model(
                            args.best_model_dir,
                            optimizer,
                            scheduler,
                            model=model,
                            results=results,
                        )
                        early_stopping_counter = 0
                    else:
                        if (args.use_early_stopping
                                and args.early_stopping_consider_epochs):
                            if early_stopping_counter < args.early_stopping_patience:
                                early_stopping_counter += 1
                                if verbose:
                                    logger.info(
                                        f" No improvement in {args.early_stopping_metric}"
                                    )
                                    logger.info(
                                        f" Current step: {early_stopping_counter}"
                                    )
                                    logger.info(
                                        f" Early stopping patience: {args.early_stopping_patience}"
                                    )
                            else:
                                if verbose:
                                    logger.info(
                                        f" Patience of {args.early_stopping_patience} steps reached"
                                    )
                                    logger.info(" Training terminated.")
                                    train_iterator.close()
                                return (
                                    global_step,
                                    tr_loss / global_step
                                    if not self.args.evaluate_during_training
                                    else training_progress_scores,
                                )

            if args.max_steps > 0 and global_step > args.max_steps:
                return (
                    global_step,
                    tr_loss /
                    global_step if not self.args.evaluate_during_training else
                    training_progress_scores,
                )

        return (
            global_step,
            tr_loss / global_step if not self.args.evaluate_during_training
            else training_progress_scores,
        )

    def load_and_cache_examples(self,
                                dataset_df,
                                evaluate=False,
                                no_cache=False,
                                verbose=True,
                                silent=False):
        """
        Reads a text file from file_path and creates training features.

        Utility function for train() and eval() methods. Not intended to be used directly.
        """

        tokenizer = self.tokenizer
        args = self.args

        if not no_cache:
            no_cache = args.no_cache

        if not no_cache:
            os.makedirs(self.args.cache_dir, exist_ok=True)

        mode = "dev" if evaluate else "train"

        if self.args.use_hf_datasets:
            dataset = load_hf_dataset(dataset_df, tokenizer, self.args)
            return dataset
        elif args.dataset_class:
            CustomDataset = args.dataset_class
            return CustomDataset(tokenizer, args, dataset_df, mode,
                                 args.block_size)
        else:
            dataset_type = args.dataset_type
            if dataset_type == "text":
                return TextDataset(tokenizer,
                                   dataset_df,
                                   args.block_size,
                                   overwrite_cache=True)
            elif dataset_type == "line_by_line":
                return LineByLineTextDataset(tokenizer, dataset_df,
                                             args.block_size)
            else:
                special_tokens_count = (3 if bool(
                    args.model_type in ["roberta", "camembert", "xlmroberta"])
                                        else 2)
                if self.args.max_seq_length > 509 and self.args.model_type not in [
                        "longformer",
                        "bigbird",
                ]:
                    self.args.max_seq_length = (509 if bool(
                        args.model_type in
                        ["roberta", "camembert", "xlmroberta"]) else 510)
                    self.args.block_size = (509 if bool(
                        args.model_type in
                        ["roberta", "camembert", "xlmroberta"]) else 510)
                return SimpleCenterIdxDataset(
                    tokenizer,
                    self.args,
                    dataset_df,
                    mode,
                    args.block_size,
                    special_tokens_count,
                    sliding_window=args.sliding_window,
                )

    def evaluate(
        self,
        eval_dataset,
        output_dir,
        multi_label=False,
        prefix="",
        verbose=True,
        silent=False,
        **kwargs,
    ):
        """
        Evaluates the model on eval_dataset.

        Utility function to be used by the eval_model() method. Not intended to be used directly.
        """

        model = self.model
        args = self.args
        eval_output_dir = output_dir
        tokenizer = self.tokenizer

        results = {}

        def collate(examples: List[torch.Tensor]):
            if isinstance(examples[0], torch.Tensor):

                if tokenizer._pad_token is None:
                    return pad_sequence(examples, batch_first=True)
                return pad_sequence(examples,
                                    batch_first=True,
                                    padding_value=tokenizer.pad_token_id)
            elif isinstance(examples[0], tuple):
                _examples = examples
                examples = [x[0] for x in _examples]
                is_center_marks = [x[1] for x in _examples]
                if tokenizer._pad_token is None:
                    return pad_sequence(
                        examples,
                        batch_first=True), torch.stack(is_center_marks)
                return pad_sequence(
                    examples,
                    batch_first=True,
                    padding_value=tokenizer.pad_token_id), torch.stack(
                        is_center_marks)
            else:
                raise ValueError

        eval_sampler = SequentialSampler(eval_dataset)
        if self.args.use_hf_datasets:
            # Inputs are already padded so default collation is fine
            eval_dataloader = DataLoader(eval_dataset,
                                         batch_size=args.train_batch_size,
                                         sampler=eval_sampler)
        else:
            eval_dataloader = DataLoader(
                eval_dataset,
                batch_size=args.train_batch_size,
                sampler=eval_sampler,
                collate_fn=collate,
            )

        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        eval_loss = 0.0
        nb_eval_steps = 0
        model.eval()

        for batch in tqdm(eval_dataloader,
                          disable=args.silent or silent,
                          desc="Running Evaluation"):
            if self.args.use_hf_datasets:
                batch = batch["input_ids"]

            inputs, labels = (mask_tokens_with_rxn(batch, tokenizer, args)
                              if args.mlm else (batch, batch))
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            with torch.no_grad():
                outputs = (model(inputs, labels=labels)
                           if args.mlm else model(inputs, labels=labels))
                if args.model_type == "electra":
                    g_loss = outputs[0]
                    d_loss = outputs[1]
                    lm_loss = g_loss + args.discriminator_loss_weight * d_loss
                else:
                    lm_loss = outputs[0]
                if self.args.n_gpu > 1:
                    lm_loss = lm_loss.mean()
                eval_loss += lm_loss.item()
            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps
        perplexity = torch.exp(torch.tensor(eval_loss))

        results["eval_loss"] = eval_loss
        results["perplexity"] = perplexity

        output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            for key in sorted(results.keys()):
                writer.write("{} = {}\n".format(key, str(results[key])))

        return results
