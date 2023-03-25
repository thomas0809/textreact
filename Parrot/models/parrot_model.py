import pickle
from dataclasses import asdict
import logging
import math
import os
import random
import warnings

import numpy as np
import pandas as pd
from queue import Queue
from collections import defaultdict
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from rxnfp.models import SmilesClassificationModel, SmilesTokenizer
from torch.nn.modules.normalization import LayerNorm
from tensorboardX import SummaryWriter
from tqdm.auto import tqdm, trange
from tqdm.contrib import tenumerate
from transformers import Adafactor, AdamW, BertConfig, BertForSequenceClassification, get_constant_schedule, \
    get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup, \
    get_cosine_with_hard_restarts_schedule_with_warmup, get_linear_schedule_with_warmup, \
    get_polynomial_decay_schedule_with_warmup
from simpletransformers.config.model_args import ClassificationArgs
from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert.modeling_bert import BertModel
from simpletransformers.classification.classification_model import (
    MODELS_WITHOUT_SLIDING_WINDOW_SUPPORT,
    MODELS_WITHOUT_CLASS_WEIGHTS_SUPPORT)
from simpletransformers.classification.classification_utils import (
    ClassificationDataset, )
import yaml

from models.model_layer import PositionalEncoding, TokenEmbedding, TransformerDecoderLayer, TransformerDecoder

from models.utils import ConditionWithTempDataset, analyze_subgraph_attention_with_condition, load_dataset, print_args, \
    viz_attention_reaction

logger = logging.getLogger(__name__)

try:
    import wandb

    wandb_available = True
except ImportError:
    wandb_available = False

BOS, EOS, PAD, MASK = '[BOS]', '[EOS]', '[PAD]', '[MASK]'


class ParrotConditionModel(BertForSequenceClassification):

    def __init__(
            self,
            config,
    ) -> None:
        # super(ConditionModel).__init__()
        super().__init__(config)

        self.num_labels = config.num_labels
        self.config = config
        num_decoder_layers = config.num_decoder_layers
        nhead = config.nhead
        tgt_vocab_size = config.tgt_vocab_size
        dim_feedforward = config.dim_feedforward
        dropout = config.dropout
        d_model = config.d_model
        self.use_temperature = config.use_temperature
        device = None
        dtype = None

        if hasattr(config, 'output_attention'):
            self.output_attention = config.output_attention
        else:
            self.output_attention = False

        self.bert = BertModel(config)
        activation = F.relu
        layer_norm_eps = 1e-5
        factory_kwargs = {'device': device, 'dtype': dtype}

        decoder_layer = TransformerDecoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            layer_norm_eps,
            batch_first=True,
            norm_first=False,
            **factory_kwargs,
            output_attention=self.output_attention)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size=d_model)
        self.positional_encoding = PositionalEncoding(emb_size=d_model, dropout=dropout)
        decoder_norm = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.decoder = TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
            output_attention=self.output_attention)
        self.generator = nn.Linear(d_model, tgt_vocab_size)
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=config.condition_label_mapping[1][PAD])
        if self.use_temperature:
            self.memory_regression_layer = nn.Sequential(
                nn.Linear(self.config.max_position_embeddings * self.config.hidden_size, d_model),
                nn.ReLU(),
            )
            self.regression_layer1 = nn.Sequential(nn.Linear(d_model * 5, d_model), nn.ReLU())
            self.regression_layer2 = nn.Linear(2 * d_model, 1)
            self.reg_loss_fn = torch.nn.MSELoss()

    def forward(
            self,
            input_ids,
            attention_mask=None,
            token_type_ids=None,
            label_input=None,
            label_mask=None,
            label_padding_mask=None,
            labels=None,
            memory_key_padding_mask=None,
            temperature=None,
    ):
        if memory_key_padding_mask is None:
            memory_key_padding_mask = (attention_mask == 0)
        memory = self.bert(input_ids,
                           attention_mask=attention_mask,
                           token_type_ids=token_type_ids)[0]
        outs, attention_weights = self.decoder(
            self.positional_encoding(self.tgt_tok_emb(label_input)),
            memory,
            tgt_mask=label_mask,
            tgt_key_padding_mask=label_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask)
        logits = self.generator(outs)

        labels_out = labels[:, 1:]
        loss = self.loss_fn(logits.reshape(-1, logits.shape[-1]), labels_out.reshape(-1))

        if self.use_temperature:
            temp_memory = memory.reshape(
                -1,
                self.config.max_position_embeddings * self.config.hidden_size)
            temp_memory = self.memory_regression_layer(temp_memory)

            temp_out = outs[:, :-1, :]
            temp_out = temp_out.reshape(-1, temp_out.size(1) * temp_out.size(2))
            temp_out = self.regression_layer1(temp_out)

            temp_out = torch.cat([temp_memory, temp_out], dim=1)
            temp_out = self.regression_layer2(temp_out)

            loss_reg = self.reg_loss_fn(temp_out.reshape(-1), temperature.reshape(-1))
            # loss += 0.001*loss_reg

            return loss, logits, attention_weights, loss_reg, temp_out
        return loss, logits, attention_weights

    def encode(self, input_ids):
        return self.bert(input_ids)[0]

    def decode(self, tgt, memory, tgt_mask, memory_key_padding_mask):
        decoder_output, attention_weightes = self.decoder(
            self.positional_encoding(self.tgt_tok_emb(tgt)),
            memory,
            tgt_mask,
            memory_key_padding_mask=memory_key_padding_mask)
        return decoder_output, attention_weightes

    def decode_temperature(self, memory, decoder_output):
        temp_memory = memory.reshape(
            -1, self.config.max_position_embeddings * self.config.hidden_size)
        temp_memory = self.memory_regression_layer(temp_memory)

        # ==> (batch_size, n_condition, d_model) 没有结束符号
        temp_out = decoder_output
        temp_out = temp_out.reshape(-1, temp_out.size(1) * temp_out.size(2))
        temp_out = self.regression_layer1(temp_out)

        temp_out = torch.cat([temp_memory, temp_out], dim=1)
        temp_out = self.regression_layer2(temp_out)
        return temp_out


class ParrotConditionPredictionModel(SmilesClassificationModel):

    def __init__(
            self,
            model_type,
            model_name,
            tokenizer_type=None,
            tokenizer_name=None,
            # num_labels=None,
            weight=None,
            args=None,
            use_cuda=True,
            cuda_device=-1,
            freeze_encoder=False,
            freeze_all_but_one=False,
            # decoder_args=None,
            **kwargs,
    ):

        MODEL_CLASSES = {
            "bert": (BertConfig, ParrotConditionModel, SmilesTokenizer),
        }

        if model_type not in MODEL_CLASSES.keys():
            raise NotImplemented(f"Currently the following model types are implemented: {MODEL_CLASSES.keys()}")

        self.args = self._load_model_args(model_name)

        decoder_args = args['decoder_args']
        try:
            self.condition_label_mapping = decoder_args[
                'condition_label_mapping']
        except:
            print('Warning: condition_label_mapping is not set!')

        if isinstance(args, dict):
            self.args.update_from_dict(args)
        elif isinstance(args, ClassificationArgs):
            self.args = args

        if (model_type in MODELS_WITHOUT_SLIDING_WINDOW_SUPPORT
                and self.args.sliding_window):
            raise ValueError(
                "{} does not currently support sliding window".format(
                    model_type))

        if self.args.thread_count:
            torch.set_num_threads(self.args.thread_count)
        if "sweep_config" in kwargs:
            self.is_sweeping = True
            sweep_config = kwargs.pop("sweep_config")
            sweep_values = sweep_config_to_sweep_values(sweep_config)
            self.args.update_from_dict(sweep_values)
        else:
            self.is_sweeping = False

        if self.args.manual_seed:
            random.seed(self.args.manual_seed)
            np.random.seed(self.args.manual_seed)
            torch.manual_seed(self.args.manual_seed)
            if self.args.n_gpu > 0:
                torch.cuda.manual_seed_all(self.args.manual_seed)

        config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]

        if tokenizer_type is not None:
            if isinstance(tokenizer_type, str):
                _, _, tokenizer_class = MODEL_CLASSES[tokenizer_type]
            else:
                tokenizer_class = tokenizer_type

        if model_name:
            self.config = config_class.from_pretrained(model_name, **self.args.config)
        else:
            self.config = config_class(**self.args.config, **kwargs)
        self.num_labels = self.config.num_labels
        self.config.update(decoder_args)
        self.config.update({'use_temperature': args['use_temperature']})
        if 'ignore_mismatched_sizes' in args:
            kwargs.update({'ignore_mismatched_sizes': args['ignore_mismatched_sizes']})
        if 'output_attention' in args:
            self.config.update({'output_attention': args['output_attention']})
        if model_type in MODELS_WITHOUT_CLASS_WEIGHTS_SUPPORT and weight is not None:
            raise ValueError(
                "{} does not currently support class weights".format(model_type))
        else:
            self.weight = weight

        if use_cuda:
            if torch.cuda.is_available():
                if cuda_device == -1:
                    self.device = torch.device("cuda")
                else:
                    self.device = torch.device(f"cuda:{cuda_device}")
            else:
                raise ValueError(
                    "'use_cuda' set to True when cuda is unavailable."
                    " Make sure CUDA is available or set use_cuda=False.")
        else:
            self.device = "cpu"
        if model_name:
            if not self.args.quantized_model:
                if self.weight:
                    self.model = model_class.from_pretrained(
                        model_name,
                        config=self.config,
                        weight=torch.Tensor(self.weight).to(self.device),
                        **kwargs,
                    )
                else:
                    self.model = model_class.from_pretrained(
                        model_name, config=self.config, **kwargs)
            else:
                quantized_weights = torch.load(
                    os.path.join(model_name, "pytorch_model.bin"))
                if self.weight:
                    self.model = model_class.from_pretrained(
                        None,
                        config=self.config,
                        state_dict=quantized_weights,
                        weight=torch.Tensor(self.weight).to(self.device),
                    )
                else:
                    self.model = model_class.from_pretrained(
                        None, config=self.config, state_dict=quantized_weights)

            if self.args.dynamic_quantize:
                self.model = torch.quantization.quantize_dynamic(
                    self.model, {torch.nn.Linear}, dtype=torch.qint8)
            if self.args.quantized_model:
                self.model.load_state_dict(quantized_weights)
            if self.args.dynamic_quantize:
                self.args.quantized_model = True
        else:
            self.model = model_class(config=self.config)
        if not hasattr(self.args, 'freeze_pretrain'):
            self.args.freeze_pretrain = False
        if self.args.freeze_pretrain:
            train_layers = [
                'tgt_tok_emb.embedding.weight', 'generator.weight',
                'generator.bias'
            ]
            print(f'Frozen load parameters, training {train_layers}')
            for p in self.model.named_parameters():
                if p[0] not in train_layers:
                    p[1].requires_grad = False

        self.results = {}

        if not use_cuda:
            self.args.fp16 = False

        if self.args.fp16:
            try:
                from torch.cuda import amp
            except AttributeError:
                raise AttributeError("fp16 requires Pytorch >= 1.6. Please update Pytorch or turn off fp16.")

        if tokenizer_name is None:
            tokenizer_name = model_name
        if hasattr(self.args, 'vocab_path'):
            pass
        else:
            self.args.vocab_path = None
        if tokenizer_name in [
            "vinai/bertweet-base",
            "vinai/bertweet-covid19-base-cased",
            "vinai/bertweet-covid19-base-uncased",
        ]:
            self.tokenizer = tokenizer_class.from_pretrained(
                tokenizer_name,
                do_lower_case=self.args.do_lower_case,
                normalization=True,
                **kwargs,
            )
        elif not self.args.vocab_path and not tokenizer_name in [
            "vinai/bertweet-base",
            "vinai/bertweet-covid19-base-cased",
            "vinai/bertweet-covid19-base-uncased",
        ]:
            self.tokenizer = tokenizer_class.from_pretrained(
                tokenizer_name,
                do_lower_case=self.args.do_lower_case,
                **kwargs)

        elif self.args.vocab_path:
            self.tokenizer = tokenizer_class(self.args.vocab_path,
                                             do_lower_case=False)
            model_to_resize = self.model.module if hasattr(self.model, "module") else self.model
            model_to_resize.resize_token_embeddings(len(self.tokenizer))

        if self.args.special_tokens_list:
            self.tokenizer.add_tokens(self.args.special_tokens_list,
                                      special_tokens=True)
            self.model.resize_token_embeddings(len(self.tokenizer))

        self.args.model_name = model_name
        self.args.model_type = model_type
        self.args.tokenizer_name = tokenizer_name
        self.args.tokenizer_type = tokenizer_type

        if model_type in ["camembert", "xlmroberta"]:
            warnings.warn(
                f"use_multiprocessing automatically disabled as {model_type}"
                " fails when using multiprocessing for feature conversion.")
            self.args.use_multiprocessing = False

        if self.args.wandb_project and not wandb_available:
            warnings.warn("wandb_project specified but wandb is not available. Wandb disabled.")
            self.args.wandb_project = None

        if freeze_encoder:
            for name, param in self.model.named_parameters():
                if 'classifier' in name:
                    continue
                param.requires_grad = False
        elif freeze_all_but_one:
            n_layers = self.model.config.num_hidden_layers
            for name, param in self.model.named_parameters():
                if str(n_layers - 1) in name:
                    continue
                elif 'classifier' in name:
                    continue
                elif 'pooler' in name:
                    continue
                param.requires_grad = False

    def load_and_cache_examples(self,
                                examples,
                                evaluate=False,
                                no_cache=False,
                                multi_label=False,
                                verbose=True,
                                silent=False):

        process_count = self.args.process_count

        tokenizer = self.tokenizer
        args = self.args

        if not no_cache:
            no_cache = args.no_cache

        if not multi_label and args.regression:
            output_mode = "regression"
        else:
            output_mode = "classification"

        if not no_cache:
            os.makedirs(self.args.cache_dir, exist_ok=True)

        mode = "dev" if evaluate else "train"
        if not args.use_temperature:
            dataset = ClassificationDataset(
                examples,
                self.tokenizer,
                self.args,
                mode=mode,
                multi_label=multi_label,
                output_mode=output_mode,
                no_cache=no_cache,
            )

        else:
            dataset = ConditionWithTempDataset(
                examples,
                self.tokenizer,
                self.args,
                mode=mode,
                multi_label=multi_label,
                output_mode=output_mode,
                no_cache=no_cache,
            )
        return dataset

    def train_model(self,
                    train_df,
                    output_dir=None,
                    show_running_loss=True,
                    args=None,
                    eval_df=None,
                    verbose=True,
                    **kwargs):
        if args:
            self.args.update_from_dict(args)

        if self.args.silent:
            show_running_loss = False

        if self.args.evaluate_during_training and eval_df is None:
            raise ValueError(
                "evaluate_during_training is enabled but eval_df is not specified."
                " Pass eval_df to model.train_model() if using evaluate_during_training."
            )

        if not output_dir:
            output_dir = self.args.output_dir

        if os.path.exists(output_dir) and os.listdir(output_dir) and not self.args.overwrite_output_dir:
            raise ValueError(
                "Output directory ({}) already exists and is not empty."
                " Set overwrite_output_dir: True to automatically overwrite.".format(output_dir))
        self._move_model_to_device()

        # 数据集加载只保留一种加载方式，其余全部删除
        train_examples = (
            train_df["text"].astype(str).tolist(),
            train_df["labels"].tolist(),
        )

        train_dataset = self.load_and_cache_examples(train_examples, verbose=verbose, no_cache=True)
        print('loaded train dataset {}'.format(train_dataset.examples['input_ids'].shape[0]))
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(
            train_dataset,
            sampler=train_sampler,
            batch_size=self.args.train_batch_size,
            num_workers=self.args.dataloader_num_workers,
        )

        os.makedirs(output_dir, exist_ok=True)

        global_step, training_details = self.train(
            train_dataloader,
            output_dir,
            show_running_loss=show_running_loss,
            eval_df=eval_df,
            verbose=verbose,
            **kwargs,
        )
        self.save_model(model=self.model)
        if verbose:
            logger.info(" Training of {} model complete. Saved to {}.".format(self.args.model_type, output_dir))

        return global_step, training_details

    def train(self,
              train_dataloader,
              output_dir,
              show_running_loss=True,
              eval_df=None,
              test_df=None,
              verbose=True,
              **kwargs):

        model = self.model
        args = self.args

        tb_writer = SummaryWriter(logdir=args.tensorboard_dir)

        t_total = (len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs)

        no_decay = ["bias", "LayerNorm.weight"]

        optimizer_grouped_parameters = []
        custom_parameter_names = set()
        for group in self.args.custom_parameter_groups:
            params = group.pop("params")
            custom_parameter_names.update(params)
            param_group = {**group}
            param_group["params"] = [p for n, p in model.named_parameters() if n in params]
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
                        if n not in custom_parameter_names and not any(nd in n for nd in no_decay)
                    ],
                    "weight_decay":
                        args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in model.named_parameters()
                        if n not in custom_parameter_names and any(nd in n for nd in no_decay)
                    ],
                    "weight_decay":
                        0.0,
                },
            ])

        warmup_steps = math.ceil(t_total * args.warmup_ratio)
        if warmup_steps != 0:
            args.warmup_steps = warmup_steps
        else:
            args.warmup_steps = (warmup_steps if args.warmup_steps == 0 else args.warmup_steps)

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
            print("Using Adafactor for T5")
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
                optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
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
            raise ValueError("{} is not a valid scheduler.".format(args.scheduler))

        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

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
        current_loss = "Initializing"

        if args.model_name and os.path.exists(args.model_name):
            try:
                # set global_step to gobal_step of last saved checkpoint from model path
                checkpoint_suffix = args.model_name.split("/")[-1].split("-")
                if len(checkpoint_suffix) > 2:
                    checkpoint_suffix = checkpoint_suffix[1]
                else:
                    checkpoint_suffix = checkpoint_suffix[-1]
                global_step = int(checkpoint_suffix)
                epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
                steps_trained_in_current_epoch = global_step % (
                        len(train_dataloader) // args.gradient_accumulation_steps)

                logger.info("   Continuing training from checkpoint, will skip to saved global_step")
                logger.info("   Continuing training from epoch %d", epochs_trained)
                logger.info("   Continuing training from global step %d", global_step)
                logger.info("   Will skip the first %d steps in the current epoch", steps_trained_in_current_epoch)
            except ValueError:
                logger.info("   Starting fine-tuning.")
        if args.evaluate_during_training:
            training_progress_scores = self._create_training_progress_scores(**kwargs)
            if args.use_temperature:
                training_progress_scores['eval_temp_mae'] = []

        if args.wandb_project:
            if not wandb.setup().settings.sweep_id:
                logger.info(" Initializing WandB run for training.")
                wandb.init(project=args.wandb_project, config={**asdict(args)}, **args.wandb_kwargs, )
                wandb.run._label(repo="simpletransformers")
            wandb.watch(self.model)

        if self.args.fp16:
            from torch.cuda import amp
            scaler = amp.GradScaler()

        for _ in train_iterator:
            model.train()
            if epochs_trained > 0:
                epochs_trained -= 1
                continue
            train_iterator.set_description(f"Epoch {epoch_number + 1} of {args.num_train_epochs}")
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

                inputs = self._get_inputs_dict(batch)
                if self.args.fp16:
                    with amp.autocast():
                        outputs = model(**inputs)
                        # model outputs are always tuple in pytorch-transformers (see doc)
                        loss = outputs[0]
                else:
                    outputs = model(**inputs)
                    # model outputs are always tuple in pytorch-transformers (see doc)
                    loss = outputs[0]

                if args.n_gpu > 1:
                    loss = (loss.mean())  # mean() to average on multi-gpu parallel training

                current_loss = loss.item()
                current_ppl = np.exp(current_loss)
                if args.use_temperature:
                    reg_loss = outputs[3]
                    loss += 0.001 * reg_loss

                if show_running_loss:
                    batch_iterator.set_description(
                        f"Epochs {epoch_number}/{args.num_train_epochs}. Running Loss: {current_loss:8.4f}")

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if self.args.fp16:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                tr_loss += loss.item()

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if self.args.fp16:
                        scaler.unscale_(optimizer)
                    if args.optimizer == "AdamW":
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                    if self.args.fp16:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1

                    if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                        # Log metrics
                        tb_writer.add_scalar("lr", scheduler.get_last_lr()[0], global_step)
                        tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                        logging_loss = tr_loss
                        if args.wandb_project or self.is_sweeping:
                            wandb.log({
                                "Training loss": current_loss,
                                "lr": scheduler.get_last_lr()[0],
                            }, step=global_step)

                    if args.save_steps > 0 and global_step % args.save_steps == 0:
                        # Save model checkpoint
                        output_dir_current = os.path.join(output_dir, "checkpoint-{}".format(global_step))
                        self.save_model(output_dir_current, optimizer, scheduler, model=model)

            epoch_number += 1
            output_dir_current = os.path.join(output_dir, "checkpoint-{}-epoch-{}".format(global_step, epoch_number))

            if args.save_model_every_epoch or args.evaluate_during_training:
                os.makedirs(output_dir_current, exist_ok=True)

            if args.save_model_every_epoch:
                self.save_model(output_dir_current, optimizer, scheduler, model=model)

            if args.evaluate_during_training and args.evaluate_each_epoch:
                results = self.eval_model(
                    eval_df,
                    verbose=verbose and args.evaluate_during_training_verbose,
                    silent=args.evaluate_during_training_silent,
                    wandb_log=False,
                    **kwargs,
                )

                self.save_model(output_dir_current, optimizer, scheduler, results=results)

                training_progress_scores["global_step"].append(global_step)
                training_progress_scores["train_loss"].append(current_loss)
                for key in results:
                    training_progress_scores[key].append(results[key])
                report = pd.DataFrame(training_progress_scores)
                report.to_csv(os.path.join(args.output_dir, "training_progress_scores.csv"), index=False)

                if args.wandb_project or self.is_sweeping:
                    wandb.log(self._get_last_metrics(training_progress_scores), step=global_step)

                tb_writer.flush()

                if not best_eval_metric:
                    best_eval_metric = results[args.early_stopping_metric]
                    self.save_model(args.best_model_dir, optimizer, scheduler, model=model, results=results)
                else:
                    if args.early_stopping_metric_minimize:
                        delta = best_eval_metric - results[args.early_stopping_metric]
                    else:
                        delta = results[args.early_stopping_metric] - best_eval_metric

                    if delta > args.early_stopping_delta:
                        best_eval_metric = results[args.early_stopping_metric]
                        self.save_model(args.best_model_dir, optimizer, scheduler, model=model, results=results)
                        early_stopping_counter = 0
                    else:
                        if args.use_early_stopping and args.early_stopping_consider_epochs:
                            if early_stopping_counter < args.early_stopping_patience:
                                early_stopping_counter += 1
                                if verbose:
                                    logger.info(f" No improvement in {args.early_stopping_metric}")
                                    logger.info(f" Current step: {early_stopping_counter}")
                                    logger.info(f" Early stopping patience: {args.early_stopping_patience}")
                            else:
                                if verbose:
                                    logger.info(f" Patience of {args.early_stopping_patience} steps reached")
                                    logger.info(" Training terminated.")
                                    train_iterator.close()
                                return (
                                    global_step,
                                    tr_loss / global_step if not self.args.evaluate_during_training
                                    else training_progress_scores,
                                )

        return (
            global_step,
            tr_loss / global_step if not self.args.evaluate_during_training
            else training_progress_scores,
        )

    def eval_model(self,
                   eval_df,
                   output_dir=None,
                   verbose=True,
                   silent=False,
                   wandb_log=True,
                   **kwargs):
        if not output_dir:
            output_dir = self.args.output_dir

        self._move_model_to_device()

        result = self.evaluate(
            eval_df,
            output_dir,
            verbose=verbose,
            silent=silent,
            wandb_log=wandb_log,
            **kwargs,
        )
        self.results.update(result)

        if verbose:
            logger.info(self.results)

        return result

    def evaluate(self,
                 eval_df,
                 output_dir,
                 prefix="",
                 verbose=True,
                 silent=False,
                 wandb_log=True,
                 **kwargs):

        model = self.model
        args = self.args
        eval_output_dir = output_dir

        results = {}

        eval_examples = (
            eval_df["text"].astype(str).tolist(),
            eval_df["labels"].tolist(),
        )
        os.makedirs(eval_output_dir, exist_ok=True)
        eval_dataset = self.load_and_cache_examples(eval_examples,
                                                    evaluate=True,
                                                    verbose=verbose,
                                                    silent=silent,
                                                    no_cache=True)
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset,
                                     sampler=eval_sampler,
                                     batch_size=args.eval_batch_size)

        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        eval_loss = 0.0
        eval_reg_loss = 0.0
        eval_temp_mae = 0.0
        nb_eval_steps = 0
        n_batches = len(eval_dataloader)
        if self.args.use_temperature:
            label_len = eval_dataset.labels[0].shape[1] - 1
        else:
            label_len = eval_dataset.labels.shape[1] - 1

        out_label_ids = np.empty((len(eval_dataset), label_len))
        model.eval()
        if self.args.fp16:
            from torch.cuda import amp

        for i, batch in enumerate(
                tqdm(
                    eval_dataloader,
                    disable=args.silent or silent,
                    desc="Running Evaluation",
                )):

            with torch.no_grad():
                inputs = self._get_inputs_dict(batch)
                if self.args.fp16:
                    with amp.autocast():
                        outputs = self._calculate_loss(
                            model,
                            inputs,
                            loss_fct=None,
                            num_labels=label_len,
                            args=self.args,
                        )
                        tmp_eval_loss, logits = outputs[:2]
                else:
                    outputs = self._calculate_loss(
                        model,
                        inputs,
                        loss_fct=None,
                        num_labels=label_len,
                        args=self.args,
                    )
                    tmp_eval_loss, _ = outputs[:2]
                eval_loss += tmp_eval_loss.item()
                if args.use_temperature:
                    eval_reg_loss += outputs[3].item()
                    eval_temp_mae += outputs[-1].item()
            nb_eval_steps += 1
            start_index = self.args.eval_batch_size * i
            end_index = (start_index + self.args.eval_batch_size if i != (n_batches - 1) else len(eval_dataset))

            out_label_ids[start_index:end_index] = (inputs["labels"][:, 1:].detach().cpu().numpy())

        eval_loss = eval_loss / nb_eval_steps
        eval_ppl = np.exp(eval_loss)
        if args.use_temperature:
            eval_reg_loss = eval_reg_loss / nb_eval_steps
            eval_loss += 0.001 * eval_reg_loss
            eval_temp_mae = eval_temp_mae / len(eval_dataset)
            results['eval_temp_mae'] = eval_temp_mae

        results['eval_loss'] = eval_loss
        # results['eval_ppl'] = eval_ppl

        return results

    def _create_training_progress_scores(self, **kwargs):
        extra_metrics = {key: [] for key in kwargs}

        training_progress_scores = {
            "global_step": [],
            "train_loss": [],
            "eval_loss": [],
            # "train_ppl": [],
            # "eval_ppl": [],
            **extra_metrics,
        }

        return training_progress_scores

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones((sz, sz), device=self.device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(
            mask == 1, float(0.0))
        return mask

    def _get_inputs_dict(self, batch):
        if isinstance(batch[0], dict):
            inputs = {
                key: value.squeeze(1).to(self.device)
                for key, value in batch[0].items()
            }
            if not self.args.use_temperature:
                inputs["labels"] = batch[1].to(self.device)
                inputs["temperature"] = None
            else:
                inputs["labels"] = batch[1][0].to(self.device)
                inputs["temperature"] = batch[1][1].to(self.device)
        else:
            batch = tuple(t.to(self.device) for t in batch)

            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[3],
            }

        inputs['label_input'] = inputs['labels'][:, :-1]
        labels_seq_len = inputs['label_input'].shape[1]
        inputs['label_mask'] = self._generate_square_subsequent_mask(
            labels_seq_len)
        # inputs['label_padding_mask'] = (inputs['label_input'] == self.condition_label_mapping[1]['[PAD]']).transpose(0, 1)
        inputs['label_padding_mask'] = (inputs['label_input'] == self.condition_label_mapping[1]['[PAD]'])
        inputs['memory_key_padding_mask'] = (inputs['attention_mask'] == 0)
        # del inputs['labels']
        return inputs

    def _calculate_loss(self, model, inputs, loss_fct, num_labels, args):
        outputs = model(**inputs)
        # model outputs are always tuple in pytorch-transformers (see doc)
        loss = outputs[0]
        if loss_fct:
            logits = outputs[1]
            labels = inputs["labels"]

            loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
        if args.use_temperature:
            batch_temp_mae = torch.abs(inputs["temperature"] - outputs[4]).sum()
            return (loss, *outputs[1:], batch_temp_mae)
        return (loss, *outputs[1:])

    def translate_beam_search(self, model, inputs, max_len, beam):

        start_symbol = self.condition_label_mapping[1][BOS]
        step2translate = defaultdict(list)
        succ_translate = []
        translate_quene = Queue()

        memory_key_padding_mask = inputs['memory_key_padding_mask']
        del inputs['memory_key_padding_mask']
        memory = model.bert(**inputs)[0]
        ys = torch.ones(memory.size(0), 1).fill_(start_symbol).type(torch.long).to(self.device)
        cumul_score = torch.ones(memory.size(0)).type(torch.float).to(self.device).view(-1, 1)
        ys = ys.transpose(0, 1)
        cumul_score = cumul_score.transpose(0, 1)
        step_number = 0
        if self.args.use_temperature:
            translate_quene.put((ys, cumul_score, step_number, None))
        else:
            translate_quene.put((ys, cumul_score, step_number))
        while not translate_quene.empty():
            if self.args.use_temperature:
                ys, cumul_score, step_number, previous_out = translate_quene.get()
            else:
                ys, cumul_score, step_number = translate_quene.get()
            if ys.size(0) >= max_len:
                if self.args.use_temperature:
                    succ_translate.append((ys, cumul_score, step_number, previous_out))
                else:
                    succ_translate.append((ys, cumul_score, step_number))
                continue
            ys = ys.transpose(0, 1)
            cumul_score = cumul_score.transpose(0, 1)
            tgt_mask = (self._generate_square_subsequent_mask(ys.size(1)).type(torch.bool)).to(self.device)
            out, _ = model.decode(
                ys,
                memory,
                tgt_mask,
                memory_key_padding_mask=memory_key_padding_mask)
            pred = model.generator(out[:, -1])
            prob = torch.softmax(pred, dim=1)
            if self.args.use_temperature:
                previous_out = out.detach()
            if isinstance(beam, int):
                next_scores, next_words = prob.topk(beam)
            elif isinstance(beam, dict):
                next_scores, next_words = prob.topk(beam[step_number])
            else:
                raise ValueError('beam should be  \'int\' or \'dict\'.')
            step_number += 1
            for i in range(next_words.size(1)):
                _ys = torch.cat([ys, next_words[:, i].unsqueeze(1)], dim=1)

                _cumul_score = cumul_score * next_scores[:, i].unsqueeze(1)
                if self.args.use_temperature:
                    step2translate[step_number].append((_ys, _cumul_score, step_number, previous_out))
                else:
                    step2translate[step_number].append((_ys, _cumul_score, step_number))
            if isinstance(beam, int):
                thread_number = beam if step_number == 1 else beam * beam
            elif isinstance(beam, dict):
                thread_number = 1
                # if step_number != 1:
                for i in range(step_number):
                    thread_number *= beam[i]
                # thread_number = beam[step_number-1] if step_number == 1 else beam[step_number-1] * beam[step_number]
            else:
                raise ValueError('beam should be  \'int\' or \'dict\'.')
            if len(step2translate[step_number]) == thread_number:
                put_list = step2translate[step_number]
                _ys_cat = torch.cat([x[0].unsqueeze(0) for x in put_list], dim=0)  # --> (beam, batch_size, tgt_len)
                # --> (beam, tgt_len, batch_size)
                _ys_cat = _ys_cat.transpose(1, 2)
                _cumul_score_cat = torch.cat([x[1] for x in put_list], dim=1)  # --> (batch_size, beam)
                _cumul_score_cat = _cumul_score_cat.transpose(0, 1)  # --> (beam, batch_size)
                _ys_cat_sorted = torch.zeros_like(_ys_cat)
                _cumul_score_cat_sorted = torch.zeros_like(_cumul_score_cat)
                if self.args.use_temperature:
                    _one_step_previous_outs = torch.cat([x[3].unsqueeze(0) for x in put_list], dim=0)
                    _one_step_previous_outs_sorted = torch.zeros_like(_one_step_previous_outs)
                for j in range(_cumul_score_cat.size(1)):
                    dim_cumul_score_sorted, _idx = _cumul_score_cat[:, j].topk(thread_number)
                    _ys_cat_sorted[:, :, j] = _ys_cat[_idx, :, j]
                    _cumul_score_cat_sorted[:, j] = dim_cumul_score_sorted
                    if self.args.use_temperature:
                        _one_step_previous_outs_sorted[:, j] = _one_step_previous_outs[_idx, j]
                if isinstance(beam, int):
                    for n in range(beam):
                        if self.args.use_temperature:
                            translate_quene.put(
                                (_ys_cat_sorted[n],
                                 _cumul_score_cat_sorted[n].unsqueeze(0),
                                 step_number,
                                 _one_step_previous_outs_sorted[n]))
                        else:
                            translate_quene.put(
                                (_ys_cat_sorted[n],
                                 _cumul_score_cat_sorted[n].unsqueeze(0),
                                 step_number))
                elif isinstance(beam, dict):
                    for n in range(thread_number):
                        if self.args.use_temperature:
                            translate_quene.put(
                                (_ys_cat_sorted[n],
                                 _cumul_score_cat_sorted[n].unsqueeze(0),
                                 step_number,
                                 _one_step_previous_outs_sorted[n]))
                        else:
                            translate_quene.put(
                                (_ys_cat_sorted[n],
                                 _cumul_score_cat_sorted[n].unsqueeze(0),
                                 step_number))
                else:
                    raise ValueError('beam should be  \'int\' or \'dict\'.')
        _tgt_tokens = torch.cat([x[0].unsqueeze(0) for x in succ_translate], dim=0)
        _cumul_scores = torch.cat([x[1] for x in succ_translate])
        tgt_tokens = torch.zeros_like(_tgt_tokens)
        cumul_scores = torch.zeros_like(_cumul_scores)
        if self.args.use_temperature:
            _previous_outs = torch.cat(
                [x[3].unsqueeze(0) for x in succ_translate], dim=0)
            previous_outs = torch.zeros_like(_previous_outs)
        for j in range(_cumul_scores.size(1)):
            if isinstance(beam, int):
                dim_cumul_scores_sorted, _idx = _cumul_score_cat[:, j].topk(beam)
            elif isinstance(beam, dict):
                dim_cumul_scores_sorted, _idx = _cumul_score_cat[:, j].topk(thread_number)
            else:
                raise ValueError('beam should be  \'int\' or \'dict\'.')
            tgt_tokens[:, :, j] = _ys_cat[_idx, :, j]
            cumul_scores[:, j] = dim_cumul_scores_sorted

            if self.args.use_temperature:
                previous_outs[:, j] = _previous_outs[_idx, j]

        # if isinstance(beam, int):
        #     pass

        # elif isinstance(beam, dict):
        #     pass

        # else:
        #     raise ValueError('beam should be  \'int\' or \'dict\'.')
        if self.args.use_temperature:
            return tgt_tokens, cumul_scores, (memory, previous_outs)
        return tgt_tokens, cumul_scores

    def _idx2condition(self, idx):
        assert isinstance(idx, list)
        idx2condition_dict = self.condition_label_mapping[0]
        condition_list = [idx2condition_dict[x] for x in idx[1:]]
        return condition_list

    def _get_accuracy_for_one(
            self,
            one_pred,
            one_ground_truth,
            topk_get=[1, 3, 5, 10, 15],
            condition_to_calculate=['c1', 's1', 's2', 'r1', 'r2']):

        condition_item2cols = {'c1': 0, 's1': 1, 's2': 2, 'r1': 3, 'r2': 4}

        calculate_cols = [condition_item2cols[x] for x in condition_to_calculate]

        repeat_number = one_pred.size(0)
        hit_mat = one_ground_truth.unsqueeze(0).repeat(repeat_number, 1) == one_pred
        hit_mat = hit_mat[:, calculate_cols]
        overall_hit_mat = hit_mat.sum(1) == hit_mat.size(1)
        topk_hit_df = pd.DataFrame()
        for k in topk_get:
            hit_mat_k = hit_mat[:k, :]
            overall_hit_mat_k = overall_hit_mat[:k]
            topk_hit = []
            for col_idx in range(hit_mat.size(1)):
                if hit_mat_k[:, col_idx].sum() != 0:
                    topk_hit.append(1)
                else:
                    topk_hit.append(0)
            if overall_hit_mat_k.sum() != 0:
                topk_hit.append(1)
            else:
                topk_hit.append(0)
            topk_hit_df[k] = topk_hit
        # topk_hit_df.index = ['c1', 's1', 's2', 'r1', 'r2']
        return topk_hit_df

    def _calculate_batch_topk_hit(
            self,
            batch_preds,
            batch_ground_truth,
            topk_get=[1, 3, 5, 10, 15],
            condition_to_calculate=['c1', 's1', 's2', 'r1', 'r2']):
        '''
        batch_pred                         <-- tgt_tokens_list
        batch_ground_truth                 <-- inputs['labels']
        '''
        batch_preds = torch.tensor(batch_preds)[:, :, 1:].to(self.device)
        batch_ground_truth = batch_ground_truth[:, 1:-1]

        one_batch_topk_acc_mat = np.zeros((len(condition_to_calculate) + 1, 5))
        # topk_get = [1, 3, 5, 10, 15]
        for idx in range(batch_preds.size(0)):
            topk_hit_df = self._get_accuracy_for_one(
                batch_preds[idx],
                batch_ground_truth[idx],
                topk_get=topk_get,
                condition_to_calculate=condition_to_calculate)
            one_batch_topk_acc_mat += topk_hit_df.values
        return one_batch_topk_acc_mat

    def condition_beam_search(
            self,
            test_df,
            output_dir,
            beam,
            n_best=15,
            test_batch_size=8,
            verbose=True,
            silent=False,
            calculate_topk_accuracy=False,
            topk_get=[1, 3, 5, 10, 15],
            topk_results_fname='topk_accuracy.csv',
            condition_to_calculate=['c1', 's1', 's2', 'r1', 'r2']):
        self._move_model_to_device()
        model = self.model
        args = self.args
        test_output_dir = output_dir
        if not os.path.exists(test_output_dir):
            os.makedirs(test_output_dir)
        # 数据集加载只保留一种加载方式，其余全部删除
        test_examples = (
            test_df["text"].astype(str).tolist(),
            test_df["labels"].tolist(),
        )

        test_dataset = self.load_and_cache_examples(test_examples,
                                                    verbose=verbose,
                                                    no_cache=True)
        print('loaded test dataset {}'.format(
            test_dataset.examples['input_ids'].shape[0]))
        test_sampler = SequentialSampler(test_dataset)
        test_dataloader = DataLoader(
            test_dataset,
            sampler=test_sampler,
            batch_size=test_batch_size,
            num_workers=self.args.dataloader_num_workers,
        )

        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        n_batches = len(test_dataloader)
        if self.args.use_temperature:
            label_len = test_dataset.labels[0].shape[1] - 1
        else:
            label_len = test_dataset.labels.shape[1] - 1

        model.eval()
        if self.args.fp16:
            from torch.cuda import amp
        pred_conditions = []
        gt_temperatures = []
        pred_temperatures = []
        topk_acc_df = None
        test_temp_mae = 0.0
        if calculate_topk_accuracy:
            topk_acc_mat = np.zeros((len(condition_to_calculate) + 1, 5))
        for i, batch in enumerate(
                tqdm(
                    test_dataloader,
                    disable=args.silent or silent,
                    desc="Running Testing Beam Search",
                )):
            with torch.no_grad():
                inputs = self._get_inputs_dict(batch)
                encoder_input_keys = [
                    'input_ids', 'attention_mask', 'token_type_ids',
                    'memory_key_padding_mask'
                ]
                _inputs = {key: inputs[key] for key in encoder_input_keys}
                search_outputs = self.translate_beam_search(model,
                                                            _inputs,
                                                            max_len=label_len,
                                                            beam=beam)
                tgt_tokens = search_outputs[0]
                scores = search_outputs[1]
                scores = torch.log(scores)
                if self.args.use_temperature:
                    memory, outputs_hidden = search_outputs[2]
                    closest_outputs = outputs_hidden[0]
                    closest_temp = model.decode_temperature(memory=memory, decoder_output=closest_outputs)
                    pred_temperatures += closest_temp.view(-1).tolist()
                    gt_temperatures += inputs['temperature'].view(-1).tolist()
                    one_batch_test_temp_mae = torch.abs(closest_temp.view(-1) - inputs['temperature'].view(-1)).sum()
                    test_temp_mae += one_batch_test_temp_mae.item()
            tgt_tokens_list = [tgt_tokens[:, :, i].tolist() for i in range(tgt_tokens.size(-1))]
            scores_list = [scores[:, i].tolist() for i in range(scores.size(1))]

            if calculate_topk_accuracy:
                one_batch_topk_acc_mat = self._calculate_batch_topk_hit(
                    tgt_tokens_list,
                    inputs['labels'],
                    condition_to_calculate=condition_to_calculate)
                topk_acc_mat += one_batch_topk_acc_mat

            batch_results = []
            for one_group_tgt_tokens, one_group_scores in zip(tgt_tokens_list, scores_list):
                one_group_sentence = []
                for tgt_tokens, score in zip(one_group_tgt_tokens, one_group_scores):
                    one_group_sentence.append(
                        (self._idx2condition(tgt_tokens), score))
                batch_results.append(one_group_sentence[:n_best])
            pred_conditions += batch_results
        assert len(
            pred_conditions) == test_dataset.examples['input_ids'].shape[0]

        if calculate_topk_accuracy:
            from sklearn.metrics import r2_score
            topk_acc_mat /= len(test_dataloader.dataset)
            topk_acc_df = pd.DataFrame(topk_acc_mat)
            topk_acc_df.columns = [f'top-{k} accuracy' for k in topk_get]
            # topk_acc_df.index = ['c1', 's1', 's2', 'r1', 'r2', 'overall']
            topk_acc_df.index = condition_to_calculate + ['overall']
            if self.args.use_temperature:
                assert len(pred_temperatures) == test_dataset.examples['input_ids'].shape[0]
                test_temp_mae = test_temp_mae / test_dataset.examples['input_ids'].shape[0]
                test_temp_r2 = r2_score(gt_temperatures, pred_temperatures)
                topk_acc_df.loc['closest_pred_temp_mae'] = test_temp_mae
                topk_acc_df.loc['closest_pred_temp_r2'] = test_temp_r2
            topk_acc_df.to_csv(os.path.join(test_output_dir, topk_results_fname))

            return pred_conditions, pred_temperatures, topk_acc_df
        return pred_conditions, pred_temperatures

    def condition_beam_search_supercls(
            self,
            test_df,
            output_dir,
            beam,
            n_best=15,
            test_batch_size=8,
            verbose=True,
            silent=False,
            calculate_topk_accuracy=False,
            topk_get=[1, 3, 5, 10, 15],
            topk_results_fname='topk_accuracy.csv',
            condition_to_calculate=['c1', 's1', 's2', 'r1', 'r2']):
        self._move_model_to_device()
        model = self.model
        args = self.args
        test_output_dir = output_dir
        if not os.path.exists(test_output_dir):
            os.makedirs(test_output_dir)
        # 数据集加载只保留一种加载方式，其余全部删除
        test_examples = (
            test_df["text"].astype(str).tolist(),
            test_df["labels"].tolist(),
        )

        test_dataset = self.load_and_cache_examples(test_examples,
                                                    verbose=verbose,
                                                    no_cache=True)
        print('loaded test dataset {}'.format(
            test_dataset.examples['input_ids'].shape[0]))
        test_sampler = SequentialSampler(test_dataset)
        test_dataloader = DataLoader(
            test_dataset,
            sampler=test_sampler,
            batch_size=test_batch_size,
            num_workers=self.args.dataloader_num_workers,
        )

        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        n_batches = len(test_dataloader)
        if self.args.use_temperature:
            label_len = test_dataset.labels[0].shape[1] - 1
        else:
            label_len = test_dataset.labels.shape[1] - 1

        model.eval()
        if self.args.fp16:
            from torch.cuda import amp
        pred_conditions = []
        gt_temperatures = []
        pred_temperatures = []
        topk_acc_df = None
        test_temp_mae = 0.0
        if calculate_topk_accuracy:
            topk_acc_mat = np.zeros((len(condition_to_calculate) + 1, 5))
        for i, batch in enumerate(
                tqdm(
                    test_dataloader,
                    disable=args.silent or silent,
                    desc="Running Testing Beam Search",
                )):
            with torch.no_grad():
                inputs = self._get_inputs_dict(batch)
                encoder_input_keys = [
                    'input_ids', 'attention_mask', 'token_type_ids',
                    'memory_key_padding_mask'
                ]
                _inputs = {key: inputs[key] for key in encoder_input_keys}
                search_outputs = self.translate_beam_search(model,
                                                            _inputs,
                                                            max_len=label_len,
                                                            beam=beam)
                tgt_tokens = search_outputs[0]
                scores = search_outputs[1]
                scores = torch.log(scores)
                if self.args.use_temperature:
                    memory, outputs_hidden = search_outputs[2]
                    closest_outputs = outputs_hidden[0]
                    closest_temp = model.decode_temperature(
                        memory=memory, decoder_output=closest_outputs)
                    pred_temperatures += closest_temp.view(-1).tolist()
                    gt_temperatures += inputs['temperature'].view(-1).tolist()
                    one_batch_test_temp_mae = torch.abs(
                        closest_temp.view(-1) -
                        inputs['temperature'].view(-1)).sum()
                    test_temp_mae += one_batch_test_temp_mae.item()
            tgt_tokens_list = [
                tgt_tokens[:, :, i].tolist()
                for i in range(tgt_tokens.size(-1))
            ]
            scores_list = [
                scores[:, i].tolist() for i in range(scores.size(1))
            ]

            if calculate_topk_accuracy:

                tgt_tokens_list_supercls = []

                def change_token_idx(token, map_dict):
                    if token in map_dict:
                        return map_dict[token]
                    else:
                        return self.condition_label_mapping[1][token]

                for one_sample_results in tgt_tokens_list:
                    one_sample_results_supercls = []
                    for pred in one_sample_results:
                        pred_tokens = [
                            self.condition_label_mapping[0][x] for x in pred
                        ]
                        pred_supercls = [
                            self.condition_label_mapping[1][
                                pred_tokens[0]],  # BOS
                            self.condition_label_mapping[1][
                                pred_tokens[1]],  # c1
                            change_token_idx(pred_tokens[2],
                                             self.uspto_solvent_to_cls_idx) +
                            500,  # s1
                            change_token_idx(pred_tokens[3],
                                             self.uspto_solvent_to_cls_idx) +
                            500,  # s2
                            change_token_idx(pred_tokens[4],
                                             self.uspto_reagent_to_cls_idx) +
                            1000,  # r1
                            change_token_idx(pred_tokens[5],
                                             self.uspto_reagent_to_cls_idx) +
                            1000,  # r2
                        ]
                        one_sample_results_supercls.append(pred_supercls)
                    tgt_tokens_list_supercls.append(
                        one_sample_results_supercls)
                labels_supercls = []
                for one_label in inputs['labels'].tolist():
                    one_tokens = [
                        self.condition_label_mapping[0][x] for x in one_label
                    ]
                    one_label_supercls = [
                        self.condition_label_mapping[1][one_tokens[0]],  # BOS
                        self.condition_label_mapping[1][one_tokens[1]],  # c1
                        self.uspto_solvent_to_cls_idx[one_tokens[2]] +
                        500,  # s1
                        self.uspto_solvent_to_cls_idx[one_tokens[3]] +
                        500,  # s2
                        self.uspto_reagent_to_cls_idx[one_tokens[4]] +
                        1000,  # r1
                        self.uspto_reagent_to_cls_idx[one_tokens[5]] +
                        1000,  # r1
                        self.condition_label_mapping[1][one_tokens[6]],  # EOS
                    ]
                    labels_supercls.append(one_label_supercls)

                labels_supercls = torch.tensor(labels_supercls,
                                               device=inputs['labels'].device)

                one_batch_topk_acc_mat = self._calculate_batch_topk_hit(
                    tgt_tokens_list_supercls,
                    labels_supercls,
                    condition_to_calculate=condition_to_calculate)
                topk_acc_mat += one_batch_topk_acc_mat

            batch_results = []
            for one_group_tgt_tokens, one_group_scores in zip(
                    tgt_tokens_list, scores_list):
                one_group_sentence = []
                for tgt_tokens, score in zip(one_group_tgt_tokens,
                                             one_group_scores):
                    one_group_sentence.append(
                        (self._idx2condition(tgt_tokens), score))
                batch_results.append(one_group_sentence[:n_best])
            pred_conditions += batch_results
        assert len(
            pred_conditions) == test_dataset.examples['input_ids'].shape[0]

        if calculate_topk_accuracy:
            from sklearn.metrics import r2_score
            topk_acc_mat /= len(test_dataloader.dataset)
            topk_acc_df = pd.DataFrame(topk_acc_mat)
            topk_acc_df.columns = [f'top-{k} accuracy' for k in topk_get]
            topk_acc_df.index = condition_to_calculate + ['overall']
            if self.args.use_temperature:
                assert len(pred_temperatures
                           ) == test_dataset.examples['input_ids'].shape[0]
                test_temp_mae = test_temp_mae / test_dataset.examples[
                    'input_ids'].shape[0]
                test_temp_r2 = r2_score(gt_temperatures, pred_temperatures)
                topk_acc_df.loc['closest_pred_temp_mae'] = test_temp_mae
                topk_acc_df.loc['closest_pred_temp_r2'] = test_temp_r2
            topk_acc_df.to_csv(
                os.path.join(test_output_dir, topk_results_fname))

            return pred_conditions, pred_temperatures, topk_acc_df
        return pred_conditions, pred_temperatures

    def view_condition_bert_attentions(self,
                                       input_rxn_smiles,
                                       heatmap=True,
                                       fig_save_path=None,
                                       figsize=(46, 20),
                                       output_prediction=False,
                                       output_demo_fname=None,
                                       show_attention_fig=False,
                                       mean_attn=False):
        from models.utils import condition_bert_head_view
        from models.utils import condition_bert_head_heatmap
        start_symbol = self.condition_label_mapping[1][BOS]
        example_df = pd.DataFrame({
            'text': [input_rxn_smiles],
            'labels': [[0, 0, 0, 0, 0]],
        })
        tokenizer = self.tokenizer
        input_tokens = tokenizer.tokenize(input_rxn_smiles)
        self._move_model_to_device()
        model = self.model
        model.output_attentions = True
        args = self.args
        # test_output_dir = output_dir
        # 数据集加载只保留一种加载方式，其余全部删除
        test_examples = (
            example_df["text"].astype(str).tolist(),
            example_df["labels"].tolist(),
        )
        input_dataset = self.load_and_cache_examples(test_examples,
                                                     verbose=False,
                                                     no_cache=True)
        test_sampler = SequentialSampler(input_dataset)
        test_dataloader = DataLoader(
            input_dataset,
            sampler=test_sampler,
            batch_size=1,
            num_workers=self.args.dataloader_num_workers,
        )

        model.eval()
        for i, batch in enumerate(test_dataloader):
            with torch.no_grad():
                inputs = self._get_inputs_dict(batch)
                encoder_input_keys = [
                    'input_ids', 'attention_mask', 'token_type_ids'
                ]
                memory_key_padding_mask = inputs['memory_key_padding_mask']
                _inputs = {key: inputs[key] for key in encoder_input_keys}
                # _inputs['memory_key_padding_mask'] = (_inputs['attention_mask'] == 0)
                memory = model.bert(**_inputs)[0]
                # memory = memory[:,:,:len(input_tokens)+2]
                ys = torch.ones(memory.size(0), 1).fill_(start_symbol).type(
                    torch.long).to(self.device)
                for i in range(6):
                    tgt_mask = (self._generate_square_subsequent_mask(
                        ys.size(1)).type(torch.bool)).to(self.device)
                    out, attention_weights = model.decode(
                        ys,
                        memory,
                        tgt_mask,
                        memory_key_padding_mask=memory_key_padding_mask)
                    pred = model.generator(out[:, -1])
                    prob = torch.softmax(pred, dim=1)
                    _, next_word = prob.topk(1)
                    next_word = next_word.item()
                    ys = torch.cat(
                        [ys, torch.ones(1, 1).type_as(ys).fill_(next_word)],
                        dim=1)
        predicted_conditions = [
                                   self.condition_label_mapping[0][x] for x in ys.tolist()[0]
                               ][1:-1]
        attention_weights = [
            x[:, :, :, :len(input_tokens) + 2] for x in attention_weights
        ]
        # for i in range(attention_weights.size(0)):
        #     for j in range(attention_weights.size(1)):
        #         for k in range(attention_weights.size(2)):
        #             attention_weights[i, j, k] = attention_weights[i, j, k] / attention_weights[i, j, k].sum()
        attention_weights = [
            x[:, :, :-1, 1:-1].transpose(2, 3) for x in attention_weights
        ]
        attention_weights = torch.cat(attention_weights, dim=0)
        if output_demo_fname:
            if not os.path.exists(os.path.dirname(output_demo_fname)):
                os.makedirs(os.path.dirname(output_demo_fname))
            attention_weights_numpy = [x.tolist() for x in attention_weights]
            demo_data = (attention_weights_numpy, input_tokens,
                         predicted_conditions, input_rxn_smiles)
            with open(output_demo_fname, 'wb') as f:
                pickle.dump(demo_data, f)

        if show_attention_fig:
            viz_attention_reaction(attention_weights=attention_weights,
                                   src_tokens=input_tokens,
                                   tgt_tokens=predicted_conditions,
                                   rxn_smiles=input_rxn_smiles,
                                   mean_attn=mean_attn)

        if heatmap:
            condition_bert_head_heatmap(attention_weights,
                                        src_tokens=input_tokens,
                                        tgt_tokens=predicted_conditions,
                                        fig_save_path=fig_save_path,
                                        figsize=figsize)

            if output_prediction:
                return None, predicted_conditions
            else:
                return None, None
        else:
            html = condition_bert_head_view(attention_weights,
                                            src_tokens=input_tokens,
                                            tgt_tokens=predicted_conditions,
                                            html_action='return')
            if output_prediction:
                return html, predicted_conditions
            else:
                return html, None

    def analyze_function_group_attention_with_condition(
            self,
            test_df,
            test_batch_size=8,
            subgraph_fpath=None,
            analysis_results_save_path=None):
        if os.path.exists(analysis_results_save_path):
            print('Reading calculated analysis results...')
            with open(
                    os.path.join(analysis_results_save_path,
                                 'score_map_condition_type_dict.pkl'),
                    'rb') as f:
                score_map_condition_type_dict = pickle.load(f)
            return score_map_condition_type_dict
        print('Analysing function group attention with condition...')
        if subgraph_fpath == None:
            raise ValueError()
        subgraph_data_df = pd.read_csv(subgraph_fpath, header=None)
        subgraph_data_df.columns = ['smiles', 'rxn_split_count']
        subgraph_smiles = subgraph_data_df['smiles'].tolist()
        print(subgraph_smiles)
        self._move_model_to_device()
        model = self.model
        args = self.args

        # 数据集加载只保留一种加载方式，其余全部删除
        test_examples = (
            test_df["text"].astype(str).tolist(),
            test_df["labels"].tolist(),
        )

        test_dataset = self.load_and_cache_examples(test_examples,
                                                    verbose=False,
                                                    no_cache=True)
        print('loaded test dataset {}'.format(
            test_dataset.examples['input_ids'].shape[0]))
        test_sampler = SequentialSampler(test_dataset)
        test_dataloader = DataLoader(
            test_dataset,
            sampler=test_sampler,
            batch_size=test_batch_size,
            num_workers=self.args.dataloader_num_workers,
        )

        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        n_batches = len(test_dataloader)
        if self.args.use_temperature:
            label_len = test_dataset.labels[0].shape[1] - 1
        else:
            label_len = test_dataset.labels.shape[1] - 1
        start_symbol = self.condition_label_mapping[1][BOS]

        tokenizer = self.tokenizer
        input_tokens_list = [
            tokenizer.tokenize(x)
            for x in tqdm(test_df["text"].astype(str).tolist())
        ]

        # test_output_dir = output_dir
        # 数据集加载只保留一种加载方式，其余全部删除

        predicted_labels = []
        test_attention_weights = []
        model.eval()
        for batch in tqdm(
                test_dataloader,
                desc="Running Testing Greedy Search",
        ):
            with torch.no_grad():
                inputs = self._get_inputs_dict(batch)
                encoder_input_keys = [
                    'input_ids', 'attention_mask', 'token_type_ids'
                ]
                memory_key_padding_mask = inputs['memory_key_padding_mask']
                _inputs = {key: inputs[key] for key in encoder_input_keys}
                # _inputs['memory_key_padding_mask'] = (_inputs['attention_mask'] == 0)
                memory = model.bert(**_inputs)[0]
                # memory = memory[:,:,:len(input_tokens)+2]
                ys = torch.ones(memory.size(0), 1).fill_(start_symbol).type(
                    torch.long).to(self.device)
                for i in range(6):
                    tgt_mask = (self._generate_square_subsequent_mask(
                        ys.size(1)).type(torch.bool)).to(self.device)
                    out, attention_weights = model.decode(
                        ys,
                        memory,
                        tgt_mask,
                        memory_key_padding_mask=memory_key_padding_mask
                    )  # attention_weights --> (batch_size, nhead, tgt_size, source_size)
                    pred = model.generator(out[:, -1])
                    prob = torch.softmax(pred, dim=1)
                    _, next_word = prob.topk(1)
                    # next_word = next_word.item()
                    ys = torch.cat([ys, next_word], dim=1)
                predicted_labels.extend(ys.tolist())
                batch_attention_weights = torch.cat(
                    [x.unsqueeze(1) for x in attention_weights], dim=1).cpu()
                batch_attention_weights = [
                    x.squeeze() for x in torch.chunk(
                        batch_attention_weights, test_batch_size, dim=0)
                ]
                test_attention_weights.extend(batch_attention_weights)

        predicted_conditions = []
        for one_prediction in tqdm(
                predicted_labels,
                desc="Converting Labels to Conditions",
        ):
            predicted_conditions.append([
                self.condition_label_mapping[0][x]
                for x in one_prediction[1:-1]
            ])

        # test_attention_weights_rm_masked_attn --> list(n_sample * (nlayer, nhead, source_size, tgt_size))

        test_attention_weights_rm_masked_attn = []
        for input_tokens, attention_weights in tqdm(
                zip(input_tokens_list, test_attention_weights),
                desc='Remove Masked Attentions',
                total=len(input_tokens_list)):
            attention_weights = attention_weights[:, :, :, :len(input_tokens) +
                                                            2]
            attention_weights = attention_weights[:, :, :-1,
                                1:-1].transpose(2, 3)
            test_attention_weights_rm_masked_attn.append(attention_weights)

        rxn_smiles = test_df["text"].astype(str).tolist()

        score_map_condition_type_dict = analyze_subgraph_attention_with_condition(
            input_tokens_list, rxn_smiles, predicted_conditions,
            test_attention_weights_rm_masked_attn, subgraph_smiles)
        # analysis_results_save_path = './data/bert_condition_data/USPTO_condition_frag_analysis_results.pkl'
        if analysis_results_save_path is not None:
            if not os.path.exists(analysis_results_save_path):
                os.makedirs(analysis_results_save_path)
            with open(
                    os.path.join(analysis_results_save_path,
                                 'score_map_condition_type_dict.pkl'),
                    'wb') as f:
                pickle.dump(score_map_condition_type_dict, f)
            with open(
                    os.path.join(analysis_results_save_path,
                                 'model_config.pkl'), 'wb') as f:
                pickle.dump(self.args, f)
            subgraph_data_df.to_csv(os.path.join(analysis_results_save_path,
                                                 'subgraph.csv'),
                                    index=False)

        return score_map_condition_type_dict


if __name__ == '__main__':
    config_file = './data/bert_condition_data/config_debug.yaml'
    print_args(config_file)
    config = yaml.load(open(config_file, "r"), Loader=yaml.FullLoader)
    model_args = config['model_args']
    model_args.update({'output_attention': True})
    dataset_args = config['dataset_args']
    dataset_df, condition_label_mapping = load_dataset(**dataset_args)
    model_args['decoder_args'].update({
        'tgt_vocab_size':
            len(condition_label_mapping[0]),
        'condition_label_mapping':
            condition_label_mapping
    })
    model = ParrotConditionPredictionModel("bert",
                                           model_args['best_model_dir'],
                                           args=model_args,
                                           use_cuda=torch.cuda.is_available())

    model.view_condition_bert_attentions(
        input_rxn_smiles=dataset_df['canonical_rxn'][0],
        heatmap=True,
        show_attention_fig=True,
        mean_attn=True)
    # model.view_condition_bert_attentions(
    #     input_rxn_smiles=dataset_df['canonical_rxn'][0], heatmap=False)
