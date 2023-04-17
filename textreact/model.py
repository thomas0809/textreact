import torch.nn as nn
from transformers import get_scheduler, EncoderDecoderModel, EncoderDecoderConfig, AutoTokenizer, AutoConfig, AutoModel
from transformers.models.bert.modeling_bert import BertLMPredictionHead
from . import utils


def get_model(args, enc_tokenizer=None):
    if args.encoder_pretrained and args.decoder_pretrained:
        model = EncoderDecoderModel.from_encoder_decoder_pretrained(
            encoder_pretrained_model_name_or_path=args.encoder, decoder_pretrained_model_name_or_path=args.decoder)
    else:
        encoder_config = AutoConfig.from_pretrained(args.encoder)
        decoder_config = AutoConfig.from_pretrained(args.decoder)
        config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder_config, decoder_config)
        model = EncoderDecoderModel(config=config)
        if args.encoder_pretrained:
            encoder = AutoModel.from_pretrained(args.encoder)
            model.encoder = encoder
    if args.max_length > model.encoder.config.max_position_embeddings:
        utils.expand_position_embeddings(model.encoder, args.max_length)
    if args.encoder_tokenizer == 'smiles_text':
        utils.expand_word_embeddings(model.encoder, len(enc_tokenizer))
    return model


def get_mlm_head(args, model):
    if args.mlm_layer == 'linear':
        mlm_head = nn.Linear(model.encoder.config.hidden_size, model.encoder.config.vocab_size)
    elif args.mlm_layer == 'mlp':
        mlm_head = BertLMPredictionHead(model.encoder.config)
    else:
        raise NotImplementedError
    return mlm_head
