import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from transformers import get_scheduler, EncoderDecoderModel, EncoderDecoderConfig, AutoTokenizer, AutoConfig, AutoModel
from transformers.models.bert.modeling_bert import BertLMPredictionHead
from transformers.utils import ModelOutput
from . import utils


def get_model(args, enc_tokenizer=None, dec_tokenizer=None):
    if args.template_based:
        assert args.decoder is None and not args.decoder_pretrained
        if args.encoder_pretrained:
            encoder = AutoModel.from_pretrained(pretrained_model_name_or_path=args.encoder)
        else:
            encoder_config = AutoConfig.from_pretrained(args.encoder)
            encoder = AutoModel.from_config(encoder_config)
        template_head = TemplatePredictionHead(encoder.config.hidden_size, len(dec_tokenizer[0]), len(dec_tokenizer[1]))
        model = TemplateBasedModel(encoder, template_head)
    else:
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
        encoder = model.encoder
    if args.max_length > encoder.config.max_position_embeddings:
        utils.expand_position_embeddings(encoder, args.max_length)
    if args.encoder_tokenizer == 'smiles_text':
        utils.expand_word_embeddings(encoder, len(enc_tokenizer))
    return model


def get_mlm_head(args, model):
    if args.mlm_layer == 'linear':
        mlm_head = nn.Linear(model.encoder.config.hidden_size, model.encoder.config.vocab_size)
    elif args.mlm_layer == 'mlp':
        mlm_head = BertLMPredictionHead(model.encoder.config)
    else:
        raise NotImplementedError
    return mlm_head


class TemplateBasedModel(nn.Module):
    def __init__(self, encoder, template_head):
        super().__init__()
        self.encoder = encoder
        self.template_head = template_head

    def forward(self, **inputs):
        encoder_output = self.encoder(**{k: v for k, v in inputs.items()
            if not k.startswith('decoder_') and k not in ['atom_indices', 'bonds']})
        atom_hidden_states = []
        for hidden_states, atom_indices in zip(encoder_output.last_hidden_state, inputs['atom_indices']):
            atom_hidden_states.append(hidden_states[atom_indices])
        atom_hidden_states = pad_sequence(atom_hidden_states, batch_first=True)
        return ModelOutput(logits=self.template_head(atom_hidden_states), encoder_last_hidden_state=encoder_output.last_hidden_state)


class TemplatePredictionHead(nn.Module):
    def __init__(self, input_size, num_atom_templates, num_bond_templates):
        super().__init__()
        self.atom_template_head = nn.Linear(input_size, num_atom_templates + 1)
        self.bond_template_head = BondTemplatePredictor(input_size, num_bond_templates)

    def forward(self, input_states):
        """
        Input: [B x] L x d_in
        Output: [B x] L x n_a, [B x] L x L x n_b
        """
        return self.atom_template_head(input_states), self.bond_template_head(input_states)


class BondTemplatePredictor(nn.Module):
    def __init__(self, input_size, num_bond_templates):
        super().__init__()
        self.linear = nn.Linear(2 * input_size, num_bond_templates + 1)

    def forward(self, input_states):
        concat_pair_shape = input_states.shape[:-1] + input_states.shape[-2:]
        concat_pairs = torch.cat((input_states.unsqueeze(-2).expand(concat_pair_shape),
                                  input_states.unsqueeze(-3).expand(concat_pair_shape)),
                                 dim=-1)
        return self.linear(concat_pairs)
