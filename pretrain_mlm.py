import os
import pandas as pd
import yaml
from argparse import ArgumentParser
from models.utils import generate_vocab
from rxnfp.models import SmilesLanguageModelingModel


def main(parser_args, debug):
    train_args = yaml.load(open(parser_args.config_path, "r"),
                           Loader=yaml.FullLoader)
    vocab_path = './dataset/pretrain_data/vocab.txt'
    train_file = train_args['train_file']
    eval_file = train_args['eval_file']
    if not os.path.exists(vocab_path):
        with open(train_file, 'r', encoding='utf-8') as f:
            train_rxn = [x.strip() for x in f.readlines()]
        with open(eval_file, 'r', encoding='utf-8') as f:
            eval_rxn = [x.strip() for x in f.readlines()]
        rxn_smiles = train_rxn + eval_rxn
        generate_vocab(rxn_smiles, vocab_path)
    print(
        '########################\nTraining configs:\n########################\n'
    )
    print(yaml.dump(train_args))
    print('########################\n')

    model = SmilesLanguageModelingModel(
        model_type='bert',
        model_name=None,
        args=train_args,
        use_cuda=True if parser_args.gpu >= 0 else False,
        cuda_device=parser_args.gpu)

    model.train_model(train_file=train_file, eval_file=eval_file)
    print('Done!')


if __name__ == '__main__':
    parser = ArgumentParser('Training Masked Language Modeling Arguements')
    parser.add_argument('--gpu', default=0, help='GPU device to use', type=int)
    parser.add_argument('--config_path',
                        default='configs/pretrain_mlm_config.yaml',
                        help='Path to config file',
                        type=str)

    parser_args = parser.parse_args()
    debug = False
    main(parser_args, debug=debug)
