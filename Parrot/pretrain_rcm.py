import os
import pandas as pd
import yaml
from argparse import ArgumentParser
from models.utils import generate_vocab
from models.pretrain_model import RXNCenterModelingModel


def main(parser_args, debug):
    train_args = yaml.load(open(parser_args.config_path, "r"),
                           Loader=yaml.FullLoader)

    database_file = train_args[
        'database_file'] if not debug else './dataset/pretrain_data/rxn_center_modeling_debug.pkl'
    database = pd.read_pickle(database_file)
    train_df = database.loc[database['dataset'] == 'train']
    train_df = train_df[['clean_map_rxn', 'all_reaction_center_index_mask']]
    train_df.columns = ['text', 'labels']
    val_df = database.loc[database['dataset'] == 'val']
    val_df = val_df[['clean_map_rxn', 'all_reaction_center_index_mask']]
    val_df.columns = ['text', 'labels']
    vocab_path = train_args['vocab_path']
    if not os.path.exists(vocab_path):
        rxn_smiles = database['can_rxn_smiles'].tolist()
        generate_vocab(rxn_smiles, vocab_path)

    if debug:
        train_args['wandb_project'] = 'test_load'
        train_args['output_dir'] = './out/debug'
        train_args['best_model_dir'] = './outputs/debug'
    print(
        '########################\nTraining configs:\n########################\n'
    )
    print(yaml.dump(train_args))
    print('########################\n')

    model = RXNCenterModelingModel(
        model_name=None,
        args=train_args,
        use_cuda=True if parser_args.gpu >= 0 else False,
        cuda_device=parser_args.gpu)
    print(model.model)
    model.train_model(train_df=train_df, eval_df=val_df)
    print('Done!')


if __name__ == '__main__':
    parser = ArgumentParser('Training Reaction Center Modeling Arguements')
    parser.add_argument('--gpu', default=0, help='GPU device to use', type=int)
    parser.add_argument('--config_path',
                        default='configs/pretrain_rcm_config.yaml',
                        help='Path to config file',
                        type=str)

    parser_args = parser.parse_args()
    debug = False
    main(parser_args, debug=debug)
