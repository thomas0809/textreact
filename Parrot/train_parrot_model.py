import torch
import yaml
from argparse import ArgumentParser
from models.parrot_model import ParrotConditionPredictionModel
from models.utils import load_dataset

torch.multiprocessing.set_sharing_strategy('file_system')


def main(parser_args, debug):
    config = yaml.load(open(parser_args.config_path, "r"), Loader=yaml.FullLoader)
    print('\n########################\nConfigs:\n########################\n')
    print(yaml.dump(config))
    print('########################\n')
    model_args = config['model_args']
    dataset_args = config['dataset_args']
    try:
        model_args['use_temperature'] = dataset_args['use_temperature']
        print('Using Temperature:', model_args['use_temperature'])
    except:
        print('No temperature information is specified!')

    database_df, condition_label_mapping = load_dataset(**dataset_args)
    model_args['decoder_args'].update({
        'tgt_vocab_size': len(condition_label_mapping[0]),
        'condition_label_mapping': condition_label_mapping
    })

    train_df = database_df.loc[database_df['dataset'] == 'train']
    train_df = train_df[['canonical_rxn', 'condition_labels']]
    train_df.columns = ['text', 'labels']
    print(train_df.head())
    print('train dataset number: {}'.format(len(train_df)))

    eval_df = database_df.loc[database_df['dataset'] == 'val']
    eval_df = eval_df[['canonical_rxn', 'condition_labels']]
    eval_df.columns = ['text', 'labels']
    print('validation dataset number: {}'.format(len(eval_df)))

    if model_args['model_type']:
        model_type = model_args['model_type']
    else:
        model_type = 'bert'

    if model_args['pretrained_path']:
        pretrained_path = model_args['pretrained_path']
    else:
        pretrained_path = None
    model = ParrotConditionPredictionModel(
        model_type,
        pretrained_path,
        args=model_args,
        use_cuda=True if parser_args.gpu >= 0 else False,
        cuda_device=parser_args.gpu)
    model.train_model(train_df, eval_df=eval_df)

    print('Done!')


if __name__ == '__main__':
    parser = ArgumentParser('Training Arguements')
    parser.add_argument('--gpu', default=0, help='GPU device to use', type=int)
    parser.add_argument('--config_path',
                        default='configs/config_uspto_condition.yaml',
                        help='Path to config file',
                        type=str)

    parser_args = parser.parse_args()
    debug = False
    main(parser_args, debug)
