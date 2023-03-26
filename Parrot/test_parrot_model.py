from argparse import ArgumentParser
import json
import torch
import yaml
from models.parrot_model import ParrotConditionPredictionModel

from models.utils import load_dataset, load_test_dataset

torch.multiprocessing.set_sharing_strategy('file_system')


def main(parser_args, debug=False):

    config = yaml.load(open(parser_args.config_path, "r"), Loader=yaml.FullLoader)
    model_args = config['model_args']
    dataset_args = config['dataset_args']

    try:
        model_args['use_temperature'] = dataset_args['use_temperature']
        print('Using Temperature:', model_args['use_temperature'])
    except:
        print('No temperature information is specified!')

    dataset_df, condition_label_mapping = load_dataset(**dataset_args)
    model_args['decoder_args'].update({
        'tgt_vocab_size': len(condition_label_mapping[0]),
        'condition_label_mapping': condition_label_mapping
    })

    if model_args['model_type']:
        model_type = model_args['model_type']
    else:
        model_type = 'bert'

    trained_path = model_args['best_model_dir']
    model = ParrotConditionPredictionModel(
        model_type,
        trained_path,
        args=model_args,
        use_cuda=True if parser_args.gpu >= 0 else False,
        cuda_device=parser_args.gpu)

    testset_args = config['testset_args']
    if 'dataset_root' in testset_args:
        testset_df, _ = load_test_dataset(
            dataset_root=testset_args['dataset_root'],
            database_fname=testset_args['database_fname'],
            condition_label_mapping=condition_label_mapping)
    else:
        testset_df = dataset_df.loc[dataset_df['dataset'] == 'test']
    test_df = testset_df

    if testset_args['testset_distinguish_catalyst']:
        test_df_catalsyt_na = test_df.loc[test_df['catalyst1'].isna()]
        test_df_catalsyt_na = test_df_catalsyt_na[['canonical_rxn', 'condition_labels']]
        test_df_catalsyt_na.columns = ['text', 'labels']
        pred_conditions, pred_temperatures, topk_acc_df_catalyst_na = model.condition_beam_search(
            test_df_catalsyt_na,
            output_dir=model_args['best_model_dir'],
            beam={
                0: 1,
                1: 3,
                2: 1,
                3: 5,
                4: 1
            },
            test_batch_size=8,
            calculate_topk_accuracy=True,
            topk_results_fname='catalyst_na_topk_accuracy_s1r1.csv',
            condition_to_calculate=['s1', 'r1'])
        print(topk_acc_df_catalyst_na)

        test_df_catalsyt_have = test_df.loc[~test_df['catalyst1'].isna()]
        test_df_catalsyt_have = test_df_catalsyt_have[['canonical_rxn', 'condition_labels']]
        test_df_catalsyt_have.columns = ['text', 'labels']
        pred_conditions, pred_temperatures, topk_acc_df_catalyst_have = model.condition_beam_search(
            test_df_catalsyt_have,
            output_dir=model_args['best_model_dir'],
            beam={
                0: 2,
                1: 3,
                2: 1,
                3: 3,
                4: 1
            },
            test_batch_size=8,
            calculate_topk_accuracy=True,
            topk_results_fname='have_catalyst_topk_accuracy_c1s1r1.csv',
            condition_to_calculate=['c1', 's1', 'r1'])
        print(topk_acc_df_catalyst_have)

    else:
        test_df = test_df[['canonical_rxn', 'condition_labels']]
        test_df.columns = ['text', 'labels']
        print('test dataset number: {}'.format(len(test_df)))

        if 'condition_to_calculate' in testset_args:
            topk_fmark = ''.join(testset_args['test_condition_items'])
            topk_results_fname = testset_args['topk_results_fname'].replace('.csv', f'_{topk_fmark}.csv')
        else:
            topk_results_fname = testset_args['topk_results_fname']

        beam = testset_args['beam']

        pred_conditions, pred_temperatures, topk_acc_df = model.condition_beam_search(
            test_df,
            output_dir=model_args['best_model_dir'],
            beam=beam,
            test_batch_size=model_args['eval_batch_size'],
            calculate_topk_accuracy=True,
            topk_results_fname=topk_results_fname,
            condition_to_calculate=testset_args['test_condition_items'])
        print(topk_acc_df)


if __name__ == '__main__':

    parser = ArgumentParser('Test Arguements')
    parser.add_argument('--gpu', default=0, help='GPU device to use', type=int)
    parser.add_argument('--config_path',
                        default='configs/config_uspto_condition.yaml',
                        help='Path to config file',
                        type=str)

    parser_args = parser.parse_args()
    main(parser_args, debug=False)
