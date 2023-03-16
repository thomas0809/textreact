from argparse import ArgumentParser
import json
import pandas as pd
import torch
import yaml
from pandarallel import pandarallel
from models.parrot_model import ParrotConditionPredictionModel
from preprocess_script.uspto_script.utils import canonicalize_smiles
from models.utils import caonicalize_rxn_smiles, get_output_results, inference_load

torch.multiprocessing.set_sharing_strategy('file_system')





def main(parser_args, debug=False):
    pandarallel.initialize(nb_workers=parser_args.num_workers,
                           progress_bar=True)
    config = yaml.load(open(parser_args.config_path, "r"),
                       Loader=yaml.FullLoader)

    print(
        '\n########################\nInference configs:\n########################\n'
    )
    print(yaml.dump(config))
    print('########################\n')
    model_args = config['model_args']
    dataset_args = config['dataset_args']

    try:
        model_args['use_temperature'] = dataset_args['use_temperature']
        print('Using Temperature:', model_args['use_temperature'])
    except:
        print('No temperature information is specified!')

    condition_label_mapping = inference_load(**dataset_args)
    model_args['decoder_args'].update({
        'tgt_vocab_size':
        len(condition_label_mapping[0]),
        'condition_label_mapping':
        condition_label_mapping
    })

    trained_path = model_args['best_model_dir']
    model = ParrotConditionPredictionModel(
        "bert",
        trained_path,
        args=model_args,
        use_cuda=True if parser_args.gpu >= 0 else False,
        cuda_device=parser_args.gpu)

    with open(parser_args.input_path, 'r', encoding='utf-8') as f:
        input_rxn_smiles = [x.strip() for x in f.readlines()]

    test_df = pd.DataFrame({
        'text': input_rxn_smiles,
        'labels': [[0] * 7] * len(input_rxn_smiles) if not model_args['use_temperature'] else [[0] * 8] * len(input_rxn_smiles) 
    })
    print('Caonicalize reaction smiles and remove invalid reaction...')
    test_df['text'] = test_df.text.parallel_apply(
        lambda x: caonicalize_rxn_smiles(x))
    test_df = test_df.loc[test_df['text'] != ''].reset_index(drop=True)

    inference_args = config['inference_args']
    config['thread_count'] = parser_args.num_workers

    beam = inference_args['beam']
    pred_conditions, pred_temperatures = model.condition_beam_search(
        test_df,
        output_dir=model_args['best_model_dir'],
        beam=beam,
        test_batch_size=8,
        calculate_topk_accuracy=False)

    output_df = get_output_results(test_df.text.tolist(), pred_conditions,
                              pred_temperatures)
    output_df = output_df.round(4)
    output_df.to_csv(parser_args.output_path, index=False)

    print('Done!')


if __name__ == '__main__':

    parser = ArgumentParser('Test Arguements')
    parser.add_argument('--gpu', default=0, help='GPU device to use', type=int)
    parser.add_argument('--config_path',
                        default='configs/config_inference_use_reaxys.yaml',
                        help='Path to config file',
                        type=str)

    parser.add_argument('--input_path',
                        default='test_files/input_demo.txt',
                        help='Path to input file (txt)',
                        type=str)
    parser.add_argument('--output_path',
                        default='test_files/predicted_conditions.csv',
                        help='Path to output file (csv)',
                        type=str)
    parser.add_argument('--inference_batch_size',
                        default=8,
                        help='Batch size',
                        type=int)
    parser.add_argument('--num_workers',
                        default=10,
                        help='number workers',
                        type=int)

    parser_args = parser.parse_args()
    main(parser_args, debug=False)
