'''
半成品，按照已有的反应条件去smiles数据集里面匹配试剂等反应条件分子
'''
from collections import defaultdict
import pickle
import pandas as pd
from tqdm import tqdm
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from utils import canonicalize_smiles, covert_to_series

NONE_PANDDING = '[None]'

class AssignmentCondition:
    def __init__(self, condition_compound_dict) -> None:
        self.padding = 0
        self.condition_compound_dict = condition_compound_dict
        self.condition_compound_df_dict= {}
        for k in condition_compound_dict:
            new_data = {}
            data = condition_compound_dict[k]
            max_col = 0
            for idx in data:
                smi = data[idx]
                if smi == '':
                    smi = NONE_PANDDING
                new_data[smi] = smi.split('.')
                if max_col < len(new_data[smi]):
                    max_col = len(new_data[smi])
            for smi in new_data:
                new_data[smi] = new_data[smi] + [self.padding] * (max_col - len(new_data[smi]))
            self.condition_compound_df_dict[k] = pd.DataFrame(new_data)

        print('Read solvet {}, reagent {}, catalyst {}'.format(
            len(condition_compound_dict['s1']), 
            len(condition_compound_dict['r1']), 
            len(condition_compound_dict['c1'])))
    
    def apply(self, reag_ser):
        one_condition_dict = defaultdict(list)
        for name in ['c1', 'r1', 's1']:
            df = self.condition_compound_df_dict[name]
            for smi in df.columns.tolist():
                data_series = df[smi][df[smi]!=self.padding]
                if data_series.isin(reag_ser).sum() == len(data_series):
                    one_condition_dict[name].append(smi)

        return one_condition_dict

if __name__ == '__main__':
    debug = True

    condition_compound_fpath = '../check_data/condition_compound_add_name2smiles.pkl'
    if not debug:
        print('Reading USPTO-1k-tpl...')
        uspto_1k_tpl_train = pd.read_csv('../../dataset/source_dataset/uspto_1k_TPL_train_valid.tsv.gzip', compression='gzip', sep='\t', index_col=0)
        uspto_1k_tpl_test = pd.read_csv('../../dataset/source_dataset/uspto_1k_TPL_test.tsv.gzip', compression='gzip', sep='\t', index_col=0)
        uspto_1k_tpl = uspto_1k_tpl_train.append(uspto_1k_tpl_test)
        print('# data: {}'.format(len(uspto_1k_tpl)))
    else:
        print('Debug...')
        print('Reading debug tsv...')
        debug_df = pd.read_csv('../../dataset/source_dataset/debug_df.tsv', sep='\t')
        uspto_1k_tpl = debug_df
    

    uspto_1k_tpl['reagents_series'] = [covert_to_series(canonicalize_smiles(x), none_pandding=NONE_PANDDING) for x in tqdm(uspto_1k_tpl.reagents.tolist())]
    with open(condition_compound_fpath, 'rb') as f:
        condition_compound_dict = pickle.load(f)
    assignment_cls = AssignmentCondition(condition_compound_dict=condition_compound_dict)
    for reag_ser in tqdm(uspto_1k_tpl['reagents_series'].tolist()):
        assignment_cls.apply(reag_ser)
    pass

