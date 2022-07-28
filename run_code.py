import numpy as np
import sys, csv, os
import pickle as pkl
import torch
from torch.utils.data import DataLoader
from dgl.data.utils import split_dataset, Subset
from dataset import GraphDataset
from util import collate_reaction_graphs

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--rtype', '-t', type=str, choices=['example', 'suzuki', 'cn', 'negishi', 'pkr'])
parser.add_argument('--method', '-m', type=str, choices=['rxnfp', 'baseline', 'proposed'], default='proposed')
parser.add_argument('--iterid', '-i', type=int, default=0)
parser.add_argument('--mode', '-o', type=str, choices=['trn', 'tst'], default='trn')
args = parser.parse_args()

rtype = args.rtype
method = args.method
iterid = args.iterid
mode = args.mode

if method == 'baseline':
    from model_GNN import reactionMPNN as Model
    from model_GNN import Trainer
    use_rxnfp = False
    collate_fn = collate_reaction_graphs
elif method == 'proposed':
    from model_VAE import VAE as Model
    from model_VAE import Trainer
    use_rxnfp = False
    collate_fn = collate_reaction_graphs
elif method == 'rxnfp':
    from model_rxnfp import FNN as Model
    from model_rxnfp import Trainer
    use_rxnfp = True
    collate_fn = None

random_state = 134 + iterid
batch_size = 128
cuda = torch.device('cuda:0')

if not os.path.exists('./model/'): os.makedirs('./model/')
model_path = './model/model_%s_%s_%d.pt' %(method, rtype, iterid)

tstdata = GraphDataset(rtype, split = 'tst', use_rxnfp = use_rxnfp)
tst_loader = DataLoader(dataset=tstdata, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

n_classes = tstdata.n_classes
rmol_max_cnt = tstdata.rmol_max_cnt
pmol_max_cnt = tstdata.pmol_max_cnt


# training 
if method == 'rxnfp': net = Model(tstdata.fp_dim, n_classes)
else: net = Model(tstdata.node_dim, tstdata.edge_dim, n_classes)
trainer = Trainer(net, n_classes, rmol_max_cnt, pmol_max_cnt, batch_size, model_path, cuda)

print('-- TRAINING')
if mode == 'trn':
    trndata = GraphDataset(rtype, split = 'trn', use_rxnfp = use_rxnfp, seed = random_state)
    valdata = GraphDataset(rtype, split = 'val', use_rxnfp = use_rxnfp, seed = random_state)
    trn_loader = DataLoader(dataset=trndata, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(dataset=valdata, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    print('-- CONFIGURATIONS')
    print('--- reaction type:', rtype)
    print('--- no. classes:', n_classes)
    print('--- trn/val/tst: %d/%d/%d' %(trndata.n_reactions, valdata.n_reactions, tstdata.n_reactions))
    print('--- max no. reactants/products: %d/%d'%(tstdata.rmol_max_cnt, tstdata.pmol_max_cnt))
    print('--- model_path:', model_path)
    
    len_list = trndata.cnt_list
    print('--- (trn) total no. conditions:', trndata.n_conditions)
    print('--- (trn) no. conditions per reaction (min/avg/max): %d/%.2f/%d'%(np.min(len_list), np.mean(len_list), np.max(len_list)))

    trainer.training(trn_loader, val_loader)
    
elif mode == 'tst':
    trainer.load()


# inference
tst_y = tstdata.y

print('-- EVALUATION')
print(model_path)
  
if method in ['rxnfp', 'baseline']:
    tst_y_preds = trainer.inference(tst_loader)
    T = 1
    accuracy = np.mean([np.max([(c in tst_y_preds[i][:T]) for c in tst_y[i]]) for i in range(len(tst_y))])
    macro_recall = np.mean([np.mean([(c in tst_y_preds[i][:T]) for c in tst_y[i]]) for i in range(len(tst_y))])
    micro_recall = np.sum([np.sum([(c in tst_y_preds[i][:T]) for c in tst_y[i]]) for i in range(len(tst_y))]) / np.sum([len(a) for a in tst_y])

    print('--- T=%d accuracy/macro-recall/micro-recall:'%T, accuracy, macro_recall, micro_recall) 

elif method == 'proposed':
    tst_y_preds = trainer.inference(tst_loader, n_sampling = 100)
    for T in [1, 10, 30, 100]:
        accuracy = np.mean([np.max([(c in tst_y_preds[i][:T]) for c in tst_y[i]]) for i in range(len(tst_y))])
        macro_recall = np.mean([np.mean([(c in tst_y_preds[i][:T]) for c in tst_y[i]]) for i in range(len(tst_y))])
        micro_recall = np.sum([np.sum([(c in tst_y_preds[i][:T]) for c in tst_y[i]]) for i in range(len(tst_y))]) / np.sum([len(a) for a in tst_y])
    
        print('--- T=%d accuracy/macro-recall/micro-recall:'%T, accuracy, macro_recall, micro_recall)