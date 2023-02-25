import numpy as np
import time

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
from sklearn.metrics import accuracy_score, top_k_accuracy_score, log_loss
from scipy.special import expit

import dgl
from dgl.nn.pytorch import NNConv, Set2Set


class MPNN(nn.Module):

    def __init__(self, node_in_feats, edge_in_feats, hidden_feats = 64,
                 num_step_message_passing = 3, num_step_set2set = 3, num_layer_set2set = 1,
                 readout_feats = 1024):
        
        super(MPNN, self).__init__()

        self.project_node_feats = nn.Sequential(
            nn.Linear(node_in_feats, hidden_feats), nn.ReLU()
        )
        
        self.num_step_message_passing = num_step_message_passing
        
        edge_network = nn.Linear(edge_in_feats, hidden_feats * hidden_feats)
        
        self.gnn_layer = NNConv(
            in_feats = hidden_feats,
            out_feats = hidden_feats,
            edge_func = edge_network,
            aggregator_type = 'sum'
        )
        
        self.activation = nn.ReLU()
        
        self.gru = nn.GRU(hidden_feats, hidden_feats)

        self.readout = Set2Set(input_dim = hidden_feats * 2,
                               n_iters = num_step_set2set,
                               n_layers = num_layer_set2set)

        self.sparsify = nn.Sequential(
            nn.Linear(hidden_feats * 4, readout_feats), nn.PReLU()
        )
             
    def forward(self, g):
            
        node_feats = g.ndata['node_attr']
        edge_feats = g.edata['edge_attr']
        
        node_feats = self.project_node_feats(node_feats)
        hidden_feats = node_feats.unsqueeze(0)

        node_aggr = [node_feats]        
        for _ in range(self.num_step_message_passing):
            node_feats = self.activation(self.gnn_layer(g, node_feats, edge_feats)).unsqueeze(0)
            node_feats, hidden_feats = self.gru(node_feats, hidden_feats)
            node_feats = node_feats.squeeze(0)
        
        node_aggr.append(node_feats)
        node_aggr = torch.cat(node_aggr, 1)
        
        readout = self.readout(g, node_aggr)
        graph_feats = self.sparsify(readout)
        
        return graph_feats


class reactionMPNN(nn.Module):

    def __init__(self, node_in_feats, edge_in_feats, n_classes,
                 readout_feats = 1024,
                 predict_hidden_feats = 512):
        
        super(reactionMPNN, self).__init__()

        self.mpnn = MPNN(node_in_feats, edge_in_feats)

        self.predict = nn.Sequential(
            nn.Linear(readout_feats * 2, predict_hidden_feats), nn.PReLU(),
            nn.Linear(predict_hidden_feats, predict_hidden_feats), nn.PReLU(),
            nn.Linear(predict_hidden_feats, predict_hidden_feats), nn.PReLU(),
            nn.Linear(predict_hidden_feats, n_classes)
        )
                    
    def forward(self, rmols, pmols):

        r_graph_feats = torch.sum(torch.stack([self.mpnn(mol) for mol in rmols]), 0)
        p_graph_feats = torch.sum(torch.stack([self.mpnn(mol) for mol in pmols]), 0)

        concat_feats = torch.cat([r_graph_feats, p_graph_feats], 1)
        out = self.predict(concat_feats)

        return out


class Trainer:

    def __init__(self, net, n_classes, rmol_max_cnt, pmol_max_cnt, batch_size, model_path, cuda):
    
        self.net = net.to(cuda)
        self.n_classes = n_classes
        self.rmol_max_cnt = rmol_max_cnt
        self.pmol_max_cnt = pmol_max_cnt
        self.batch_size = batch_size
        self.model_path = model_path
        self.cuda = cuda


    def load(self):
      
        self.net.load_state_dict(torch.load(self.model_path)) 
        
               
    def training(self, train_loader, val_loader, max_epochs = 500, lr = 1e-3, patience = 20, weight_decay = 1e-10, val_every = 1):
   
        loss_fn = nn.BCEWithLogitsLoss(reduction = 'none')
        optimizer = Adam(self.net.parameters(), lr = lr, weight_decay = weight_decay)
        lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=patience, min_lr=1e-6, verbose=True)
    
        train_size = train_loader.dataset.__len__()
        val_size = val_loader.dataset.__len__()
    
        val_y = val_loader.dataset.y
        best_macro_recall = np.mean([1/len(val_y[i]) for i in range(len(val_y))])
        best_micro_recall = len(val_y) / np.sum([len(a) for a in val_y])
        
        val_log = np.zeros(max_epochs) + np.inf
        for epoch in range(max_epochs):
            
            # training
            self.net.train()
            start_time = time.time()
            grad_norm_list = []
            for batchidx, batchdata in enumerate(train_loader):
    
                inputs_rmol = [b.to(self.cuda) for b in batchdata[:self.rmol_max_cnt]]
                inputs_pmol = [b.to(self.cuda) for b in batchdata[self.rmol_max_cnt:self.rmol_max_cnt+self.pmol_max_cnt]]
                
                labels = batchdata[-1].to(self.cuda)
                preds = self.net(inputs_rmol, inputs_pmol)
    
                loss = loss_fn(preds, labels).sum(axis = 1).mean()
    
                optimizer.zero_grad()
                loss.backward()
                assert not torch.isnan(loss)
                grad_norm = torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1e3)
                grad_norm_list.append(grad_norm.cpu().numpy())
                optimizer.step()
                
                train_loss = loss.detach().item()
    
            print('--- training epoch %d, lr %f, processed %d/%d, loss %.3f, time elapsed(min) %.2f'
                  %(epoch, optimizer.param_groups[-1]['lr'], train_size, train_size, train_loss, (time.time()-start_time)/60), np.max(grad_norm_list))
    
            # validation
            if epoch % val_every == 0:
                start_time = time.time()
                val_y_preds = self.inference(val_loader)
                
                accuracy = np.mean([np.max([(c in val_y_preds[i]) for c in val_y[i]]) for i in range(len(val_y))])
                macro_recall = np.mean([np.mean([(c in val_y_preds[i]) for c in val_y[i]]) for i in range(len(val_y))])
                micro_recall = np.sum([np.sum([(c in val_y_preds[i]) for c in val_y[i]]) for i in range(len(val_y))]) / np.sum([len(a) for a in val_y])
                val_loss = 1 - (accuracy + macro_recall + micro_recall) / 3
        
                lr_scheduler.step(val_loss)
                val_log[epoch] = val_loss
        
                print('--- validation at epoch %d, total processed %d, ACC %.3f/1.000, macR %.3f/%.3f, micR %.3f/%.3f, monitor %d, time elapsed(min) %.2f'
                          %(epoch, val_size, accuracy, macro_recall, best_macro_recall, micro_recall, best_micro_recall, epoch - np.argmin(val_log[:epoch + 1]), (time.time()-start_time)/60))  
    
                # earlystopping
                if np.argmin(val_log[:epoch + 1]) == epoch:
                    torch.save(self.net.state_dict(), self.model_path) 
                
                elif np.argmin(val_log[:epoch + 1]) <= epoch - 50:
                    break
    
        print('training terminated at epoch %d' %epoch)
        self.load()
  
    
    def inference(self, tst_loader):
                 
        self.net.eval()    
        tst_y_scores = []
        with torch.no_grad():
            for batchidx, batchdata in enumerate(tst_loader):
            
                inputs_rmol = [b.to(self.cuda) for b in batchdata[:self.rmol_max_cnt]]
                inputs_pmol = [b.to(self.cuda) for b in batchdata[self.rmol_max_cnt:self.rmol_max_cnt+self.pmol_max_cnt]]
                
                preds_list = self.net(inputs_rmol, inputs_pmol).cpu().numpy()
                tst_y_scores.append(preds_list)
    
        tst_y_scores = expit(np.vstack(tst_y_scores))
        tst_y_preds = [[np.where(x > 0.5)[0].tolist()] for x in tst_y_scores]
    
        return tst_y_preds
