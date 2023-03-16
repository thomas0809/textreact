import numpy as np
import time
import wandb

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
from sklearn.metrics import accuracy_score, top_k_accuracy_score, log_loss
from scipy.special import expit


class FNN(nn.Module):

    def __init__(self, in_feats, n_classes,
                 predict_hidden_feats = 512):
        
        super(FNN, self).__init__()

        self.predict = nn.Sequential(
            nn.Linear(in_feats, predict_hidden_feats), nn.PReLU(),
            nn.Linear(predict_hidden_feats, predict_hidden_feats), nn.PReLU(),
            nn.Linear(predict_hidden_feats, predict_hidden_feats), nn.PReLU(),
            nn.Linear(predict_hidden_feats, n_classes)
        )
                    
    def forward(self, inp):

        out = self.predict(inp)

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
        
               
    def training(self, train_loader, val_loader, max_epochs = 500, lr = 1e-3, patience = 20, weight_decay = 1e-10, val_every = 1, use_wandb = False, config = {}):
   
        loss_fn = nn.BCEWithLogitsLoss(reduction = 'none')
        optimizer = Adam(self.net.parameters(), lr = lr, weight_decay = weight_decay)
        lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=patience, min_lr=1e-6, verbose=True)
    
        train_size = train_loader.dataset.__len__()
        val_size = val_loader.dataset.__len__()
    
        val_y = val_loader.dataset.y
        best_macro_recall = np.mean([1/len(val_y[i]) for i in range(len(val_y))])
        best_micro_recall = len(val_y) / np.sum([len(a) for a in val_y])
        
        val_log = np.zeros(max_epochs) + np.inf

        if use_wandb:
            wandb.init(
                entity="textreact",
                project="baselines",
                name=f"rxnfp_{config.get('rtype')}",
                config=config
            )

        for epoch in range(max_epochs):
            
            # training
            self.net.train()
            start_time = time.time()
            grad_norm_list = []
            for batchidx, batchdata in enumerate(train_loader):
    
                inputs = batchdata[0].to(self.cuda)
                labels = batchdata[-1].to(self.cuda)
                preds = self.net(inputs)
    
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
                val_y_preds, _ = self.inference(val_loader)
                
                accuracy = np.mean([np.max([(c in val_y_preds[i]) for c in val_y[i]]) for i in range(len(val_y))])
                macro_recall = np.mean([np.mean([(c in val_y_preds[i]) for c in val_y[i]]) for i in range(len(val_y))])
                micro_recall = np.sum([np.sum([(c in val_y_preds[i]) for c in val_y[i]]) for i in range(len(val_y))]) / np.sum([len(a) for a in val_y])
                val_loss = 1 - (accuracy + macro_recall + micro_recall) / 3
        
                lr_scheduler.step(val_loss)
                val_log[epoch] = val_loss
        
                print('--- validation at epoch %d, total processed %d, ACC %.3f/1.000, macR %.3f/%.3f, micR %.3f/%.3f, monitor %d, time elapsed(min) %.2f'
                      %(epoch, val_size, accuracy, macro_recall, best_macro_recall, micro_recall, best_micro_recall, epoch - np.argmin(val_log[:epoch + 1]), (time.time()-start_time)/60))  
        
                if use_wandb:
                    wandb.log({
                        "train/loss": train_loss,
                        "train/grad_norm": grad_norm_list[-1].item(),
                        "test/loss": val_loss,  # 1 - (acc + macR + micR) / 3
                        "test/acc": accuracy,
                        "test/macR": macro_recall,
                        "test/micR": micro_recall
                    })

                # earlystopping
                if np.argmin(val_log[:epoch + 1]) == epoch:
                    torch.save(self.net.state_dict(), self.model_path) 
                
                elif np.argmin(val_log[:epoch + 1]) <= epoch - 50:
                    break
    
            elif use_wandb:
                wandb.log({
                    "train/loss": train_loss,
                    "train/grad_norm": grad_norm_list[-1].item()
                })
    
        print('training terminated at epoch %d' %epoch)
        self.load()
  
    
    def inference(self, tst_loader):
                 
        self.net.eval()    
        tst_y_scores = []
        with torch.no_grad():
            for batchidx, batchdata in enumerate(tst_loader):
            
                inputs = batchdata[0].to(self.cuda)
                preds_list = self.net(inputs).cpu().numpy()
                tst_y_scores.append(preds_list)
    
        tst_y_scores = expit(np.vstack(tst_y_scores))
        tst_y_preds = [[np.where(x > 0.5)[0].tolist()] for x in tst_y_scores]
    
        return tst_y_preds, tst_y_scores
