import torch
import dgl
import numpy as np
import time

                        
def collate_reaction_graphs(batch):

    start = time.time()
    batchdata = list(map(list, zip(*batch)))
    # print("collate time 1:", time.time() - start)
    start = time.time()
    gs = [dgl.batch(s) for s in batchdata[:-1]]
    # print("collate time 2:", time.time() - start)
    start = time.time()
    labels = torch.FloatTensor(np.array(batchdata[-1]))
    # print("collate time 3:", time.time() - start)
    
    return *gs, labels


def MC_dropout(model):

    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()
    
    pass
