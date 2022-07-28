import torch
import dgl

                        
def collate_reaction_graphs(batch):

    batchdata = list(map(list, zip(*batch)))
    gs = [dgl.batch(s) for s in batchdata[:-1]]
    labels = torch.FloatTensor(batchdata[-1])
    
    return *gs, labels


def MC_dropout(model):

    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()
    
    pass
