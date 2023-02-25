# reaction_condition_vae
Pytorch implementation of the model described in the paper [Generative Modeling to Predict Multiple Suitable Conditions for Chemical Reactions](#). Here, we use the model on the Pistachio data set and aim to augment the model with text retrieval.

## Setup

Everything is implemented in Python 3.9 with PyTorch and DGL v0.9.1. The chemistry toolkit RDKit is used for processing molecules. Please check their corresponding websites for how to install them. In addition, you should install `numpy`, `scikit-learn`, and `wandb`.

<!---
To install all dependencies, run
```
pip install -r requirements.txt
```

Alternatively, to create a Conda environment with the dependencies installed, run
```
conda create -n [insert name of environment] --file requirements.txt
```
--->

We're using the Pistachio data set. Construct a soft link to the data set like this:
```
ln -s path/to/pistachio/data pistachio
```

## Components
- **pistachio** - soft link to the `data` directory of the Pistachio data
- **data/convert_pistachio.py** - script for converting Pistachio data into format that **data/get_data.py** takes
- **data/get_data.py** - script for preprocessing data
- **run_code.py** - script for model training/evaluation
- **dataset.py** - data structure & functions
- **model_VAE.py** - model architecture for the proposed method (ReactionVAE)
- **model_GNN.py** - model architecture for the baseline method (ReactionGNN)
- **model_rxnfp.py** - model architecture for the baseline method (ReactionFP)
- **util.py**

## Example usage

In the **data** folder, preprocess the data as follows:
```
python convert_pistachio.py --rtype pistachio --split trn -y 2019 -y 2020
python convert_pistachio.py --rtype pistachio --split tst -y 2021 --clist clist_pistachio.npz
python get_data.py --rtype pistachio --split trn --clist clist_pistachio.npz
python get_data.py --rtype pistachio --split tst --clist clist_pistachio.npz
```
The first two commands convert Pistachio data into train and test sets and generates a list of condition molecules.
The last two commands preprocesses the data.

In the root, train the model as follows:
```
python run_code.py -t pistachio_more -m baseline --val_every 5 --lr 0.0002 --bs 256
```
Replace `baseline` with `proposed` for VAE and with `rxnfp` for reaction fingerprint model. Other possible arguments
are `--patience` for the learning rate scheduler, and `--weight_decay`.
