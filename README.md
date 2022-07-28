# reaction_condition_vae
Pytorch implementation of the model described in the paper [Generative Modeling to Predict Multiple Suitable Conditions for Chemical Reactions](#)

## Components
- **data/get_data.py** - script for preprocessing data
- **run_code.py** - script for model training/evaluation
- **dataset.py** - data structure & functions
- **model_VAE.py** - model architecture for the proposed method (ReactionVAE)
- **model_GNN.py** - model architecture for the baseline method (ReactionGNN)
- **model_rxnfp.py** - model architecture for the baseline method (ReactionFP)
- **util.py**

## Data
**Note**: Because the Reaxys database is commercially available, we do not have permission to release the datasets used in this paper to the public. Instead, we provide an example dataset (**data/data_example.npz**) so that anyone can test the code. It contains some chemical reaction records in [USPTO](https://figshare.com/articles/dataset/Chemical_reactions_from_US_patents_1976-Sep2016_/5104873) curated by [W. Beker et al. (2022)](https://pubs.acs.org/doi/10.1021/jacs.1c12005).

## Usage
`python run_code.py -t example -m proposed`

## Dependencies
- **Python**
- **Pytorch**
- **DGL**
- **RDKit**

## Citation


