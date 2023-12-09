# TextReact

This repository contains the code for TextReact, a novel method that directly augments predictive chemistry with 
text retrieval.

![](assets/textreact.png)

```
@inproceedings{TextReact,
  author    = {Qian, Yujie and 
               Li, Zhening and 
               Tu, Zhengkai and 
               Coley, Connor W and 
               Barzilay, Regina},
  title     = {Predictive Chemistry Augmented with Text Retrieval},
  booktitle = {Proceedings of the 2023 Conference on Empirical Methods in Natural
                Language Processing, Singapore, December 6 - 10, 2023},
  publisher = {Association for Computational Linguistics},
  year      = {2018},
}
```

## Requirements
We implement the code with `torch==1.11.0`, `pytorch-lightning==2.0.0`, and `transformers==4.27.3`. 
To reproduce our experiments, we recommend creating a conda environment with the same dependencies:
```bash
conda env create -f environment.yml -n textreact
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```

## Data

Run the following commands to download and unzip the preprocessed datasets:
```
git clone https://huggingface.co/datasets/yujieq/TextReact data
cd data
unzip '*'
```

## Training Scripts

TextReact consists of two modules: SMILES-To-text retriever and 
text-augmented predictor. This repository only contains the code for 
training the predictor, while the code for the retriever is available in
a separate repository: https://github.com/thomas0809/tevatron.

The training scripts are located under [`scripts`](scripts):
* [`train_RCR.sh`](scripts/train_RCR.sh) trains a model for reaction condition recommendation (RCR)
on the random split of the USPTO dataset.
* [`train_RetroSyn_tf.sh`](scripts/train_RetroSyn_tf.sh) trains a template-free model for retrosynthesis
on the random split of the USPTO-50K dataset.
* [`train_RetroSyn_tb.sh`](scripts/train_RetroSyn_tb.sh) trains a template-based model for retrosynthesis
on the random split of the USPTO-50K dataset.
In addition, [`train_RCR_TS.sh`](scripts/train_RCR_TS.sh), [`train_RetroSyn_tf_TS.sh`](scripts/train_RetroSyn_tf_TS.sh)
and [`train_RetroSyn_tb_TS.sh`](scripts/train_RetroSyn_tb_TS.sh) train the corresponding models
on the time-based split of the dataset.

If you're working on a distributed file system, it is recommended to
add to the script a `--cache_path` option specifying a local path to reduce network time.

To run the script `scripts/train_MODEL.sh`, run the following command at the root of the folder:
```
bash scripts/train_MODEL.sh
```

At the end of training, two dictionaries are printed with the top-k test accuracies.
The first one corresponds to retrieving from the full corpus
and the second one corresponds to retrieving from the gold-removed corpus.

Models and test predictions are stored under the path specified by the `SAVE_PATH` variable in the script.
* `best.ckpt` is the checkpoint with the highest validation accuracy so far, whereas
* `last.ckpt` is the last checkpoint.
* `prediction_test_0.json` contains the test predictions when retrieving from the full corpus.
* `prediction_test_1.json` contains the predictions when retrieving from the gold-removed corpus.
