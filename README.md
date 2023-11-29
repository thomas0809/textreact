# TextReact

This is the repository for TextReact, a predictive chemistry model that leverages text retrieval
from USPTO patents to recommend reaction conditions for reactions
and suggest reactants for synthesizing target products.

![TextReact](assets/example.png)

```
@article{qian2022textreact,
  title={Predictive Chemistry Augmented with Text Retrieval},
  author={Qian, Yujie and Li, Zhening and Tu, Zhengkai and Coley, Connor W and Barzilay, Regina},
  journal={Empirical Methods in Natural Language Processing (EMNLP)},
  year={2023}
}
```

## Setup

Run the following command to download the repository and install the required dependencies:
```
git clone git@github.com:thomas0809/textreact.git
cd textreact
python -r requirements.txt
```
In addition, install PyTorch 1.12 following [these instructions](https://pytorch.org/get-started/previous-versions/)
as well as Faiss 1.7.4 following [these instructions](https://github.com/facebookresearch/faiss/blob/35dac924d132f97986df05a2e11905d945ba9a2c/INSTALL.md).

Run the following commands to download and unzip the preprocessed USPTO and USPTO-50K datasets:
```
git clone https://huggingface.co/datasets/yujieq/TextReact data
cd data
unzip '*'
```

## Training and evaluation

The training scripts are located under `scripts/exp/`.
The naming format of the scripts is `train_TASK_MODEL.sh` where
where:
- `TASK` specifies the task to train the model on:
`condition` for reaction condition recommendation (RCR) and `retro` retrosynthesis (RetroSyn)
- `MODEL` is the model to train and whether to use the random split or time split of the data:
`mlm` for TextReact on the random split, `transformer` for the Transformer baseline on the random split,
and `year` for TextReact on the time split.

If you're working on a distributed file system, it is recommended to change the `CACHE_PATH` variable
in the script to a local path to reduce network time.

To run the script `scripts/exp/train_TASK_MODEL.sh`, use the following command:
```
bash scripts/exp/train_TASK_MODEL.sh
```

At the end of training, a dictionary is printed with the top-k test accuracies.
For the TextReact model, two dictionaries are printed,
the first one corresponding to retrieving from the full corpus
and the second one corresponding to retrieving from the gold-removed corpus.

Models and test predictions are stored under the path specified by the `SAVE_PATH` variable in the script.
`best.ckpt` is the checkpoint with the highest validation accuracy so far, whereas
`last.ckpt` is the last checkpoint.
`prediction_test_0.json` contains predictions on the test set.
For the TextReact model, `prediction_test_0.json` contains the test predictions when retrieving from the full corpus,
whereas `prediction_test_1.json` contains the predictions when retrieving from the gold-removed corpus.
