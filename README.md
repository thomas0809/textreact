# TextReact

## Data
Soft link the data
```
ln -s /Mounts/rbg-storage1/users/yujieq/textreact/data data
```

## Example script
```
bash scripts/train_retro_text_nn.sh
```
- Change `--cache_path` to a local path (not on NFS) to reduce network time.
- Change `--nn_path` to the path to the retrieval results. NNs for train/valid/test should be in json files. Example:
  `/Mounts/rbg-storage1/users/yujieq/tevatron/output/rxntext_retro_rn_b512_ep400`
