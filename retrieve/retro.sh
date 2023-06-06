#python retrieve_faiss.py \
#    --data_path ../data/USPTO_50K/matched \
#    --train_file train.csv \
#    --valid_file valid.csv \
#    --test_file test.csv \
#    --field product_smiles \
#    --output_path output/USPTO_50K

python retrieve_faiss.py \
    --data_path ../data/USPTO_50K/matched1 \
    --train_file ../../USPTO_rxn_smiles.csv \
    --valid_file valid.csv \
    --test_file test.csv \
    --field product_smiles \
    --output_path output/USPTO_50K/full
