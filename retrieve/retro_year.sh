#python retrieve_faiss.py \
#    --data_path ../data/USPTO_50K_year \
#    --train_file ../USPTO_rxn_smiles.csv \
#    --valid_file valid.csv \
#    --test_file test.csv \
#    --field product_smiles \
#    --output_path output/USPTO_50K_year/full

python retrieve_faiss.py \
    --data_path ../data/USPTO_50K_year \
    --train_file  ../USPTO_rxn_smiles.csv \
    --before 2012 \
    --valid_file valid.csv \
    --test_file test.csv \
    --field product_smiles \
    --output_path output/USPTO_50K_year/corpus_before_2012

