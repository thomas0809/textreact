# USPTO-Condition Curation
```
cd Parrot/preprocess_script/uspto_script
mkdir ../../dataset/source_dataset/uspto_org_xml/
```
Download the original USPTO reaction dataset from [here](https://figshare.com/articles/dataset/Chemical_reactions_from_US_patents_1976-Sep2016_/5104873). Put `1976_Sep2016_USPTOgrants_cml.7z` and `2001_Sep2016_USPTOapplications_cml.7z` under `../../dataset/source_dataset/uspto_org_xml/`.<br>


The running environment of rxnmapper is required when generating USPTO-Condition. This environment is not compatible with parrot_env. Please re-create a virtual environment of rxnmapper. See [rxnmapper github repository](https://github.com/rxn4chemistry/rxnmapper) for details.<br>


Then:
```
cd ../../dataset/source_dataset/uspto_org_xml/
7z x 1976_Sep2016_USPTOgrants_cml.7z
7z x 2001_Sep2016_USPTOapplications_cml.7z
cd Parrot/preprocess_script/uspto_script
python 1.get_condition_from_uspto.py
sh 2.0.clean_up_rxn_condition.sh 
python 2.1.merge_clean_up_rxn_conditon.py
python 3.0.split_condition_and_slect.py
python 4.0.split_train_val_test.py
python 5.0.convert_context_tokens.py
```
Done!
