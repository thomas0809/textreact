# Reaxys-TotalSyn-Condition Curation

The original data of this data set comes from about 17,000 total synthetic literatures. Use the DOI number to extract the corresponding reaction data from Reaxys, and select data items with a chemical yield greater than 30% to download in batches. See `Parrot/preprocess_script/reaxys_script/qurey_file` for batch query scripts. Combine all extracted data and select 'Reaction Details: Reaction Classification'=='Preparation'. Save to `Parrot/dataset/source_dataset/reaxys_total_syn_merge_Preparation_clean_up_mapped.csv` The csv header is as follows:
```
'Reaction ID',
'Record Type',
'Reactant',
'Product',
'Reaction',
'Reaction Details: Reaction Classification',
'Time (Reaction Details) [h]',
'Temperature (Reaction Details) [C]',
'Pressure (Reaction Details) [Torr]',
'pH-Value (Reaction Details)',
'Other Conditions',
'Reaction Type',
'Product.1',
'Yield',
'Yield (numerical)',
'Yield (optical)',
'Reagent',
'Catalyst',
'Solvent (Reaction Details)',
'References',
'canonical_rxn_smiles'                # canonical reaction smiles  (value != na)
```

Then:
```
cd Parrot/preprocess_script/reaxys_script
python 1.0.get_reaxys_total_syn_condition.py
python 2.0.convert_context_tokens.py
```
Done!