
Adapted from https://github.com/rmrmg/SuzukiConditions/tree/master/uspto/data_extraction

Hwere are files and scripts to extract Suzuki coupling reaction from USPTO data


- parseUSPTOdata.py  - this file extract Suzuki reaction from USPTO by default it looking for two files
    `1976_Sep2016_USPTOgrants_smiles.rsmi` `2001_Sep2016_USPTOapplications_smiles.rsmi` and products six files:
    with following suffix: 'raw_homo.txt' 'raw_hete.txt' 'parsed_homo.txt' 'parsed_het.txt', 'clear_homo.txt' 'clear_het.txt'
    where:
    - raw - means raw reaction as in USPTO (including not smiles with NOT recognized role)
    - clear - reaction without compounds of unknown role, only substrates, products and reaction conditions, i.e. catalyst(s), ligand(s), base(s) and solvent(s)
    - het - reactions where at least one coupling partner is heteroaromatic
    - parsed - text representation of python dict with parsed information about reaction this can be used as an input to `makeReactionFromParsed.py`
    - homo - other Suzuki reactions 
  The scripts use correction files:
    - basescanon.smi  - list of bases. Format: 1st column smiles from USPTO, 2nd column is comment, 3rd column (optional) is corrected smiles - smiles from first column will be replace to this
    - solventscanon.smi - list of solvents, format as above
    - liganscanon.smi  - list of 'rare' ligand(s) format as above. Please note logic for ligand detection is hardcoded in script if you want NOT treat as ligand some common ligand see getSuzukiRoles function in the code.
    
    - uspto_exclude_list.smi  - list of reaction which will be excluded from further consideration
    - uspto_replace_list.csv - smiles from first column will be replaced to smiles from second column
   
- makeReactionFromParsed.py  convert *parsed* file into input for AI, Run the script with --help option for detail

