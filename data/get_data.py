import os, csv, time
import argparse
from collections import defaultdict
import numpy as np
import pickle as pkl
from rdkit import Chem, RDConfig, rdBase, RDLogger
from rdkit.Chem import AllChem, ChemicalFeatures
RDLogger.DisableLog('rdApp.*') 

import torch
from dgl import graph

import warnings
warnings.filterwarnings("ignore")


def get_graph_data(data, keys, filename):

    chem_feature_factory = ChemicalFeatures.BuildFeatureFactory(os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef'))

    # atom_dict = defaultdict(int)
    # charge_dict = defaultdict(int)
    # degree_dict = defaultdict(int)
    # hybridization_dict = defaultdict(int)
    # hydrogen_dict = defaultdict(int)
    # valence_dict = defaultdict(int)
    # ringsize_dict = defaultdict(int)
    # bond_dict = defaultdict(int)

    def mol_to_graph(mol):

        def _DA(mol):
            D_list, A_list = [], []
            for feat in chem_feature_factory.GetFeaturesForMol(mol):
                if feat.GetFamily() == 'Donor': D_list.append(feat.GetAtomIds()[0])
                if feat.GetFamily() == 'Acceptor': A_list.append(feat.GetAtomIds()[0])
            
            return D_list, A_list

        def _chirality(atom):
            return [(atom.GetProp('Chirality') == 'Tet_CW'), (atom.GetProp('Chirality') == 'Tet_CCW')] if atom.HasProp('Chirality') else [0, 0]
            
        def _stereochemistry(bond):
            return [(bond.GetProp('Stereochemistry') == 'Bond_Cis'), (bond.GetProp('Stereochemistry') == 'Bond_Trans')] if bond.HasProp('Stereochemistry') else [0, 0]    

        n_node = mol.GetNumAtoms()
        n_edge = mol.GetNumBonds() * 2
        
        D_list, A_list = _DA(mol)  
        # atom_list = ['Ag','Al','As','B','Bi','Br','C','Cl','Co','Cr','Cu','F','Ge','H','I','In','K','Li','Mg','Mo','N','Na','O','P','Pd','S','Sb','Se','Si','Sn','Te','Zn']
        # charge_list = [-1, 0, 1, 2, 0]
        # degree_list = [1, 2, 3, 4, 5, 6, 0]
        # hybridization_list = ['SP','SP2','SP3','SP3D','SP3D2','S','UNSPECIFIED']
        # hydrogen_list = [1, 2, 3, 0]
        # valence_list = [1, 2, 3, 4, 5, 6, 12, 0]
        # ringsize_list = [3, 4, 5, 6, 7, 8]
        # bond_list = ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC']
        # for a in mol.GetAtoms():
            # atom_dict[a.GetSymbol()] += 1
            # charge_dict[a.GetFormalCharge()] += 1
            # degree_dict[a.GetDegree()] += 1
            # hybridization_dict[str(a.GetHybridization())] += 1
            # hydrogen_dict[a.GetTotalNumHs(includeNeighbors = True)] += 1
            # valence_dict[a.GetTotalValence()] += 1
        # for ring in mol.GetRingInfo().AtomRings():
            # ringsize_dict[len(ring)] += 1
        # for b in mol.GetBonds():
            # bond_dict[str(b.GetBondType())] += 1
        atom_fea1 = np.eye(len(atom_list), dtype = bool)[[atom_list.index(a.GetSymbol()) for a in mol.GetAtoms()]]
        atom_fea2 = np.eye(len(charge_list), dtype = bool)[[charge_list.index(a.GetFormalCharge()) for a in mol.GetAtoms()]][:,:-1]
        atom_fea3 = np.eye(len(degree_list), dtype = bool)[[degree_list.index(a.GetDegree()) for a in mol.GetAtoms()]][:,:-1]
        atom_fea4 = np.eye(len(hybridization_list), dtype = bool)[[hybridization_list.index(str(a.GetHybridization())) for a in mol.GetAtoms()]][:,:-2]###
        atom_fea5 = np.eye(len(hydrogen_list), dtype = bool)[[hydrogen_list.index(a.GetTotalNumHs(includeNeighbors = True)) for a in mol.GetAtoms()]][:,:-1]
        atom_fea6 = np.eye(len(valence_list), dtype = bool)[[valence_list.index(a.GetTotalValence()) for a in mol.GetAtoms()]][:,:-2]###
        atom_fea7 = np.array([[(j in D_list), (j in A_list)] for j in range(mol.GetNumAtoms())], dtype = bool)
        atom_fea8 = np.array([[a.GetIsAromatic(), a.IsInRing()] for a in mol.GetAtoms()], dtype = bool)
        atom_fea9 = np.array([[a.IsInRingSize(s) for s in ringsize_list] for a in mol.GetAtoms()], dtype = bool)
        atom_fea10 = np.array([_chirality(a) for a in mol.GetAtoms()], dtype = bool)
        
        node_attr = np.hstack([atom_fea1, atom_fea2, atom_fea3, atom_fea4, atom_fea5, atom_fea6, atom_fea7, atom_fea8, atom_fea9, atom_fea10])

        if n_edge > 0:
            bond_fea1 = np.eye(len(bond_list), dtype = bool)[[bond_list.index(str(b.GetBondType())) for b in mol.GetBonds()]]
            bond_fea2 = np.array([[b.IsInRing(), b.GetIsConjugated()] for b in mol.GetBonds()], dtype = bool)
            bond_fea3 = np.array([_stereochemistry(b) for b in mol.GetBonds()], dtype = bool)
            
            edge_attr = np.hstack([bond_fea1, bond_fea2, bond_fea3])
            edge_attr = np.vstack([edge_attr, edge_attr])
            
            bond_loc = np.array([[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in mol.GetBonds()], dtype=int)
            src = np.hstack([bond_loc[:,0], bond_loc[:,1]])
            dst = np.hstack([bond_loc[:,1], bond_loc[:,0]])

        else:
            edge_attr = np.empty((0, edge_dim)).astype(bool)
            src = np.empty(0).astype(int)
            dst = np.empty(0).astype(int)

        g = graph((src, dst), num_nodes = n_node)
        g.ndata['node_attr'] = torch.from_numpy(node_attr).bool()
        g.edata['edge_attr'] = torch.from_numpy(edge_attr).bool()

        return g

    def dummy_graph():

        g = graph(([], []), num_nodes = 1)
        g.ndata['node_attr'] = torch.from_numpy(np.empty((1, node_dim))).bool()
        g.edata['edge_attr'] = torch.from_numpy(np.empty((0, edge_dim))).bool()
        
        return g
    
    # atom_list = ['Ag','Al','As','B','Bi','Br','C','Cl','Co','Cr','Cu','F','Ge','H','I','In','K','Li','Mg','Mo','N','Na','O','P','Pd','S','Sb','Se','Si','Sn','Te','Zn']
    # atom_list = ['Ag', 'Al', 'As', 'Au', 'B', 'Bi', 'Br', 'C', 'Ca', 'Cl', 'Co', 'Cr', 'Cs', 'Cu', 'F', 'Fe', 'Ge', 'H', 'Hf', 'Hg', 'I', 'In', 'Ir', 'K', 'La', 'Li', 'Mg', 'Mn', 'Mo', 'N', 'Na', 'Ni', 'O', 'Os', 'P', 'Pb', 'Pd', 'Ru', 'S', 'Sb', 'Se', 'Si', 'Sm', 'Sn', 'Te', 'Ti', 'U', 'V', 'Y', 'Zn', 'Zr']
    atom_list = ['Ag', 'Al', 'As', 'At', 'Au', 'B', 'Be', 'Bi', 'Br', 'C', 'Ca', 'Cl', 'Co', 'Cr', 'Cs', 'Cu', 'F', 'Fe', 'Ge', 'H', 'Hf', 'Hg', 'I', 'In', 'Ir', 'K', 'La', 'Li', 'Mg', 'Mn', 'Mo', 'N', 'Na', 'Ni', 'O', 'Os', 'P', 'Pb', 'Pd', 'Ru', 'S', 'Sb', 'Se', 'Si', 'Sm', 'Sn', 'Tb', 'Te', 'Ti', 'U', 'V', 'W', 'Y', 'Zn', 'Zr']
    # charge_list = [-1, 0, 1, 2, 0]
    charge_list = [-1, 0, 1, 2, 3]
    degree_list = [1, 2, 3, 4, 5, 6, 0]
    hybridization_list = ['SP','SP2','SP3','SP3D','SP3D2','S','UNSPECIFIED']
    hydrogen_list = [1, 2, 3, 0]
    # valence_list = [1, 2, 3, 4, 5, 6, 12, 0]
    valence_list = [1, 2, 3, 4, 5, 6, 7, 8, 12, 0]
    ringsize_list = [3, 4, 5, 6, 7, 8]
    bond_list = ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC']
    
    node_dim = len(atom_list) + len(charge_list) + len(degree_list) + len(hybridization_list) + len(hydrogen_list) + len(valence_list) + len(ringsize_list) - 1
    edge_dim = len(bond_list) + 4
    
    rmol_max_cnt = 2
    pmol_max_cnt = 1

    rmol_graphs = [[] for _ in range(rmol_max_cnt)]
    pmol_graphs = [[] for _ in range(pmol_max_cnt)]
    reaction_dict = {'y': [], 'rsmi': []}     
    
    print('--- generating graph data for %s' %filename)
    print('--- n_reactions: %d, reactant_max_cnt: %d, product_max_cnt: %d' %(len(keys), rmol_max_cnt, pmol_max_cnt)) 

    start_time = time.time()
    num_err = 0
    errors = []
    for i, rsmi in enumerate(keys):
    
        [reactants_smi, products_smi] = rsmi.split('>>')
        ys = data[rsmi]

        # processing reactants
        try:
            rmols = [None for _ in range(rmol_max_cnt)]
            reactants_smi_list = reactants_smi.split('.')
            for _ in range(rmol_max_cnt - len(reactants_smi_list)): reactants_smi_list.append('')
            if len(reactants_smi_list) > 2:
                breakpoint()
            for j, smi in enumerate(reactants_smi_list):
                if smi == '':
                    rmols[j] = dummy_graph()
                else:
                    rmol = Chem.MolFromSmiles(smi)
                    rs = Chem.FindPotentialStereo(rmol)
                    for element in rs:
                        if str(element.type) == 'Atom_Tetrahedral' and str(element.specified) == 'Specified': rmol.GetAtomWithIdx(element.centeredOn).SetProp('Chirality', str(element.descriptor))
                        elif str(element.type) == 'Bond_Double' and str(element.specified) == 'Specified': rmol.GetBondWithIdx(element.centeredOn).SetProp('Stereochemistry', str(element.descriptor))

                    rmols[j] = mol_to_graph(Chem.RemoveHs(rmol))
                    
            # processing products
            pmols = [None for _ in range(pmol_max_cnt)]
            products_smi_list = products_smi.split('.')
            for _ in range(pmol_max_cnt - len(products_smi_list)): products_smi_list.append('')
            for j, smi in enumerate(products_smi_list):
                if smi == '':
                    pmols[j] = dummy_graph()
                else: 
                    pmol = Chem.MolFromSmiles(smi)
                    ps = Chem.FindPotentialStereo(pmol)
                    for element in ps:
                        if str(element.type) == 'Atom_Tetrahedral' and str(element.specified) == 'Specified': pmol.GetAtomWithIdx(element.centeredOn).SetProp('Chirality', str(element.descriptor))
                        elif str(element.type) == 'Bond_Double' and str(element.specified) == 'Specified': pmol.GetBondWithIdx(element.centeredOn).SetProp('Stereochemistry', str(element.descriptor))
                            
                    pmols[j] = mol_to_graph(Chem.RemoveHs(pmol))
            
            for j, rmol in enumerate(rmols):
                rmol_graphs[j].append(rmol)
            for j, pmol in enumerate(pmols):
                pmol_graphs[j].append(pmol)
        
            reaction_dict['y'].append(ys)
            reaction_dict['rsmi'].append(rsmi)

        except ValueError as e:
            num_err += 1
            errors.append(e)
    
        # monitoring
        if (i+1) % 10000 == 0:
            time_elapsed = (time.time() - start_time)/60
            print('--- %d/%d processed, %.2f min elapsed' %(i+1, len(keys), time_elapsed)) 
            print('--- %d/%d (%.3f) errors' % (num_err, len(keys), num_err / len(keys)))
            print('--- errors:')
            print(errors)
            # print(atom_dict)
            # print(charge_dict)
            # print(degree_dict)
            # print(hybridization_dict)
            # print(hydrogen_dict)
            # print(valence_dict)
            # print(ringsize_dict)
            # print(bond_dict)

    time_elapsed = (time.time() - start_time)/60
    print('--- %d/%d processed, %.2f min elapsed' %(i+1, len(keys), time_elapsed)) 
    print('--- %d/%d (%.3f) errors' % (num_err, len(keys), num_err / len(keys)))
    print('--- errors:')
    print(errors)
    # print(atom_dict)
    # print(charge_dict)
    # print(degree_dict)
    # print(hybridization_dict)
    # print(hydrogen_dict)
    # print(valence_dict)
    # print(ringsize_dict)
    # print(bond_dict)

    rmol_graphs = list(map(list, zip(*rmol_graphs)))
    pmol_graphs = list(map(list, zip(*pmol_graphs))) 
    
    # save file
    with open(filename, 'wb') as f:
        pkl.dump([rmol_graphs, pmol_graphs, reaction_dict['y'], reaction_dict['rsmi']], f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frac", type=float, default=1.0)
    parser.add_argument("--rtype", required=True)
    parser.add_argument("--split", choices=["trn", "tst"], required=True)
    parser.add_argument("--clist", required=True)

    args = parser.parse_args()

    file_path = './data_%s_%s.npz' % (args.rtype, args.split)

    reaction_dict = np.load(file_path, allow_pickle = True)['data'].item()
    clist = np.load(args.clist, allow_pickle = True)['data']

    np.random.seed(134)
    reaction_keys = np.array(list(reaction_dict.keys()))
    if args.frac != 1.:
        reaction_keys = reaction_keys[np.random.permutation(len(reaction_keys))]
        split_amnt = int(len(reaction_keys) * args.frac)
        reaction_keys = reaction_keys[:split_amnt]

    print(file_path)
    print(f"NUM REACTIONS: {len(reaction_keys)}")
    print(f"NUM CONDITION MOLS: {len(clist)}")
    
    filename = './data_dgl_%s_%s.pkl' % (args.rtype, args.split)
    get_graph_data(reaction_dict, reaction_keys, filename)
