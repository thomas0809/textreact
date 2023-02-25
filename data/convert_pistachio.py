import argparse
import numpy as np
import json
import os
import time
from collections import defaultdict
import rdkit.Chem as Chem
from rdkit import RDLogger

parser = argparse.ArgumentParser()
parser.add_argument("--rtype", required=True)
parser.add_argument("--split", choices=["trn", "tst"], required=True)
parser.add_argument("--years", type=int, action="append")
parser.add_argument("--clist")

args = parser.parse_args()

RDLogger.DisableLog('rdApp.*')

data_dirs = [f"../pistachio/extract/grants/{i}" for i in args.years]  # years 2000 to 2021
start = time.time()

# data = defaultdict(list)
# condition_dict = {}
# num_canon_err = 0
# i = 0
# for data_dir in data_dirs:
#     print("Loading:", data_dir)
#     for file_name in os.listdir(data_dir):
#         path = os.path.join(data_dir, file_name)
#         if os.path.isfile(path) and file_name[0] != '.':
#             with open(path, 'r') as f:
#                 data_lines = f.readlines()
#                 for line in data_lines:
#                     item = json.loads(line)
#                     reaction = item["data"]["smiles"]
#                     i = reaction.index('>') + 1  # start of conditions
#                     j = reaction[i:].index('>') + i  # second >
#                     k = reaction.index(' ') if ' ' in reaction else len(reaction)
#                     reaction_smiles = reaction[:i] + reaction[j:k]
#                     conditions = []
#                     for component in item["components"]:
#                         if component["role"] in ["Solvent", "Agent", "Catalyst"] and "smiles" in component:
#                             try:
#                                 canon_smiles = Chem.CanonSmiles(component["smiles"])
#                             except:
#                                 canon_smiles = component["smiles"]
#                                 # print("BAD COND:", component["smiles"])
#                                 num_canon_err += 1
#                             if canon_smiles not in condition_dict:
#                                 condition_dict[canon_smiles] = len(condition_dict)
#                             conditions.append(condition_dict[canon_smiles])
#                     if conditions not in data[reaction_smiles]:
#                         data[reaction_smiles].append(conditions)
#         # break
#         i += 1
# print("NUM FILES:", i)
# print("NUM REACTIONS:", len(data))
# print("NUM CANON ERR:", num_canon_err)
# print("NUM CONDITIONS:", len(condition_dict))
# print("TIME:", time.time() - start)
# 
# condition_list = [None for _ in range(len(condition_dict))]
# for k, v in condition_dict.items():
#     condition_list[v] = k
# 
# np.savez("data_example.npz", data=np.array([dict(data), condition_list], dtype='O'))

i = 0
num_err = 0
reaction_data = defaultdict(list)
for data_dir in data_dirs:
    print("Loading:", data_dir)
    for file_name in os.listdir(data_dir):
        path = os.path.join(data_dir, file_name)
        if os.path.isfile(path) and file_name[0] != '.':
            with open(path, 'r') as f:
                data_lines = f.readlines()
                for line in data_lines:
                    reaction = json.loads(line)
                    reactants = []
                    products = []
                    agents = []
                    for component in reaction["components"]:
                        try:
                            smiles = Chem.CanonSmiles(component["smiles"])
                        except:
                            num_err += 1
                            continue
                        multiple = '.' in smiles
                        if component["role"] == "Product":
                            if multiple:
                                products.extend(smiles.split('.'))
                            else:
                                products.append(smiles)
                        elif component["role"] == "Reactant":
                            if multiple:
                                reactants.extend(smiles.split('.'))
                            else:
                                reactants.append(smiles)
                        else:
                            if multiple:
                                mult_smiles = smiles.split('.')
                                agents.extend(mult_smiles)
                                # agent_type_dist[component["role"]] += len(mult_smiles)
                            else:
                                agents.append(smiles)
                                # agent_type_dist[component["role"]] += 1
                    if len(reactants) in [1, 2] and len(products) == 1 and len(agents) > 0:
                        reaction_data[(frozenset(reactants), products[0])].append(agents)
        i += 1
        print(i, len(reaction_data))
if args.clist:
    condition_list = np.load(args.clist, allow_pickle=True)['data']
    dictionary = {agent: index for index, agent in enumerate(condition_list)}
else:
    agent_freqs = defaultdict(int)
    for conds in reaction_data.values():
        for agents in conds:
            for agent in agents:
                agent_freqs[agent] += 1
                agent_freq_list = sorted(agent_freqs.items(), key=lambda x: x[1], reverse=True)
                total_freq = sum(freq for agent, freq in agent_freq_list)
    cum_freq = 0
    cum_agent_freq_list = []
    for agent, freq in agent_freq_list:
        cum_freq += freq
        norm_freq = cum_freq/total_freq
        cum_agent_freq_list.append((agent, norm_freq))
        if norm_freq >= 0.95:
            break
    dictionary = {agent: j for j, (agent, freq) in enumerate(cum_agent_freq_list)}
    condition_list = [None for _ in range(len(dictionary))]
    for k, v in dictionary.items():
        condition_list[v] = k
    np.savez(f"clist_{args.rtype}.npz", data=np.array(condition_list, dtype='O'))
print(dictionary)
new_reaction_data = []
for (reactants, product), conditions in reaction_data.items():
    new_conditions = []
    for agents in conditions:
        new_agents = list(filter(lambda agent: agent in dictionary, agents))
        if new_agents:
            new_conditions.append(new_agents)
    if new_conditions:
        new_reaction_data.append((reactants, product, new_conditions))
data = {}
for reactants, product, conditions in new_reaction_data:
    reaction_smiles = '.'.join(reactants) + '>>' + product
    data[reaction_smiles] = [[dictionary[agent] for agent in agents] for agents in conditions]
print("NUM FILES:", i)
print("NUM REACTIONS:", len(data))
print("NUM ERR:", num_err)
print("NUM AGENTS:", len(dictionary))
print("TIME:", time.time() - start)

np.savez(f"data_{args.rtype}_{args.split}.npz", data=dict(data), dtype='O')
