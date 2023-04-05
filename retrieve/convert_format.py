import json

input_path = '/data/scratch/zhengkai/neural-retrieval/retrieved/USPTO_condition_MIT_smiles/test.jsonl'
output_path = 'contrastive/test.json'

output = []
with open(input_path) as f:
    for line in f:
        data = json.loads(line)
        output.append({
            'id': data['query_id'],
            'nn': [p['docid'] for p in data['negative_passages']]
        })

with open(output_path, 'w') as f:
    json.dump(output, f)
