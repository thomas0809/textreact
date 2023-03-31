import numpy as np
from .dataset import CONDITION_COLS


def evaluate_reaction_condition(prediction, data_df):
    cnt = {x: 0 for x in [1, 3, 5, 10, 15]}
    for i, output in prediction.items():
        label = data_df.loc[i, CONDITION_COLS].tolist()
        hit_map = [pred == label for pred in output['prediction']]
        for x in cnt:
            cnt[x] += np.any(hit_map[:x])
    num_example = len(data_df)
    accuracy = {x: cnt[x] / num_example for x in cnt}
    return accuracy
