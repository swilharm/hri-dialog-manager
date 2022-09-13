import json
import pandas as pd
import numpy as np
from tqdm import tqdm


def load_data(x_path, y_path, no_puzzlepieces):
    with open(x_path) as f:
        data = json.load(f)

    X = pd.DataFrame.from_dict(data, orient='index')

    with open(y_path) as f:
        data = json.load(f)

    y = pd.DataFrame(pd.DataFrame.from_dict(data).values.T)

    df_new = X.drop(['Instruction', 'Coordinate'], axis=1)

    instr_df = pd.DataFrame(df_new['Instruction Confidence'].to_list())
    instr_df.head()

    coord_df = pd.DataFrame(df_new['Coordinate Confidence'].to_list())

    coord_df0 = pd.DataFrame(coord_df[0].to_list())
    coord_df1 = pd.DataFrame(coord_df[1].to_list())

    a = [f'Instr{i}' for i in range(3)]
    b = [f'LV{i}' for i in range(no_puzzlepieces)]
    c = [f'G{i}' for i in range(no_puzzlepieces)]

    col_names = []
    col_names.extend(a)
    col_names.extend(b)
    col_names.extend(c)

    final_df_noisy = pd.concat([instr_df, coord_df0, coord_df1], axis=1)
    final_df_noisy.columns = col_names

    return final_df_noisy, y


def dt(instr_confs, lv_coord_confs, g_coord_confs):
    instr_thresh = 0.8
    coord_thresh = 0.1
    if instr_confs[2] > instr_thresh:
        return 1
    else:
        if instr_confs[0] > instr_thresh > instr_confs[1]:
            coordinate = np.argmax(lv_coord_confs)
            probability = lv_coord_confs[coordinate]
            if probability > coord_thresh:
                return coordinate + 2
            else:
                return 0
        elif instr_confs[0] < instr_thresh < instr_confs[1]:
            coordinate = np.argmax(g_coord_confs)
            probability = g_coord_confs[coordinate]
            if probability > coord_thresh:
                return coordinate + 2
            else:
                return 0
        elif instr_confs[0] > instr_thresh and instr_confs[1] > instr_thresh:
            mean = [(lv + g) / 2 for lv, g in zip(lv_coord_confs, g_coord_confs)]
            max_mean = np.argmax(mean)
            if mean[max_mean] > coord_thresh:
                return max_mean + 2
            else:
                return 0
        elif instr_confs[0] < instr_thresh and instr_confs[1] < instr_thresh:
            return 0
    return None


if __name__ == '__main__':
    NUM_PIECES = 15
    X, y = load_data('data/X_DM.json', 'data/y_DM.json', NUM_PIECES)
    X, y = X[:], y[:]
    preds = []
    for row in tqdm(X.values, total=len(X)):
        pred = dt(row[:3], row[3:3 + NUM_PIECES], row[3 + NUM_PIECES:])
        preds.append(pred)

    acc = sum([i == j for i, j in zip(preds, y.values)]) / len(y)
    print(acc)
