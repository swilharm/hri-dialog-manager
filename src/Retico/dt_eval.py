import pickle
import numpy as np
from tqdm import tqdm


def load_data(x_path, y_path):

    with open(x_path, 'rb') as X_file:
        X_df = pickle.load(X_file)
    with open(y_path, 'rb') as y_file:
        y_df = pickle.load(y_file)

    return X_df, y_df

def dt(instr_confs, lv_coord_confs, g_coord_confs):
    instr_thresh = 0.8
    coord_thresh = 0.7
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
    X, y = load_data('data/X_DM.pickle', 'data/y_DM.pickle')
    preds = []
    for row in tqdm(X, total=len(X)):
        pred = dt(row[:3], row[3:3 + NUM_PIECES], row[3 + NUM_PIECES:])
        preds.append(pred)

    acc = np.mean(np.equal(np.array(preds), y.flatten()))
    print(f"Accuracy: {acc}")
