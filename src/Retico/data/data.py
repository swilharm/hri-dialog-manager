import pickle
import random

NUM_PIECES = 15


class PentominoData:

    def __init__(self):
        with open('src/Retico/data/X_DM.pickle', 'rb') as X_file:
            self.dataset = pickle.load(X_file)
        self.index = random.randint(0, len(self.dataset))

    def get_sample(self):
        datapoint = self.dataset[self.index]
        self.index = random.randint(0, len(self.dataset))
        return datapoint


DATASET = PentominoData()
