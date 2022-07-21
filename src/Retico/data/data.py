import json


class PentominoData:

    def __init__(self, path="src/Retico/data/test_data_examples.json"):
        self.index = 0

        with open(path) as file:
            data = json.loads(file.read())

        self.language_and_vision = []
        self.gesture = []
        self.language = []
        for i in data[0]:
            self.language_and_vision.append(i[0])
            self.gesture.append(i[1])
            self.language.append(i[2])

    def get_language_and_vision(self):
        return self.language_and_vision[self.index]

    def get_gesture(self):
        return self.gesture[self.index]

    def get_language(self):
        return self.language[self.index]


DATASET = PentominoData()
