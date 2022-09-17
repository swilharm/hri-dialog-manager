import time

from retico_core.abstract import IncrementalUnit, UpdateMessage, UpdateType, AbstractModule, AbstractProducingModule
from retico_core.text import SpeechRecognitionIU
from data.data import DATASET

NUM_PIECES = 15

class GestureIU(IncrementalUnit):
    def __init__(self, creator=None, iuid=0, previous_iu=None, grounded_in=None, payload=None, **kwargs
                 ):
        super().__init__(
            creator,
            iuid,
            previous_iu,
            grounded_in,
            payload,
            **kwargs
        )

        self.payload = payload
        self.confidence_instruction = 0.0
        self.coordinates = {i:0.0 for i in range(NUM_PIECES)}

    def set_confidence_and_coordinates(self, confidence_instruction, coordinates):
        self.confidence_instruction = confidence_instruction
        self.coordinates = coordinates

    @staticmethod
    def type():
        return "Gesture IU"


class GestureModule(AbstractProducingModule):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.last_update = time.time()

    @staticmethod
    def name():
        return "Gesture Module"

    @staticmethod
    def description():
        return "Module that represents task 4"

    @staticmethod
    def output_iu():
        return GestureIU

    def process_update(self, update_message):
        if time.time() - self.last_update > 1:
            self.last_update = time.time()
            iu = GestureIU()
            iu.grounded_in = iu
            datapoint = DATASET.get_sample()
            iu.confidence_instruction = datapoint[1]
            iu.coordinates = {i: j for i, j in enumerate(datapoint[3 + NUM_PIECES:])}
            return UpdateMessage.from_iu(iu, UpdateType.ADD)
        pass
