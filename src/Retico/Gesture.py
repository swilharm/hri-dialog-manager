import time

from retico_core.abstract import IncrementalUnit, UpdateMessage, UpdateType, AbstractModule, AbstractProducingModule
from retico_core.text import SpeechRecognitionIU
from data.data import DATASET


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
        self.coordinates = dict()

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
        if time.time() - self.last_update > 5:
            self.last_update = time.time()
            iu:GestureIU = self.create_iu()
            gesture_input = DATASET.get_gesture()
            iu.confidence_instruction = gesture_input[0]
            iu.coordinates = gesture_input[1]
            return UpdateMessage.from_iu(iu, UpdateType.ADD)
        pass
