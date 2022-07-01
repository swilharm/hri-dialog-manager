import time

from retico_core.abstract import IncrementalUnit, UpdateMessage, UpdateType, AbstractModule
from retico_core.text import SpeechRecognitionIU


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


class GestureModule(AbstractModule):

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
    def input_ius():
        return [SpeechRecognitionIU]

    @staticmethod
    def output_iu():
        return GestureIU

    def process_update(self, update_message):
        if time.time() - self.last_update > 1:
            self.last_update = time.time()
            return UpdateMessage.from_iu(self.create_iu(), UpdateType.ADD)
        pass
