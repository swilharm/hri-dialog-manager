import time

from retico_core.abstract import IncrementalUnit, AbstractModule, UpdateMessage, UpdateType, AbstractProducingModule
from retico_core.text import SpeechRecognitionIU

from data.data import DATASET


class LanguageAndVisionIU(IncrementalUnit):

    def __init__(
            self, creator, iuid=0, previous_iu=None, grounded_in=None, payload=None
    ):
        super().__init__(
            creator,
            iuid=iuid,
            previous_iu=previous_iu,
            grounded_in=grounded_in,
            payload=payload,
        )
        self.payload = payload
        self.confidence_instruction = 0.0
        self.coordinates = dict()

    def set_confidence_and_coordinates(self, confidence_instruction, coordinates):
        self.confidence_instruction = confidence_instruction
        self.coordinates = coordinates

    @staticmethod
    def type():
        return "Language and Vision IU"


class LanguageAndVisionModule(AbstractProducingModule):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.last_update = time.time()

    @staticmethod
    def name():
        return "Language and Vision Module"

    @staticmethod
    def description():
        return "Module that represents task 2"

    # @staticmethod
    # def input_ius():
    #    return [SpeechRecognitionIU]

    @staticmethod
    def output_iu():
        return LanguageAndVisionIU

    def process_update(self, update_message):
        if time.time() - self.last_update > 5:
            self.last_update = time.time()
            iu: LanguageAndVisionIU = self.create_iu()
            language_and_vision_input = DATASET.get_gesture()
            iu.confidence_instruction = language_and_vision_input[0]
            iu.coordinates = language_and_vision_input[1]
            return UpdateMessage.from_iu(iu, UpdateType.ADD)
        pass
