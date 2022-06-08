from retico_core.abstract import IncrementalUnit, AbstractModule
from retico_core.text import SpeechRecognitionIU


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

class LanguageAndVisionModule(AbstractModule):

    @staticmethod
    def name():
        return "Language and Vision Module"

    @staticmethod
    def description():
        return "Module that represents task 2"

    @staticmethod
    def input_ius():
        return [SpeechRecognitionIU]

    @staticmethod
    def output_iu():
        return [LanguageAndVisionIU]

    def process_update(self, update_message):
        pass