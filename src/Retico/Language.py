import time

from retico_core.abstract import IncrementalUnit, AbstractModule, UpdateType, UpdateMessage
from retico_core.text import SpeechRecognitionIU
from data.dataset import DATASET, DATASET_INDEX, DATASET_INDEX_COUNTER
from src.ROS.run import main as language


class LanguageIU(IncrementalUnit):

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
        self.coordinates = (0.0, 0.0, 0.0)
        self.flag = 0.0

    def set_outputIUVariables(self, confidence_instruction, coordinates, confidence_pickup):
        ''' coordinate in this case is a vector indicating relative movement'''
        self.confidence_instruction = confidence_instruction
        self.coordinates = coordinates
        self.flag = confidence_pickup

    @staticmethod
    def type():
        return "Language IU"


class LanguageModule(AbstractModule):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.last_update = time.time()

    @staticmethod
    def name():
        return "Language Module"

    @staticmethod
    def description():
        return "Module that represents task 3"

    @staticmethod
    def input_ius():
        return [SpeechRecognitionIU]

    @staticmethod
    def output_iu():
        '''define output "type" '''
        return LanguageIU

    def process_update(self, update_message: UpdateMessage):
        asr_iu: SpeechRecognitionIU = next(update_message.incremental_units())
        if asr_iu.text:
            print(asr_iu.text)
            language_iu: LanguageIU = self.create_iu()
            vectors = language(asr_iu.predictions[0][0], True)
            vector = vectors[0]
            language_iu.payload = vector
            language_iu.confidence_instruction = vector[3]
            print(vector)
            if vector[0] == vector[1] == vector[2]:
                language_iu.flag = vector[0]
            else:
                language_iu.coordinates = (vector[0], vector[1], vector[2])
            return UpdateMessage.from_iu(language_iu, UpdateType.ADD)


        # if time.time() - self.last_update > 1:
        #     self.last_update = time.time()
        #     iu: LanguageIU = self.create_iu()
        #     iu.confidence_instruction = DATASET["l"][DATASET_INDEX][0]
        #     iu.coordinates = DATASET["l"][DATASET_INDEX][1]
        #     return UpdateMessage.from_iu(iu, UpdateType.ADD)
        # pass
