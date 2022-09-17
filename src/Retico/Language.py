import random
import time

from retico_core.abstract import IncrementalUnit, AbstractModule, UpdateType, UpdateMessage, AbstractProducingModule
from retico_core.text import SpeechRecognitionIU
from retico_googleasr import GoogleASRModule

from data.data import DATASET
from src.ROS.run import main as language


class LanguageIU(IncrementalUnit):

    def __init__(
            self, creator=None, iuid=0, previous_iu=None, grounded_in=None, payload=None
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

    @staticmethod
    def type():
        return "Language IU"


class LanguageModule(AbstractProducingModule):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.last_update = time.time()

    @staticmethod
    def name():
        return "Language Module"

    @staticmethod
    def description():
        return "Module that represents task 3"

    # @staticmethod
    # def input_ius():
    #     return [SpeechRecognitionIU]

    @staticmethod
    def output_iu():
        '''define output "type" '''
        return LanguageIU

    def process_update(self, update_message: UpdateMessage):
        # asr_iu: SpeechRecognitionIU = next(update_message.incremental_units())
        # if asr_iu.text:
        #     language_iu: LanguageIU = self.create_iu(grounded_in=asr_iu)
        #     vectors = language(asr_iu.predictions[0][0], True)
        #     vector = vectors[0]
        #     language_iu.payload = vector
        #     language_iu.confidence_instruction = vector[3]
        #     if vector[0] == vector[1] == vector[2]:
        #         language_iu.flag = vector[0]
        #         if language_iu.flag == 0:
        #             language_iu.confidence_instruction = 0
        #     else:
        #         language_iu.coordinates = (vector[0], vector[1], vector[2])
        #     return UpdateMessage.from_iu(language_iu, UpdateType.ADD)

        if time.time() - self.last_update > 1:
            self.last_update = time.time()
            iu = LanguageIU()
            iu.grounded_in = iu
            datapoint = DATASET.get_sample()
            iu.confidence_instruction = datapoint[2]
            iu.coordinates = (random.randrange(-2, 2), random.randrange(-2, 2), random.randrange(-2, 2))
            iu.flag = 0
            return UpdateMessage.from_iu(iu, UpdateType.ADD)
        pass
