import random
import threading

from retico_core.abstract import IncrementalUnit, AbstractModule, UpdateType, UpdateMessage, AbstractTriggerModule
from retico_core.text import SpeechRecognitionIU

from data.data import DATASET


class LanguageIU(IncrementalUnit):

    def __init__(
            self, creator=None, iuid=0, previous_iu=None, grounded_in=None, payload=None, processed=False
    ):
        super().__init__(
            creator,
            iuid=iuid,
            previous_iu=previous_iu,
            grounded_in=grounded_in,
            payload=payload,
        )
        self.processed = processed
        self.payload = payload
        self.confidence_instruction = 0.0
        self.coordinates = (0.0, 0.0, 0.0)
        self.flag = 0.0

    @staticmethod
    def type():
        return "Language IU"


class LanguageModule(AbstractTriggerModule):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loop = threading.Timer(1, self.trigger)

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
        return LanguageIU

    def prepare_run(self):
        self.loop.start()

    def shutdown(self):
        self.loop.cancel()

    def process_update(self, update_message: UpdateMessage):
        # INTEGRATION
        asr_iu: SpeechRecognitionIU = next(update_message.incremental_units())
        if asr_iu.predictions[0][0]:
            language_iu: LanguageIU = self.create_iu(grounded_in=asr_iu)
            # TODO Integrate Language team
            return UpdateMessage.from_iu(language_iu, UpdateType.ADD)

    def trigger(self, **kwargs):
        # DATASET
        iu:LanguageIU = self.create_iu()
        iu.grounded_in = iu
        datapoint = DATASET.get_sample()
        iu.confidence_instruction = datapoint[2]
        iu.coordinates = (random.randrange(-2, 2), random.randrange(-2, 2), random.randrange(-2, 2))
        iu.flag = 0
        self.append(UpdateMessage.from_iu(iu, UpdateType.ADD))
        self.loop = threading.Timer(1, self.trigger)
        self.loop.start()
