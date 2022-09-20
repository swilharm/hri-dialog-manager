import threading

from retico_core.abstract import IncrementalUnit, UpdateMessage, UpdateType, AbstractTriggerModule
from retico_core.text import SpeechRecognitionIU

from data.data import DATASET

NUM_PIECES = 15


class LanguageAndVisionIU(IncrementalUnit):

    def __init__(self, creator=None, iuid=0, previous_iu=None, grounded_in=None, payload=None, processed=False
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
        self.coordinates = {i: 0.0 for i in range(NUM_PIECES)}

    @staticmethod
    def type():
        return "Language and Vision IU"


class LanguageAndVisionModule(AbstractTriggerModule):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loop = threading.Timer(0.05, self.trigger)

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

    def prepare_run(self):
        self.loop.start()

    def shutdown(self):
        self.loop.cancel()

    def process_update(self, update_message: UpdateMessage):
        asr_iu: SpeechRecognitionIU = next(update_message.incremental_units())
        if asr_iu.predictions[0][0]:
            language_and_vision_iu: LanguageAndVisionIU = self.create_iu(grounded_in=asr_iu)
            # TODO Integrate Language & Vision team
            return UpdateMessage.from_iu(language_and_vision_iu, UpdateType.ADD)

    def trigger(self, **kwargs):
        iu = LanguageAndVisionIU()
        iu.grounded_in = iu
        datapoint = DATASET.get_sample()
        iu.confidence_instruction = datapoint[0]
        iu.coordinates = {i: j for i, j in enumerate(datapoint[3:3 + NUM_PIECES])}
        self.append(UpdateMessage.from_iu(iu, UpdateType.ADD))
        self.loop = threading.Timer(1, self.trigger)
        self.loop.start()
