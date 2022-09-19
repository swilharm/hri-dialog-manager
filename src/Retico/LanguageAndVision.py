import threading

from retico_core.abstract import IncrementalUnit, UpdateMessage, UpdateType, AbstractTriggerModule

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

    def trigger(self, **kwargs):
        iu = LanguageAndVisionIU()
        iu.grounded_in = iu
        datapoint = DATASET.get_sample()
        iu.confidence_instruction = datapoint[0]
        iu.coordinates = {i: j for i, j in enumerate(datapoint[3:3 + NUM_PIECES])}
        self.right_buffers()[-1].put(UpdateMessage.from_iu(iu, UpdateType.ADD))
        threading.Timer(1, self.trigger).start()
