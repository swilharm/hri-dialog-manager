import threading

from retico_core.abstract import IncrementalUnit, UpdateMessage, UpdateType, AbstractTriggerModule

from data.data import DATASET

NUM_PIECES = 15


class GestureIU(IncrementalUnit):

    def __init__(self, creator=None, iuid=0, previous_iu=None, grounded_in=None, payload=None, processed=False
                 ):
        super().__init__(
            creator,
            iuid,
            previous_iu,
            grounded_in,
            payload,
        )
        self.processed = processed
        self.payload = payload
        self.confidence_instruction = 0.0
        self.coordinates = {i: 0.0 for i in range(NUM_PIECES)}

    @staticmethod
    def type():
        return "Gesture IU"


class GestureModule(AbstractTriggerModule):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loop = threading.Timer(1, self.trigger)

    @staticmethod
    def name():
        return "Gesture Module"

    @staticmethod
    def description():
        return "Module that represents task 4"

    @staticmethod
    def output_iu():
        return GestureIU

    def prepare_run(self):
        self.loop.start()

    def shutdown(self):
        self.loop.cancel()

    def trigger(self, **kwargs):
        iu = GestureIU()
        iu.grounded_in = iu
        datapoint = DATASET.get_sample()
        iu.confidence_instruction = datapoint[1]
        iu.coordinates = {i: j for i, j in enumerate(datapoint[3 + NUM_PIECES:])}
        self.append(UpdateMessage.from_iu(iu, UpdateType.ADD))
        self.loop = threading.Timer(1, self.trigger)
        self.loop.start()
