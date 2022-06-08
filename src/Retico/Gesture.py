from retico_core.abstract import IncrementalUnit, AbstractProducingModule


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
        return "Gesture"


class GestureModule(AbstractProducingModule):

    @staticmethod
    def name():
        return "Gesture Module"

    @staticmethod
    def description():
        return "Module that represents task 4"

    @staticmethod
    def output_iu():
        return [GestureIU]

    def process_update(self, update_message):
        pass
