from retico_core.abstract import IncrementalUnit, AbstractModule, UpdateType, UpdateMessage
from retico_core.text import SpeechRecognitionIU


# module's resulting IU
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
        self.confidence_pickup = 0.0

    def set_outputIUVariables(self, confidence_instruction, coordinates, confidence_pickup):
        ''' coordinate in this case is a vector indicating relative movement'''
        self.confidence_instruction = confidence_instruction
        self.coordinates = coordinates
        self.confidence_pickup = confidence_pickup

    @staticmethod
    def type():
        return "Language IU"


class LanguageModule(AbstractModule):

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
        '''define output values'''
        #ut: SpeechRecognitionIU = next(update_message.update_types())
        #iu: SpeechRecognitionIU = next(update_message.incremental_units())
        #print(ut, iu.payload)
        return UpdateMessage.from_iu(self.create_iu(), UpdateType.ADD)
