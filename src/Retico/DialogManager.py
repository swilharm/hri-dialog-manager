from retico_core.abstract import IncrementalUnit, AbstractModule

# module's resulting IU
from Gesture import GestureIU
from Language import LanguageIU
from LanguageAndVision import LanguageAndVisionIU


class DialogManagerIU(IncrementalUnit):

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
        self.decision_coordinate = (0.0, 0.0, 0.0)
        self.confidence_decision = 0.0
        self.decision_pickup = 0.0

    def set_outputIUVariables(self, decision_coordinate, confidence_decision, decision_pickup):
        # confidence_pickup should always be zero -- should we drop it then?
        self.decision_coordinate = decision_coordinate
        self.confidence_decision = confidence_decision
        self.decision_pickup = decision_pickup

    @staticmethod
    def type():
        return "Dialog Manager IU"


class DialogManagerModule(AbstractModule):

    @staticmethod
    def name():
        return "Dialog Manager Module"

    @staticmethod
    def description():
        return "Module that represents task 1"

    @staticmethod
    def input_ius():
        return [GestureIU, LanguageAndVisionIU, LanguageIU]

    @staticmethod
    def output_iu():
        ''' define output "type" '''
        return [DialogManagerIU]

    def process_update(self, update_message):
        ''' define output values'''
        pass
