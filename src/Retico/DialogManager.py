import time

from retico_core.abstract import IncrementalUnit, AbstractModule

# module's resulting IU
from Gesture import GestureIU
from Language import LanguageIU
from LanguageAndVision import LanguageAndVisionIU

DECAY = 3

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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.language_and_vision = {'iu': None, 'time': time.time()}
        self.language = {'iu': None, 'time': time.time()}
        self.gesture = {'iu': None, 'time': time.time()}

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
        iu = next(update_message.incremental_units())
        if iu.type() == "Language and Vision IU":
            self.language_and_vision['iu'] = iu
            self.language_and_vision['time'] = time.time()
        elif iu.type() == "Language IU":
            self.language['iu'] = iu
            self.language['time'] = time.time()
        elif iu.type() == "Gesture IU":
            self.gesture['iu'] = iu
            self.gesture['time'] = time.time()
        else:
            print("What is happening, I am frightened:", iu.type())

        if time.time() - self.language_and_vision['time'] > DECAY:
            self.language_and_vision['iu'] = None
        if time.time() - self.language['time'] > DECAY:
            self.language['iu'] = None
        if time.time() - self.gesture['time'] > DECAY:
            self.gesture['iu'] = None

        rule_thresh = 0.95
        coord_thresh = 0.95

#        if self.language_and_vision['iu'] is None or self.language['iu'] is None or self.gesture['iu'] is None:
#            pass

        if self.language['iu'] and self.language['iu'].confidence_instruction > rule_thresh:
            print(self.language['iu'].coordinates)
        else:
            if self.language_and_vision['iu'] is not None and self.gesture['iu'] is not None:
                if self.language_and_vision['iu'].confidence_instruction > rule_thresh > self.gesture['iu'].confidence_instruction:
                    print(self.language_and_vision['iu'].coordinates)
                    # TODO coordinate confidence, maybe even uncertainty?
                if self.language_and_vision['iu'].confidence_instruction < rule_thresh < self.gesture['iu'].confidence_instruction:
                    print(self.gesture['iu'].coordinates)
                    # TODO coordinate confidence, maybe even uncertainty?
                if self.language_and_vision['iu'].confidence_instruction > rule_thresh and self.gesture['iu'].confidence_instruction > rule_thresh:
                    print("OOGA BOOGA BOOGA")
                    # TODO become sentient, learn actual language, code this
                if self.language_and_vision['iu'].confidence_instruction < rule_thresh and self.gesture['iu'].confidence_instruction < rule_thresh:
                    print("I don't know what to do UwU")
                    # TODO move out our mom's basement, meet a cute girl/boy/whatever you prefer

