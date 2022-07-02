import json
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
        self.iu_type = None

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
        self.language_and_vision = {'confidence_instruction': 0.0, 'coordinates': dict(), 'time': time.time()}
        self.language = {'confidence_instruction': 0.0, 'coordinates': dict(), 'confidence_pickup': 0.0, 'time': time.time()}
        self.gesture = {'confidence_instruction': 0.0, 'coordinates': dict(), 'time': time.time()}

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
        return DialogManagerIU

    def process_update(self, update_message):
        ''' define output values'''
        input_iu = next(update_message.incremental_units())
        if input_iu.type() == "Language and Vision IU":
            self.language_and_vision['confidence_instruction'] = input_iu.confidence_instruction
            self.language_and_vision['coordinates'] = input_iu.coordinates
            self.language_and_vision['time'] = time.time()
        elif input_iu.type() == "Language IU":
            self.language['confidence_instruction'] = input_iu.confidence_instruction
            self.language['coordinates'] = input_iu.coordinates
            self.language['confidence_pickup'] = input_iu.confidence_pickup
            self.language['time'] = time.time()
        elif input_iu.type() == "Gesture IU":
            self.gesture['confidence_instruction'] = input_iu.confidence_instruction
            self.gesture['coordinates'] = input_iu.coordinates
            self.gesture['time'] = time.time()
        else:
            print("What is happening, I am frightened:", input_iu.type())

        if time.time() - self.language_and_vision['time'] > DECAY:
            self.language_and_vision['confidence_instruction'] = 0.0
            self.language_and_vision['coordinates'] = dict()
        if time.time() - self.language['time'] > DECAY:
            self.language['confidence_instruction'] = 0.0
            self.language['coordinates'] = dict()
            self.language['confidence_pickup'] = 0.0
        if time.time() - self.gesture['time'] > DECAY:
            self.gesture['confidence_instruction'] = 0.0
            self.gesture['coordinates'] = dict()

        instr_thresh = 0.90
        coord_thresh = 0.80

        output_iu = self.create_iu()
        if self.language['confidence_instruction'] > instr_thresh:
            output_iu.decision_coordinate = self.language['coordinates']
            output_iu.confidence_decision = self.language['confidence_instruction']
            output_iu.decision_pickup = 3
            output_iu.iu_type = "LanguageIU"
        else:
            if self.language_and_vision['confidence_instruction'] > instr_thresh > self.gesture['confidence_instruction']:
                coordinate = max(self.language_and_vision['coordinates'], key=self.language_and_vision['coordinates'].get)
                probability = self.language_and_vision['coordinates'][coordinate]
                if probability > coord_thresh:
                    output_iu.decision_coordinate = coordinate
                    output_iu.confidence_decision = probability
                    output_iu.decision_pickup = 1
                    output_iu.iu_type = "LanguageAndVisionIU"
                else:
                    output_iu.decision_coordinate = (0.0, 0.0, 0.0)
                    output_iu.confidence_decision = 0
                    output_iu.decision_pickup = 2
                    output_iu.iu_type = None
                    print("UNCERTAINTY {•̃_•̃}")
            elif self.language_and_vision['confidence_instruction'] < instr_thresh < self.gesture['confidence_instruction']:
                print(self.gesture['coordinates'])
                coordinate = max(self.gesture['coordinates'], key=self.gesture['coordinates'].get)
                probability = self.gesture['coordinates'][coordinate]
                if probability > coord_thresh:
                    output_iu.decision_coordinate = coordinate
                    output_iu.confidence_decision = probability
                    output_iu.decision_pickup = 1
                    output_iu.iu_type = "GestureIU"
                else:
                    output_iu.decision_coordinate = (0.0, 0.0, 0.0)
                    output_iu.confidence_decision = 0
                    output_iu.decision_pickup = 2
                    output_iu.iu_type = None
                    print("UNCERTAINTY {•̃_•̃}")
            elif self.language_and_vision['confidence_instruction'] > instr_thresh and self.gesture['confidence_instruction'] > instr_thresh:
                mean = dict()
                for key in self.language_and_vision['coordinates']:
                    mean[key] = (self.language_and_vision['coordinates'][key]+self.gesture['coordinates'][key])/2
                max_mean = max(mean, key=mean.get)
                if mean[max_mean] > coord_thresh:
                    output_iu.decision_coordinate = max(mean, key=mean.get)
                    output_iu.confidence_decision = mean[output_iu.decision_coordinate]
                    output_iu.decision_pickup = 4
                    output_iu.iu_type = "LanguageAndVisionIU <3 GestureIU"
                else:
                    output_iu.decision_coordinate = (0.0, 0.0, 0.0)
                    output_iu.confidence_decision = 0
                    output_iu.decision_pickup = 2
                    output_iu.iu_type = None
                    print("UNCERTAINTY {•̃_•̃}")
            elif self.language_and_vision['confidence_instruction'] < instr_thresh and self.gesture['confidence_instruction'] < instr_thresh:
                print("UNCERTAINTY {•̃_•̃}")
                return

        print(output_iu, output_iu.iu_type)
        print(output_iu.confidence_decision, output_iu.decision_coordinate, output_iu.decision_pickup)

