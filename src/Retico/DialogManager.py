import time
import enum
import numpy as np
import torch
from stable_baselines3 import PPO

from retico_core.abstract import IncrementalUnit, AbstractModule, UpdateMessage, UpdateType, IncrementalQueue
from Gesture import GestureIU, ID2COORD
from Language import LanguageIU
from LanguageAndVision import LanguageAndVisionIU
from dl import Net
from Periodic import PeriodicIU

NUM_PIECES = 15
ARTIFICIAL_DELAY = 0

class Flag(enum.Enum):
    UNCERTAINTY = 0
    ABSOLUTE_MOVEMENT = 1
    RELATIVE_MOVEMENT = 2
    STOP = 3
    GRAB = 4
    RELEASE = 5
    RESET = 6
    REVERSE = 7


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
        self.flag = -1

    @staticmethod
    def type():
        return "Dialog Manager IU"


class DialogManagerModule(AbstractModule):

    def __init__(self, model="DT", **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.times = list()

    @staticmethod
    def name():
        return "Dialog Manager Module"

    @staticmethod
    def description():
        return "Module that represents task 1"

    @staticmethod
    def input_ius():
        return [LanguageAndVisionIU, LanguageIU, GestureIU, PeriodicIU]

    @staticmethod
    def output_iu():
        return DialogManagerIU

    def setup(self):
        self.lv = LanguageAndVisionIU(grounded_in=LanguageAndVisionIU(), processed=True)
        self.l = LanguageIU(grounded_in=LanguageIU(), processed=True)
        self.g = GestureIU(grounded_in=GestureIU(), processed=True)
        # Load the chosen model
        if self.model == "DL":
            self.setup_deep_learning()
        elif self.model == "RL":
            self.setup_reinforcement_learning()

    def process_update(self, update_message):
        ''' define output values'''
        output_ius = []
        for input_iu, update_type in zip(update_message.incremental_units(), update_message.update_types()):
            input_iu: IncrementalUnit
            # Revokes reset the memory to the previous IU
            if update_type == UpdateType.REVOKE:
                input_iu = input_iu.previous_iu
                input_iu.processed = False

            # Store IU in memory
            if input_iu.type() == "Language and Vision IU":
                self.lv = input_iu
            elif input_iu.type() == "Language IU":
                self.l = input_iu
            elif input_iu.type() == "Gesture IU":
                self.g = input_iu

            # Figure out which IU to  mainly process
            oldest_iu_time = min(self.lv.grounded_in.created_at if not self.lv.processed else np.inf,
                                 self.l.grounded_in.created_at if not self.l.processed else np.inf,
                                 self.g.grounded_in.created_at if not self.g.processed else np.inf)

            # Only process if main IU has at least a certain age
            if oldest_iu_time != np.inf and time.time() - oldest_iu_time > ARTIFICIAL_DELAY:
                # Find IUs that happened at the same time, use empty IU if none found
                lv = self.lv
                while lv.previous_iu and lv.grounded_in.created_at - oldest_iu_time > 0:
                    lv = lv.previous_iu
                if oldest_iu_time - lv.grounded_in.created_at > ARTIFICIAL_DELAY:
                    lv = LanguageAndVisionIU()

                l = self.l
                while l.previous_iu and l.grounded_in.created_at - oldest_iu_time > 0:
                    l = l.previous_iu
                if oldest_iu_time - l.grounded_in.created_at > ARTIFICIAL_DELAY:
                    l = LanguageIU()

                g = self.g
                while g.previous_iu and g.grounded_in.created_at - oldest_iu_time > 0:
                    g = g.previous_iu
                if oldest_iu_time - g.grounded_in.created_at > ARTIFICIAL_DELAY:
                    g = GestureIU()

                # Apply chosen model to the IUs
                if self.model == "DL":
                    decision = self.deep_learning(lv, l, g)
                elif self.model == "RL":
                    decision = self.reinforcement_learning(lv, l, g)
                else:
                    decision = self.decision_tree(lv, l, g)

                # Mark IUs as processed, so they can't be main IU anymore
                lv.processed = True
                l.processed = True
                g.processed = True

                # print("Decision:", decision)

                # Translate model output into IU
                output_iu: DialogManagerIU = self.create_iu()
                if decision == -1:
                    output_iu.flag = Flag.UNCERTAINTY.value
                elif decision == 0:
                    pass
                elif decision == 1:
                    output_iu.confidence_decision = 1
                    output_iu.decision_coordinate = l.coordinates
                    output_iu.flag = l.flag + 2
                else:
                    output_iu.confidence_decision = 1
                    output_iu.decision_coordinate = ID2COORD[decision - 2]
                    output_iu.flag = 1

                # If revoked and different from previous, revoke and add, else do nothing
                if update_type == UpdateType.REVOKE:
                    previous_iu: DialogManagerIU = output_iu.previous_iu
                    if previous_iu.confidence_decision != output_iu.confidence_decision or \
                            previous_iu.decision_coordinate != output_iu.decision_coordinate or \
                            previous_iu.flag != output_iu.flag:
                        output_ius.append((UpdateType.REVOKE, previous_iu))
                        output_ius.append((UpdateType.ADD, output_iu))
                else:
                    output_ius.append((UpdateType.ADD, output_iu))
                return UpdateMessage.from_iu_list(self, output_ius)

    def decision_tree(self, lv: LanguageAndVisionIU, l: LanguageIU, g: GestureIU) -> int:
        """Apply rule based decision tree model to the input"""
        instr_thresh = 0.8
        coord_thresh = 0.7
        if l.confidence_instruction > instr_thresh:
            return 1
        else:
            if lv.confidence_instruction > instr_thresh > g.confidence_instruction:
                coordinate = max(lv.coordinates, key=lv.coordinates.get)
                probability = lv.coordinates[coordinate]
                if probability > coord_thresh:
                    return coordinate + 2
                else:
                    return -1
            elif lv.confidence_instruction < instr_thresh < g.confidence_instruction:
                coordinate = max(g.coordinates, key=g.coordinates.get)
                probability = g.coordinates[coordinate]
                if probability > coord_thresh:
                    return coordinate + 2
                else:
                    return -1
            elif lv.confidence_instruction > instr_thresh and g.confidence_instruction > instr_thresh:
                mean = [(lv + g) / 2 for lv, g in zip(lv.coordinates.values(), g.coordinates.values())]
                max_mean = mean.index(max(mean))
                if mean[max_mean] > coord_thresh:
                    return max_mean + 2
                else:
                    return -1
            elif lv.confidence_instruction < instr_thresh and g.confidence_instruction < instr_thresh:
                return 0

    def build_vector(self, lv: LanguageAndVisionIU, l: LanguageIU, g: GestureIU):
        """Helper function to turn IU input into numpy array"""
        array = np.empty((33,))
        array[0] = lv.confidence_instruction
        array[1] = g.confidence_instruction
        array[2] = l.confidence_instruction
        array[3:3 + NUM_PIECES] = list(lv.coordinates.values())
        array[3 + NUM_PIECES:] = list(g.coordinates.values())
        return array

    def setup_deep_learning(self):
        """Loads the trained DL model"""
        print("Loading DL model 1")
        self.net_task_1 = Net(input_size=33, hidden_size=32, output_size=17)
        self.net_task_1.load_state_dict(torch.load("src/Retico/DL_action_model.pt"))
        self.net_task_1.eval()
        print("Loading DL model 2")
        self.net_task_2 = Net(input_size=33, hidden_size=32, output_size=2)
        self.net_task_2.load_state_dict(torch.load("src/Retico/DL_uncertainty_model.pt"))
        self.net_task_2.eval()

    def setup_reinforcement_learning(self):
        """
        Loads the trained RL model
        """
        print("Loading RL model 1")
        self.rl_task_1 = PPO.load('src/Retico/RL_action_model', None)
        print("Loading RL model 2")
        self.rl_task_2 = PPO.load('src/Retico/RL_uncertainty_model', None)

    def deep_learning(self, lv: LanguageAndVisionIU, l: LanguageIU, g: GestureIU) -> int:
        """Apply DL model to input"""
        tensor = torch.FloatTensor(self.build_vector(lv, l, g))
        uncertainty = torch.argmax(self.net_task_2(tensor)).item()
        if uncertainty == 1:
            return -1
        else:
            action = torch.argmax(self.net_task_1(tensor)).item()
            return action

    def reinforcement_learning(self, lv: LanguageAndVisionIU, l: LanguageIU, g: GestureIU) -> int:
        """Apply RL model to input"""
        array = self.build_vector(lv, l, g)
        uncertainty, _ = self.rl_task_2.predict(array)
        if uncertainty == 1:
            return -1
        else:
            action = self.rl_task_1.predict(array)[0]
            return action
