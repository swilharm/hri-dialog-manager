from retico_core.abstract import AbstractConsumingModule, UpdateType

from DialogManager import DialogManagerIU


class MotionModule(AbstractConsumingModule):

    def __init__(self):
        super().__init__()
        self.current_input = []

    @staticmethod
    def name():
        return "Motion Module"

    @staticmethod
    def description():
        return "Module that represents task 5"

    @staticmethod
    def input_ius():
        return [DialogManagerIU]

    def process_update(self, update_message):
        for input_iu, update_type in zip(update_message.incremental_units(), update_message.update_types()):
            input_iu: DialogManagerIU
            if update_type == UpdateType.ADD:
                if input_iu.flag != -1:
                    self.current_input.append(input_iu)
                    send_to_motion_team(input_iu)
            else:
                self.current_input.remove(input_iu)


def send_to_motion_team(dm_iu: DialogManagerIU):
    # TODO once that is figured out
    print(dm_iu.confidence_decision, dm_iu.decision_coordinate, dm_iu.flag)
    pass
