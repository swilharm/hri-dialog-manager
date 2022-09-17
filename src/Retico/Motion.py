from retico_core.abstract import AbstractConsumingModule, UpdateType

from DialogManager import DialogManagerIU


class MotionModule(AbstractConsumingModule):

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
                self.send_to_motion_team(input_iu)
            else:
                pass

    def send_to_motion_team(self, dm_iu: DialogManagerIU):
        # TODO once that is figured out
        pass
