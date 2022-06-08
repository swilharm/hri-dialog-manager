from retico_core.abstract import AbstractConsumingModule

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
        pass
