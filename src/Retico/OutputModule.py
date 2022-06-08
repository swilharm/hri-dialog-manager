from retico_core import AbstractConsumingModule
from retico_core.audio import AudioIU
from retico_core.text import SpeechRecognitionIU


class OutputModule(AbstractConsumingModule):
    """A module that writes the received text into a file."""

    @staticmethod
    def name():
        return "Output Module"

    @staticmethod
    def description():
        return "A module that outputs to console"

    @staticmethod
    def input_ius():
        return [SpeechRecognitionIU]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def setup(self):
        pass

    def shutdown(self):
        pass

    def process_update(self, update_message):
        for iu in update_message.incremental_units():
            print(iu.payload)