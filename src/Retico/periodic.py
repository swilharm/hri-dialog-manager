import threading
import time

from retico_core import IncrementalUnit, UpdateMessage, UpdateType, AbstractTriggerModule


class PeriodicIU(IncrementalUnit):

    @staticmethod
    def type():
        return "Periodic IU"


class PeriodicModule(AbstractTriggerModule):

    def __init__(self):
        super().__init__()
        self.last_update = time.time()

    @staticmethod
    def name():
        return "RecallModule"

    @staticmethod
    def description():
        return "This module periodically calls the Dialog Manager module"

    @staticmethod
    def output_iu():
        return PeriodicIU

    def trigger(self, **kwargs):
        self.right_buffers()[-1].put(UpdateMessage.from_iu(self.create_iu(), UpdateType.ADD))
        threading.Timer(0.05, self.trigger).start()