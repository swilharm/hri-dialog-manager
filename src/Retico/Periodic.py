import threading

from retico_core import IncrementalUnit, UpdateMessage, UpdateType, AbstractTriggerModule


class PeriodicIU(IncrementalUnit):

    @staticmethod
    def type():
        return "Periodic IU"


class PeriodicModule(AbstractTriggerModule):
    """
    This module periodically calls the DM to ensure that IUs that have been ignored
    because they are too new, are considered later even if no new IU arrives.
    """

    def __init__(self):
        super().__init__()
        self.loop = threading.Timer(0.05, self.trigger)

    @staticmethod
    def name():
        return "RecallModule"

    @staticmethod
    def description():
        return "This module periodically calls the Dialog Manager module"

    @staticmethod
    def output_iu():
        return PeriodicIU

    def prepare_run(self):
        self.loop.start()

    def shutdown(self):
        self.loop.cancel()

    def trigger(self, **kwargs):
        self.append(UpdateMessage.from_iu(self.create_iu(), UpdateType.ADD))
        self.loop = threading.Timer(0.05, self.trigger)
        self.loop.start()