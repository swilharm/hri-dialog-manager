import threading

from retico_core.abstract import IncrementalUnit, UpdateMessage, UpdateType, AbstractTriggerModule

from data.data import DATASET
from src.group_F_gesture_control.integration.gesture_recognition import GestureRecognition
from src.group_F_gesture_control.integration.robot_space import TopDownMap

NUM_PIECES = 15
ID2COORD = {i: (i, i, i) for i in range(NUM_PIECES)}
COORD2ID = {}
ARTIFICIAL_DELAY = 0.1


class GestureIU(IncrementalUnit):

    def __init__(self, creator=None, iuid=0, previous_iu=None, grounded_in=None, payload=None, processed=False
                 ):
        super().__init__(
            creator,
            iuid,
            previous_iu,
            grounded_in,
            payload,
        )
        self.processed = processed
        self.payload = payload
        self.confidence_instruction = 0.0
        self.coordinates = {i: 0.0 for i in range(NUM_PIECES)}

    @staticmethod
    def type():
        return "Gesture IU"


class GestureModule(AbstractTriggerModule):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loop = threading.Timer(1, self.trigger)

    @staticmethod
    def name():
        return "Gesture Module"

    @staticmethod
    def description():
        return "Module that represents task 4"

    @staticmethod
    def output_iu():
        return GestureIU

    def setup(self):
        self.map = TopDownMap(
            camera_id=-1,  # camera to photograph the robot space, < 0 defaults to test img
            scale=(0, 500),  # scale of the robot space, to be defined by robot control
            piece_height=1,  # height of the pieces in the scale of the robot space
            output_segmentation=False,  # prints the detected contours to the input image
            use_dummy_pieces=False  # for testing purposes
        )
        self.map.update()
        for i, piece in enumerate(self.map.pieces):
            ID2COORD[i] = (piece[1].x, piece[1].y, piece[1].z)
            COORD2ID[(piece[1].x, piece[1].y, piece[1].z)] = i
        self.g_rec = GestureRecognition(self.map, real_sense_test_mode=True)

    def prepare_run(self):
        self.loop.start()

    def shutdown(self):
        self.loop.cancel()

    def trigger(self, **kwargs):
        iu: GestureIU = self.create_iu()
        iu.grounded_in = iu

        # DATASET
        # datapoint = DATASET.get_sample()
        # iu.confidence_instruction = datapoint[1]
        # iu.coordinates = {i: j for i, j in enumerate(datapoint[3 + NUM_PIECES:])}

        # INTEGRATION
        targets = self.g_rec.get_targets()
        if targets:
            iu.confidence_instruction = 1.0
            piece = targets[0][1]
            piece_id = COORD2ID[(piece.x, piece.y, piece.z)]
            iu.coordinates[piece_id] = 1.0

            self.append(UpdateMessage.from_iu(iu, UpdateType.ADD))

        self.loop = threading.Timer(1, self.trigger)
        self.loop.start()
