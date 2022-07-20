import json

from retico_core.audio import MicrophoneModule
from retico_googleasr.googleasr import GoogleASRModule

from DialogManager import DialogManagerModule
from Gesture import GestureModule
from Language import LanguageModule
from LanguageAndVision import LanguageAndVisionModule
from Motion import MotionModule
from OutputModule import OutputModule
from data.dataset import DATASET, DATASET_INDEX

microphone_module = MicrophoneModule(chunk_size=44100)
asr_module = GoogleASRModule(language="en-US")
language_and_vision_module = LanguageAndVisionModule()
language_only_module = LanguageModule()
gesture_module = GestureModule()
dialog_manager_module = DialogManagerModule()
motion_module = MotionModule()
output_module = OutputModule()


def loadExampleData():
    with open("src/Retico/data/test_data_examples.json") as file:
        data = json.loads(file.read())
    for i in data[0]:
        DATASET["lv"].append(i[0])
        DATASET["g"].append(i[1])
        DATASET["l"].append(i[2])


if __name__ == '__main__':
    microphone_module.subscribe(asr_module)
    #asr_module.subscribe(output_module)
    asr_module.subscribe(language_and_vision_module)
    asr_module.subscribe(language_only_module)
    language_and_vision_module.subscribe(dialog_manager_module)
    language_only_module.subscribe(dialog_manager_module)
    gesture_module.subscribe(dialog_manager_module)
    dialog_manager_module.subscribe(motion_module)

    loadExampleData()

    microphone_module.run()
    asr_module.run()
    language_and_vision_module.run()
    language_only_module.run()
    gesture_module.run()
    dialog_manager_module.run()
    motion_module.run()
    output_module.run()

    print("READY")
    input()

    microphone_module.stop()
    asr_module.stop()
    language_and_vision_module.stop()
    language_only_module.stop()
    gesture_module.stop()
    dialog_manager_module.stop()
    motion_module.stop()
    output_module.stop()
