import threading

from retico_core.audio import MicrophoneModule
from retico_googleasr.googleasr import GoogleASRModule
from LanguageAndVision import LanguageAndVisionModule
from Language import LanguageModule
from Gesture import GestureModule
from Periodic import PeriodicModule
from DialogManager import DialogManagerModule
from Motion import MotionModule

microphone_module = MicrophoneModule(chunk_size=44100)
asr_module = GoogleASRModule()
language_and_vision_module = LanguageAndVisionModule()
language_only_module = LanguageModule()
gesture_module = GestureModule()
periodic_module = PeriodicModule()
dialog_manager_module = DialogManagerModule(model="DT")
motion_module = MotionModule()

if __name__ == '__main__':
    microphone_module.subscribe(asr_module)
    asr_module.subscribe(language_and_vision_module)
    asr_module.subscribe(language_only_module)
    language_and_vision_module.subscribe(dialog_manager_module)
    language_only_module.subscribe(dialog_manager_module)
    gesture_module.subscribe(dialog_manager_module)
    periodic_module.subscribe(dialog_manager_module)
    dialog_manager_module.subscribe(motion_module)

    # microphone_module.run()
    # asr_module.run()
    language_and_vision_module.run()
    language_only_module.run()
    gesture_module.run()
    periodic_module.run()
    dialog_manager_module.run()
    motion_module.run()

    print("READY")
    input()

    motion_module.stop()
    dialog_manager_module.stop()
    language_and_vision_module.stop()
    language_only_module.stop()
    gesture_module.stop()
    periodic_module.stop()
    # asr_module.stop()
    # microphone_module.stop()
