from retico_core.audio import MicrophoneModule
from retico_googleasr.googleasr import GoogleASRModule

from DialogManager import DialogManagerModule
from Gesture import GestureModule
from Language import LanguageModule
from LanguageAndVision import LanguageAndVisionModule
from Motion import MotionModule
from OutputModule import OutputModule

microphone_module = MicrophoneModule(chunk_size=44100)
asr_module = GoogleASRModule(language="en-US")
language_and_vision_module = LanguageAndVisionModule()
language_only_module = LanguageModule()
gesture_module = GestureModule()
dialog_manager_module = DialogManagerModule()
motion_module = MotionModule()
output_module = OutputModule()

if __name__ == '__main__':
    microphone_module.subscribe(asr_module)
    asr_module.subscribe(output_module)
    asr_module.subscribe(language_and_vision_module)
    asr_module.subscribe(language_only_module)
    language_and_vision_module.subscribe(dialog_manager_module)
    language_only_module.subscribe(dialog_manager_module)
    gesture_module.subscribe(dialog_manager_module)
    dialog_manager_module.subscribe(motion_module)

    microphone_module.run()
    asr_module.run()
    language_and_vision_module.run()
    language_only_module.run()
    gesture_module.run()
    dialog_manager_module.run()
    motion_module.run()
    output_module.run()

    input()

    microphone_module.stop()
    asr_module.stop()
    language_and_vision_module.stop()
    language_only_module.stop()
    gesture_module.stop()
    dialog_manager_module.stop()
    motion_module.stop()
    output_module.stop()