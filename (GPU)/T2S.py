# import pyttsx3
from gtts import gTTS
from io import BytesIO
import os

#text to speech function
def text_to_speech(text, language='en', output_file='output.mp3'):
    tts = gTTS(text=text, lang=language, slow=False)
    tts.save(output_file)
    return output_file
