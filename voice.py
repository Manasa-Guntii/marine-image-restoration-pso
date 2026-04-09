import pyttsx3

engine = pyttsx3.init()

def speak(text):
    engine.setProperty('rate',150)
    engine.setProperty('volume',1)

    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id)

    engine.say(text)
    engine.runAndWait()