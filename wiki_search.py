from warnings import filterwarnings

import speech_recognition as sr
import nltk.data
import pyttsx3 as tts
import wikipedia as wiki
from time import sleep

# Ignore all warnings
filterwarnings(action='ignore')

# Speech to text engine
listen = sr.Recognizer()

# Text to speech engine
engine = tts.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)

def talk(text:str) -> None:
    engine.say(text)
    engine.runAndWait()

def get_query() -> str:
    with sr.Microphone() as source:
        audio = listen.listen(source)

        voice = listen.recognize_google(audio)
        text = voice.lower()

    return text

def lookup() -> None:
    try:
        text = get_query()
        
        if text in ['quit', 'exit', 'kill', 'stop', 'break']:
            talk('I\'m here when you need me Swagless')
        else:
            talk('Searching...')
            try:
                result = wiki.summary(wiki.search(text)[0])
                tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
                info = tokenizer.tokenize(result)

                talk(info[:2])

            except IndexError:
                talk('That\'s a nonsensical request Swagless')
            except wiki.exceptions.PageError as e:
                talk('Sorry Swagless. I cannot find that page')
            except wiki.exceptions.DisambiguationError as e:
                talk (f'Sorry Swagless the query, {text}, is too ambiguous')

    except sr.UnknownValueError:
        talk('I do not understand')
