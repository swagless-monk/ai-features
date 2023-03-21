import asyncio
import base64
import json
import pyaudio
import websockets
from warnings import filterwarnings
from threading import Thread
from sys import exit 

import pyttsx3 as tts
import cv2

from function_timer import run
from emotions import emotions
from live_video_augmented_reality import live_video
from wiki_search import lookup

filterwarnings(action='ignore')

engine = tts.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)

frames_per_buffer = 3200
format = pyaudio.paInt16
channels = 1
rate = 16000
listener = pyaudio.PyAudio()

stream = listener.open(
    format = format,
    channels = channels,
    rate = rate,
    input = True,
    frames_per_buffer = frames_per_buffer
)

# AssemblyAI Real-time listener Endpoint & API key
auth_key = 'dcb60a2d902c4bfeae9a3a14b53c2cce'
transcription_endpoint = 'wss://api.assemblyai.com/v2/realtime/ws?sample_rate=16000'

def features(user_input:str) -> None:
    loop = asyncio.new_event_loop()

    if user_input.lower() == 'emotions':
        cv2.destroyAllWindows()
        for i in range(1):
            engine.say('Beginning emotion recognition')
            engine.runAndWait()
        run(emotions, timeout=22)
        for i in range(1):
            engine.say('Emotion recognition ended. Waiting for your command')
            engine.runAndWait()
        loop.close()
        loop = asyncio.new_event_loop()
        Thread(group=None, target=send_receive).start()
    elif user_input.lower() == 'video':
        cv2.destroyAllWindows()
        for i in range(1):
            engine.say('Beginning video feed.')
            engine.runAndWait()
        run(live_video, timeout=30)
        for i in range(1):
            engine.say('Video feed closed. Waiting for your command')
            engine.runAndWait()
        loop.close()
        loop = asyncio.new_event_loop()
        Thread(group=None, target=send_receive).start()
    elif user_input.lower() == 'search':
            for i in range(1):
                engine.say('What would you like to know?')
                engine.runAndWait()
            lookup()
            for i in range(1):
                engine.say('Exiting lookup. Waiting for your command')
                engine.runAndWait()
            loop.close()
            loop = asyncio.new_event_loop()
            Thread(group=None, target=send_receive).start()
    elif user_input.lower() == 'kill':
        for i in range(1):
            engine.say('Until next time Swagless')
            engine.runAndWait()
        exit()

async def send_receive():
    async with websockets.connect(
        transcription_endpoint,
        extra_headers = (("Authorization", auth_key),),
        ping_interval = 5,
        ping_timeout = 999
    ) as _ws:

        await asyncio.sleep(0.1)
        session_begins = await _ws.recv()

        async def send() -> None:
            while True:
                try:
                    data = stream.read(frames_per_buffer)
                    data = base64.b64encode(data).decode('utf-8')
                    json_data = json.dumps({'audio_data':str(data)})
                    await _ws.send(json_data)
                except websockets.exceptions.ConnectionClosedError as e:
                    assert e.code == 4008
                    break
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    assert False, "Non websockets 4008 error"
                await asyncio.sleep(0.1)
            return True
        
        async def receive() -> None:
            while True:
                try:
                    result_str = await _ws.recv()
                    if len(json.loads(result_str)['text']) > 0:
                        Thread(features(json.loads(result_str)['text'])).start()
                    else:
                        pass
                except websockets.exceptions.ConnectionClosedError as e:
                    assert e.code == 4008
                    break
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    assert False, "Non websockets 4008 error"
        
        send_result, receive_result = await asyncio.gather(send(), receive())

if __name__ == '__main__':
    for i in range(1):
        engine.say('Hello Swagless and welcome. I\'m waiting for your command.')
        engine.runAndWait()
    while True:
        try:
            asyncio.run(send_receive())
        except KeyboardInterrupt as e:
            break
