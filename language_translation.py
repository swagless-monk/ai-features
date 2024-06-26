import asyncio
import base64
import json
import pyaudio
import websockets
from translate import Translator
import pyttsx3 as tts
from warnings import filterwarnings

filterwarnings(action='ignore')

async def translate():
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

    # Translator
    translator = Translator(to_lang="es", from_lang="en")

    # Text to speech
    engine = tts.init()
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id)

    # AssemblyAI Real-time listener Endpoint & API key
    auth_key = 'dcb60a2d902c4bfeae9a3a14b53c2cce'
    transcription_endpoint = 'wss://api.assemblyai.com/v2/realtime/ws?sample_rate=16000'

    async with websockets.connect(
        transcription_endpoint,
        extra_headers = (("Authorization", auth_key),),
        ping_interval = 5,
        ping_timeout = 60
    ) as _ws:

        await asyncio.sleep(0.1)
        session_begins = await _ws.recv()
        #engine.say("I'm ready for you")

        async def sending():
            while True:
                try:
                    data = stream.read(frames_per_buffer)
                    data = base64.b64encode(data).decode('utf-8')
                    json_data = json.dumps({'audio_data':str(data)})
                    await _ws.send(json_data)
                except websockets.exceptions.ConnectionClosedError as e:
                    engine.say("The websocket connection closed unexpectedly.")
                    assert e.code == 4008
                    break
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    assert False, "Non websockets 4008 error"
                await asyncio.sleep(0.1)
            
            return True
        
        async def receiving():
            while True:
                try:
                    result_str = await _ws.recv()
                    if len(json.loads(result_str)['text']) > 0:
                        translation = translator.translate(json.loads(result_str)['text'])
                        engine.say(translation)
                        engine.runAndWait()
                        engine.stop()
                    else:
                        pass
                except websockets.exceptions.ConnectionClosedError as e:
                    #engine.say("The websocket connection closed unexpectedly.")
                    assert e.code == 4008
                    break
                except KeyboardInterrupt:
                    #engine.say("I'll be quiet now.")
                    break
                except Exception as e:
                    assert False, "Non websockets 4008 error"
        
        sending_result, receiving_result = await asyncio.gather(sending(), receiving())