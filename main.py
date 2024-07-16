import asyncio
import os

import numpy as np
import pyaudio
import torch
from faster_whisper import WhisperModel

from openvoice_tts import llm_chain


device = "cuda" if torch.cuda.is_available() else "cpu"
model = WhisperModel("tiny.en", device=device, compute_type="float16")


def transcribe_audio(audio_array: np.ndarray):
    audio_array = audio_array.squeeze() if audio_array.ndim > 1 else audio_array
    segments, info = model.transcribe(audio_array)
    content = "".join(segment.text for segment in segments)
    return content


FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 22050
CHUNK = 1024
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("Listening...")

while True:
    data = stream.read(CHUNK)
    audio_bytes = np.frombuffer(data, dtype=np.float16)
    response = transcribe_audio(audio_bytes)
    if response:
        print(response)
        asyncio.run(llm_chain(response))
