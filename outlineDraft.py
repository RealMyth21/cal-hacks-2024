#!/usr/bin/env python
# coding: utf-8

# In[2]:


import sounddevice as sd
from scipy.io.wavfile import write
from pydub import AudioSegment
import numpy as np
import scipy
from df.enhance import enhance, init_df, load_audio, save_audio
from df.utils import download_file
import librosa
import noisereduce as nr
from IPython.display import Audio, IFrame, display


# In[ ]:


def process_audio(number):
    freq = 44100  
    duration = 5
    recording = sd.rec(int(duration * freq),
                       samplerate=freq, channels=1)  # Recording in mono (1 channel)
    sd.wait()
    write(f"audio{number}.wav", freq, recording) # this will get written into medhavijam/coding/pyspark
    # converts audio from wav format to mp3 format
    AudioSegment.from_wav(f"audio{number}.wav").export(f"audio{number}.mp3", format="mp3")
    y,sr = librosa.load(f'/Users/medhavijam/coding/pyspark-project/audio{number}.mp3', mono=True, sr=16000, offset=0, duration=5)
    display(Audio(y,rate=sr))


# In[ ]:


iterations = 4
for i in range(iterations):
    # records audio
    process_audio(i)
    

    

