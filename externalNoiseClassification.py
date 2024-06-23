#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


np.version.version


# In[3]:


import scipy


# In[4]:


scipy.version.version


# In[5]:


audio_call = "/Users/medhavijam/Desktop/call_108.mp3"


# In[6]:


# from df.enhance import enhance, init_df, load_audio, save_audio
# from df.utils import download_file


# In[7]:


import librosa
import noisereduce as nr


# In[9]:


from IPython.display import Audio, IFrame, display

sr = 16000
display(Audio(audio_call, rate=sr))


# In[10]:


y1, sr = librosa.load(audio_call, mono=True, sr=sr, offset=0, duration=50)


# In[11]:


display(Audio(y1, rate=sr))


# In[12]:


audio_call2 = "/Users/medhavijam/Desktop/recording2.mp3"


# In[13]:


y2, sr = librosa.load(audio_call2, mono=True, sr=sr, offset=0, duration=50)


# In[14]:


display(Audio(y2, rate=sr))


# In[33]:


import soundfile as sf
import IPython.display as ipd


# In[34]:


# noise_clip = y2[:sr]
# reduced_noise = nr.reduce_noise(y2, noise_clip)
# display(Audio(reduced_noise, rate=sr))

# Read audio
data, samplerate = sf.read(audio_call2)
# reduce noise
y_reduced_noise = nr.reduce_noise(y=data, sr=samplerate)
# save audio
sf.write("Vocals_reduced.wav", y_reduced_noise, samplerate, subtype="PCM_24")
# load and play audio
data, samplerate = librosa.load("Vocals_reduced.wav")
ipd.Audio("Vocals_reduced.wav")


# In[16]:


import sounddevice as sd
from scipy.io.wavfile import write

# Sampling frequency
freq = 44100

# Recording duration
duration = 5

# Start recorder with the given values
# of duration and sample frequency
recording = sd.rec(
    int(duration * freq), samplerate=freq, channels=1
)  # Recording in mono (1 channel)

# Record audio for the given number of seconds
sd.wait()

# This will convert the NumPy array to an audio
# file with the given sampling frequency
write("recording4.wav", freq, recording)


# In[18]:


get_ipython().system("pip install pydub")


# In[19]:


from pydub import AudioSegment

AudioSegment.from_wav("recording4.wav").export("recording4.mp3", format="mp3")


# In[22]:


y2, sr = librosa.load(
    "/Users/medhavijam/coding/pyspark-project/recording4.mp3",
    mono=True,
    sr=sr,
    offset=0,
    duration=50,
)
display(Audio(y2, rate=sr))


# In[37]:


y2, sr = librosa.load(
    "/Users/medhavijam/Desktop/testGun1.m4a", mono=True, sr=16000, offset=0, duration=50
)
x = display(Audio(y2, rate=sr))
x


# In[38]:


# In[23]:


import asyncio
import json


# In[26]:


#!pip install hume


# In[28]:


from hume import HumeStreamClient
from hume.models.config import ProsodyConfig


# In[ ]:


from hume import HumeStreamClient
from hume.models.config import ProsodyConfig


async def main():
    client = HumeStreamClient("IawcufxKtWqusn6UgKTvkOhIu7mkMv71VS1KEMmzCF97UKok")
    config = ProsodyConfig()
    async with client.connect([config]) as socket:
        result = await socket.send_file("HELP.mp3")
        formatted_result = format_result(result)
        print(formatted_result)
        with open("output.txt", "w") as outfile:
            outfile.write(formatted_result)


def format_result(result):
    formatted_result = ""
    for prediction in result["prosody"]["predictions"]:
        formatted_result += (
            f"Time: {prediction['time']['begin']} - {prediction['time']['end']}\n"
        )
        for emotion in prediction["emotions"]:
            formatted_result += f"{emotion['name']}: {emotion['score']}\n"
        formatted_result += "\n"
    return formatted_result


asyncio.run(main())

# Read the result file
with open("output.txt", "r") as file:
    lines = file.readlines()

# Parse the emotions and scores
emotions = {}
current_emotion = None
for line in lines:
    if line.startswith("Time"):
        continue
    if not line.strip():
        continue
    if ":" in line:
        emotion, score = line.strip().split(": ")
        emotions[emotion] = float(score)

# Sort the emotions by score in descending order
sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)

# Write the sorted emotions to a new file
with open("sorted_output.txt", "w") as file:
    for emotion, score in sorted_emotions:
        file.write(f"{emotion}: {score}\n")


# In[ ]:


"""
20 second clip -- Hume AI can only take increments of 5 seconds max, so we will have 4 rounds in the for loop

in each iteration, record a 5 second clip
save that into a wav file
convert into an mp3 file
run that mp3 file to clean it up
run the cleaned file to hume_ai
hume ai program will output the top 10 emotions detected

as we get each round of hume outputs then it calculates the average of the emotions so the 
911 operator has a real time display of emotions
"""


# In[ ]:
