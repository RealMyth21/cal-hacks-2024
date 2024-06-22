#!/usr/bin/env python
# coding: utf-8

# In[1]:


audio_call = "/Users/medhavijam/Desktop/call_108.mp3"


# In[ ]:


'''
goals:
- clean up audio file, remove background noises, make it more clear
- 
'''


# In[10]:


#!pip install torch torchaudio -f https://download.pytorch.org/whl/cpu/torch_stable.html


# In[9]:


#!pip install deepfilternet


# In[6]:


from df.enhance import enhance, init_df, load_audio, save_audio
from df.utils import download_file


# In[18]:


#!pip install librosa


# In[19]:


#!pip install noisereduce


# In[26]:


import librosa
import noisereduce as nr


# In[23]:


sr = 16000
y1,sr = librosa.load(audio_call, mono=True, sr=sr, offset=0, duration=50)


# In[33]:


from IPython.display import Audio, IFrame, display
display(Audio(audio_call,rate=sr))


# In[24]:


from IPython.display import Audio, IFrame, display
display(Audio(y1,rate=sr))
# apply noisereduce on this display
# THIS IS the best one


# In[27]:


import librosa.display


# In[30]:


y, sr = librosa.load(audio_call, mono=True, sr=sr, offset=0, duration=50)
y_trimmed, _ = librosa.effects.trim(y)
y_filtered = librosa.effects.preemphasis(y_trimmed)
y_normalized = librosa.util.normalize(y_filtered)


# In[32]:


display(Audio(y_normalized, rate=sr))


# In[36]:


audio_call2 = "/Users/medhavijam/Desktop/call_110.mp3"
sr = 16000
y2,sr = librosa.load(audio_call2, mono=True, sr=sr, offset=0, duration=650)
display(Audio(y2,rate=sr))


# In[37]:


get_ipython().system('pip install sounddevice')


# In[38]:


'''
import sounddevice as sd

# Parameters for live audio processing
duration = 10  # Duration to record in seconds
sample_rate = 16000  # Sampling rate

# Load the pre-recorded audio file
audio_call = "/Users/medhavijam/Desktop/call_108.mp3"
y1, sr = librosa.load(audio_call, mono=True, sr=sample_rate, offset=0, duration=50)

# Display the pre-recorded audio file
display(Audio(audio_call, rate=sample_rate))

# Function to process live audio data
def process_audio(indata, frames, time, status):
    if status:
        print(status, flush=True)
    # Convert to numpy array
    audio_data = np.array(indata).flatten()
    
    # Perform audio analysis with librosa
    # Example: Calculate the Mel spectrogram
    S = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    
    print("Mel spectrogram shape:", S_dB.shape)

# Open a stream to record and process live audio
with sd.InputStream(callback=process_audio, channels=1, samplerate=sample_rate):
    print("Recording live audio for {} seconds...".format(duration))
    sd.sleep(duration * 1000)

print("Live audio recording complete.")
'''


# In[ ]:




