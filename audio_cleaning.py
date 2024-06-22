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


# In[ ]:




