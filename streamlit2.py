import streamlit as st
import torchaudio
import torch
import os
from torch.nn import functional as F
import torchaudio.transforms as transforms
from pydub import AudioSegment
import librosa
import scipy
import numpy as np
import aiohttp
import asyncio
from hume import HumeStreamClient
from hume.models.config import ProsodyConfig

# Define transform (MFCC)
transform = transforms.MFCC()

# Function to clean up and preprocess the audio
def clean_audio(audio_path, output_path='outputGun.wav', sr=16000, duration=5):
    y, sr = librosa.load(audio_path, mono=True, sr=sr, offset=0, duration=duration)
    scipy.io.wavfile.write(output_path, sr, (y * 32767).astype(np.int16))
    return output_path

# Function to format the result from Hume API
def format_result(result):
    formatted_result = ""
    for prediction in result["prosody"]["predictions"]:
        formatted_result += f"Time: {prediction['time']['begin']} - {prediction['time']['end']}\n"
        for emotion in prediction["emotions"]:
            formatted_result += f"{emotion['name']}: {emotion['score']}\n"
        formatted_result += "\n"
    return formatted_result

# Function to predict the emotion of the audio using Hume AI
async def predict_audio_hume(audio_path):
    client = HumeStreamClient("IawcufxKtWqusn6UgKTvkOhIu7mkMv71VS1KEMmzCF97UKok")
    config = ProsodyConfig()
    async with client.connect([config]) as socket:
        result = await socket.send_file(audio_path)
        formatted_result = format_result(result)
        return formatted_result

# Streamlit app
st.title("Audio Emotion Classifier")

uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a"])

if uploaded_file is not None:
    audio_path = f"temp_audio.{uploaded_file.name.split('.')[-1]}"
    with open(audio_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Debugging: Check if the file was written
    if not os.path.exists(audio_path):
        st.error(f"Failed to save the uploaded file to {audio_path}")
    else:
        st.audio(uploaded_file, format='audio/wav')

        # Clean and preprocess audio
        cleaned_audio_path = clean_audio(audio_path)

        # Run async function to get predictions
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        predicted_emotions = loop.run_until_complete(predict_audio_hume(cleaned_audio_path))

        if predicted_emotions:
            st.write("The predicted emotions are:")
            st.text(predicted_emotions)

        os.remove(audio_path)

