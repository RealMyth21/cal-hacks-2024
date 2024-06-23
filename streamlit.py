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
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;700&display=swap');

        body {
            font-family: 'Montserrat', sans-serif;
            margin: 0;
            padding: 0;
            color: #5a1111;
            background-color: #f0f2f5;
            line-height: 1.6;
        }

        header {
            background-color: #8a1465;
            color: white;
            padding: 15px 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }

        header h1 {
            margin: 0;
            font-weight: 700;
            letter-spacing: 1px;
        }

        header nav ul {
            list-style: none;
            padding: 0;
            margin: 0;
            display: flex;
        }

        header nav ul li {
            margin-left: 20px;
        }

        header nav ul li a {
            color: white;
            text-decoration: none;
            font-weight: 500;
            transition: color 0.3s;
        }

        header nav ul li a:hover {
            color: #d1c4e9;
        }

        main .container {
            padding: 20px;
            max-width: 900px;
            margin: 0 auto;
        }

        .hero {
            text-align: center;
            padding: 60px 0;
            background-color: #992155;
            color: white;
            margin-bottom: 30px;
            border-radius: 8px;
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
        }

        .hero h2 {
            margin: 0 0 20px;
            font-size: 2.5em;
        }

        .hero p {
            font-size: 1.2em;
            max-width: 700px;
            margin: 0 auto;
        }

        .features {
            display: flex;
            justify-content: space-around;
            margin-top: 20px;
            flex-wrap: wrap;
        }

        .feature {
            flex: 1;
            padding: 20px;
            margin: 10px;
            border: 1px solid #ddd;
            border-radius: 8px;
            text-align: center;
            background-color: #ffffff;
            transition: transform 0.3s, box-shadow 0.3s;
        }

        .feature:hover {
            transform: translateY(-5px);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }

        .feature h3 {
            margin-top: 0;
            font-size: 1.5em;
            color: #8a1443;
        }

        .upload {
            text-align: center;
            margin: 40px auto;
            background-color: #e8eaf6;
            padding: 40px;
            border-radius: 8px;
            border: 1px solid #ddd;
            max-width: 500px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }

        .upload input[type="file"] {
            display: block;
            margin: 20px auto;
            padding: 10px;
            font-size: 1em;
        }

        .upload button {
            background-color: #8a1447;
            color: white;
            border: none;
            padding: 12px 24px;
            cursor: pointer;
            border-radius: 8px;
            font-weight: bold;
            font-size: 1em;
            transition: background-color 0.3s, transform 0.3s;
        }

        .upload button:hover {
            background-color: #931b6d;
            transform: translateY(-3px);
        }

        footer {
            background-color: #8a1465;
            color: white;
            text-align: center;
            padding: 15px 0;
            margin-top: 40px;
            box-shadow: 0 -4px 6px rgba(0, 0, 0, 0.1);
        }

        footer p {
            margin: 0;
        }
    </style>
    """, unsafe_allow_html=True)

# Define the CNN model
class SimpleCNN(torch.nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = torch.nn.Linear(32 * 10 * (1000 // 4), 128)
        self.fc2 = torch.nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 32 * 10 * (1000 // 4))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load the pre-trained model
model_path = 'simple_cnn_model.pth'
num_classes = 10
model = SimpleCNN(num_classes)

# Load state_dict while ignoring size mismatch errors
state_dict = torch.load(model_path, map_location=torch.device('cpu'))
state_dict['fc2.weight'] = model.state_dict()['fc2.weight']
state_dict['fc2.bias'] = model.state_dict()['fc2.bias']
model.load_state_dict(state_dict, strict=False)
model.eval()

# Define emotion labels
emotion_labels = ['fear', 'anxiety', 'distress', 'surprise', 'sadness', 'confusion', 'horror', 'pain', 'realization']

# Define transform (MFCC)
transform = transforms.MFCC()

# Function to clean up and preprocess the audio
def clean_audio(audio_path, output_path='outputGun.wav', sr=16000, duration=5):
    y, sr = librosa.load(audio_path, mono=True, sr=sr, offset=0, duration=duration)
    scipy.io.wavfile.write(output_path, sr, (y * 32767).astype(np.int16))
    return output_path

# Function to predict the emotion of the audio
def predict_audio(model, audio_path, transform, max_len=1000):
    # Convert and clean up audio file
    cleaned_audio_path = clean_audio(audio_path)

    try:
        waveform, _ = torchaudio.load(cleaned_audio_path)
    except Exception as e:
        st.error(f"Failed to load audio: {e}")
        return None

    waveform = waveform.mean(dim=0, keepdim=True)  # Convert to mono

    if transform:
        waveform = transform(waveform)

    # Truncate or pad the waveform to the max length
    if waveform.size(2) > max_len:
        waveform = waveform[:, :, :max_len]
    elif waveform.size(2) < max_len:
        padding = max_len - waveform.size(2)
        waveform = torch.nn.functional.pad(waveform, (0, padding))

    waveform = waveform.unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = model(waveform)
        _, predicted = torch.max(output, 1)
        predicted_label = emotion_labels[predicted.item()]

    return predicted_label

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

        predicted_emotion = predict_audio(model, audio_path, transform)
        if predicted_emotion:
            st.write(f"The predicted emotion is: {predicted_emotion}")

        os.remove(audio_path)
