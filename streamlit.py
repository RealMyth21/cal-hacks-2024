import streamlit as st
import torch
import torchaudio
from torch.utils.data import DataLoader
import numpy as np
import os
import librosa

# Define the model architecture and parameters (same as in your original code)
class SimpleCNN(torch.nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = torch.nn.Linear(32 * 10 * (1000 // 4), 128)  # Adjusted dimensions after pooling
        self.fc2 = torch.nn.Linear(128, 10)  # Adjusted number of classes

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize the model
model = SimpleCNN()

# Load the model state dict
@st.cache(allow_output_mutation=True)
def load_model():
    model = SimpleCNN()  # Create an instance of the model
    model.load_state_dict(torch.load('simple_cnn_model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

# Your Streamlit app
st.title("Audio Classification")

# Load the model
model = load_model()

# Audio processing function
def process_audio(audio_file):
    waveform, _ = torchaudio.load(audio_file)
    waveform = waveform.mean(dim=0, keepdim=True)  # Convert to mono
    transform = torchaudio.transforms.MFCC()
    waveform = transform(waveform)
    if waveform.size(2) > 1000:
        waveform = waveform[:, :, :1000]
    elif waveform.size(2) < 1000:
        padding = 1000 - waveform.size(2)
        waveform = torch.nn.functional.pad(waveform, (0, padding))
    return waveform

# Streamlit app
def main():
    st.title('Audio Classification')
    uploaded_file = st.file_uploader("Choose an audio file (.wav)", type="wav")

    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')

        # Process the uploaded audio file
        audio_path = os.path.join("uploads", uploaded_file.name)
        with open(audio_path, 'wb') as f:
            f.write(uploaded_file.getvalue())

        waveform = process_audio(audio_path)
        output = model(waveform.unsqueeze(0)).detach().numpy()

        emotions = ['fear', 'anxiety', 'distress', 'surprise', 'sadness', 'confusion', 'horror', 'surprise', 'pain', 'realization']
        
        st.subheader("Emotion Scores:")
        for i, emotion in enumerate(emotions):
            st.write(f"{emotion}: {output[0][i]}")

if __name__ == '__main__':
    main()
