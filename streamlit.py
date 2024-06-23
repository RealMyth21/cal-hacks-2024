import streamlit as st
import torchaudio
import torch
import os
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
import torchaudio.transforms as transforms

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
model.load_state_dict(torch.load(model_path))
model.eval()

# Define emotion labels
emotion_labels = ['fear', 'anxiety', 'distress', 'surprise', 'sadness', 'confusion', 'horror', 'surprise', 'pain', 'realization']

# Define transform (MFCC)
transform = transforms.MFCC()

# Function to predict the emotion of the audio
def predict_audio(model, audio_path, transform, max_len=1000):
    waveform, _ = torchaudio.load(audio_path)
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
    
    st.audio(uploaded_file, format='audio/wav')
    
    predicted_emotion = predict_audio(model, audio_path, transform)
    st.write(f"The predicted emotion is: {predicted_emotion}")

    os.remove(audio_path)
