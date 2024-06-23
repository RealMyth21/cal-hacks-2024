#!/usr/bin/env python
# coding: utf-8

# In[37]:


# pip install torch torchaudio numpy


# In[141]:


import sounddevice as sd
from scipy.io.wavfile import write
from pydub import AudioSegment
import numpy as np
import scipy
import librosa
import noisereduce as nr
from IPython.display import Audio, IFrame, display


AudioSegment.from_wav("/Users/charlottelaw/medhaviNew.wav").export(
    f"medhavi.mp3", format="mp3"
)
y, sr = librosa.load(
    "/Users/charlottelaw/medhavi.mp3", mono=True, sr=16000, offset=0, duration=5
)
scipy.io.wavfile.write("outputGun.wav", sr, (y * 32767).astype(np.int16))
display(Audio(y, rate=sr))
type(y)


from scipy.io import wavfile

# Load audio file using librosa
# y2, sr = librosa.load('/Users/medhavijam/Desktop/testGun1.m4a', mono=True, sr=16000, offset=0, duration=50)

# Specify the output WAV file path
output_file = "/Users/charlottelaw/outputGun.wav"

# Convert y2 to 16-bit PCM format (assuming y2 is in the range -1 to 1)
y2_pcm = (y * 32767).astype(np.int16)

# Write the audio data to a WAV file
wavfile.write(output_file, sr, y2_pcm)


# In[74]:


import os
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim


class AudioDataset(Dataset):
    def __init__(self, dataset_dir, transform=None, max_len=1000):
        self.dataset_dir = dataset_dir
        self.classes = [
            d
            for d in os.listdir(dataset_dir)
            if os.path.isdir(os.path.join(dataset_dir, d))
        ]
        self.files = []
        self.max_len = max_len
        self.transform = transform

        for label in self.classes:
            class_dir = os.path.join(dataset_dir, label)
            for file in os.listdir(class_dir):
                if file.endswith(".wav"):  # Assuming audio files are in WAV format
                    self.files.append((os.path.join(class_dir, file), label))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path, label = self.files[idx]
        waveform, _ = torchaudio.load(file_path)
        waveform = waveform.mean(dim=0, keepdim=True)  # Convert to mono

        if self.transform:
            waveform = self.transform(waveform)

        # Truncate or pad the waveform to the max length
        if waveform.size(2) > self.max_len:
            waveform = waveform[:, :, : self.max_len]
        elif waveform.size(2) < self.max_len:
            padding = self.max_len - waveform.size(2)
            waveform = torch.nn.functional.pad(waveform, (0, padding))

        return waveform, torch.tensor(self.classes.index(label))


def collate_fn(batch):
    waveforms, labels = zip(*batch)
    waveforms = torch.stack(waveforms)
    labels = torch.tensor(labels)
    return waveforms, labels


# Define the transform (MFCC)
transform = torchaudio.transforms.MFCC()

# Load the dataset and split into training and testing sets
dataset = AudioDataset("dataset/", transform=transform)
train_set, test_set = train_test_split(dataset, test_size=0.2, random_state=42)

# Create DataLoader instances
train_loader = DataLoader(train_set, batch_size=32, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False, collate_fn=collate_fn)


# Define a simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(
            32 * 10 * (dataset.max_len // 4), 128
        )  # Adjusted dimensions after pooling
        self.fc2 = nn.Linear(128, len(dataset.classes))  # Adjusted number of classes

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Create an instance of the model
model = SimpleCNN()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")

# Testing loop
model.eval()
total_correct = 0
total_samples = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total_samples += labels.size(0)
        total_correct += (predicted == labels).sum().item()

print(f"Accuracy: {total_correct / total_samples:.4f}")


# In[142]:


audio_path = "/Users/charlottelaw/outputGun.wav"


# In[143]:


import torchaudio
import torch
import os


def predict_audio(model, audio_path, transform, max_len=1000):
    model.eval()

    # Print absolute path for debugging
    abs_audio_path = os.path.abspath(audio_path)
    print(f"Absolute path of the audio file: {abs_audio_path}")

    # List files in the directory for debugging
    directory = os.path.dirname(abs_audio_path)
    print(f"Files in directory '{directory}': {os.listdir(directory)}")

    try:
        waveform, _ = torchaudio.load(abs_audio_path)
    except Exception as e:
        print(f"Error loading audio file {abs_audio_path}: {e}")
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
    waveform = waveform.to(device)

    with torch.no_grad():
        output = model(waveform)
        _, predicted = torch.max(output, 1)
        predicted_label = dataset.classes[predicted.item()]

    return predicted_label


# Test the function
test_audio_path = (
    "/Users/charlottelaw/outputGun.wav"  # Update this with your actual audio file path
)
predicted_label = predict_audio(model, test_audio_path, transform)
if predicted_label:
    print(f"The predicted label for the test audio is: {predicted_label}")


# In[147]:


import asyncio
import json

from hume import HumeStreamClient
from hume.models.config import ProsodyConfig


async def main():
    client = HumeStreamClient("IawcufxKtWqusn6UgKTvkOhIu7mkMv71VS1KEMmzCF97UKok")
    config = ProsodyConfig()
    async with client.connect([config]) as socket:
        result = await socket.send_file("outputGun.wav")
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


# Use asyncio.create_task() instead of asyncio.run()
asyncio.create_task(main())


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


# In[149]:


with open("sorted_output.txt", "r") as file:
    # Read the first 10 lines
    for _ in range(10):
        # Read a line from the file
        line = file.readline()
        # If the line is not empty, print it
        if line:
            print(line.strip())


# In[ ]:
