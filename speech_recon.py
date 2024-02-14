import os
import torch
import librosa
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim


def audio_to_mfcc(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return mfcc


class AudioDataset(Dataset):
    def __init__(self, folder_path, labels):
        self.audio_paths = [os.path.join(folder_path, file) for file in
                            os.listdir(folder_path) if file.endswith('.wav')]
        self.labels = labels

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        mfcc = audio_to_mfcc(self.audio_paths[idx])
        mfcc_tensor = torch.tensor(mfcc, dtype=torch.float32)
        # Adjust MFCC tensor size here if necessary
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return mfcc_tensor, label


class SimpleAudioNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleAudioNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)  # Flatten the MFCC
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


folder_path = 'AI_voice_recon/training_files/I_think_were_gonna_win'
audio_files = [file for file in os.listdir(folder_path)
               if file.endswith('.wav')]
labels = [1] * len(audio_files)  # Assuming all files are positive examples

# Initialize dataset with the folder path
dataset = AudioDataset(folder_path=folder_path, labels=labels)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Adjust these values based on the shape of your MFCCs
input_size = 13 * 44  # Example: 13 MFCCs * 44 frames
num_classes = 2  # Binary classification

model = SimpleAudioNet(input_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 5  # Number of epochs to train for

for epoch in range(num_epochs):
    loss = None
    for inputs, labels in dataloader:
        # Resize inputs to match model's input size
        inputs = inputs.view(inputs.size(0), -1)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')
