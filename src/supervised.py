import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import Dataset
from torchvision import models
from .audio_preprocessing import *

class AudioDataset(Dataset):
    def __init__(self, fbank_features, annotation):
        self.fbank_features = fbank_features
        self.annotation = annotation

    def __len__(self):
        return len(self.fbank_features)

    def __getitem__(self, idx):
        fbank_features = self.fbank_features[idx]
        annotation = self.annotation[idx]
        fbank_features_array = np.array(fbank_features)
        fbank_features_tensor = torch.tensor(fbank_features_array, dtype=torch.float32)
        annotation_tensor = torch.tensor(annotation, dtype=torch.float32)
        return fbank_features_tensor, annotation_tensor

class AudioMobileNetV2(nn.Module):
    def __init__(self):
        super(AudioMobileNetV2, self).__init__()
        self.mobilenetv2 = models.mobilenet_v2(pretrained=True)
        self.mobilenetv2.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.mobilenetv2.classifier[1] = nn.Linear(self.mobilenetv2.last_channel, 1)  # Binary classification

    def forward(self, x):
        batch_size, num_frame, feature_dim = x.size()
        x = x.view(batch_size * num_frame, 1, 1, feature_dim)  
        x = self.mobilenetv2(x)
        x = x.view(batch_size, num_frame, -1)  
        return x

def train(model, train_loader, device, num_epochs=10):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=0.001)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        count = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            count += 1
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs.view(-1)
            labels = labels.view(-1).float()  # Convert labels to float for BCEWithLogitsLoss
                
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        scheduler.step()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}, LR: {scheduler.get_last_lr()[0]}')

def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")