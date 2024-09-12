import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import Dataset
from torchvision import models
from sklearn.metrics import f1_score, accuracy_score
from metrics.event_based_metrics import event_metrics
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

class AudioBiLSTM(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.lstm1 = nn.LSTM(num_features, 128, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(256, 128, bidirectional=True, batch_first=True)
        self.dense = nn.Linear(256, 1)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        out = self.dense(out)
        return out
    
class AudioTransformer(nn.Module):
    def __init__(self, input_dim=41, hidden_dim=128, num_heads=4, num_layers=2):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, 1) 

    def forward(self, x):
        batch_size, num_frame, feature_dim = x.size()
        x = x.view(batch_size*num_frame, 1, feature_dim)
        x = self.input_projection(x)
        x = x.permute(1, 0, 2)  
        transformer_out = self.transformer_encoder(x)
        out = transformer_out[0, :, :]
        out = self.fc(out)
        out = out.view(batch_size, num_frame, 1)
        return out

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

def eval(model, test_loader, device):
    model.eval()
    acc_list = []
    framef_list = []
    eventf_list = []
    iou_list = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            # Move inputs and labels to GPU
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            outputs = outputs.view(-1)
            labels = labels.view(-1).float()  # Convert labels to float for BCEWithLogitsLoss
            preds = torch.sigmoid(outputs)  # Apply sigmoid to get probabilities
            preds = (preds > 0.5).float()  # Convert probabilities to binary predictions
            labels = labels.cpu().numpy()
            preds = preds.cpu().numpy()
            # Frame-based accuracy
            accuracy = accuracy_score(labels, preds)
            acc_list.append(accuracy)
            # Frame-based F1 score
            framef = f1_score(labels, preds)
            framef_list.append(framef)
            # Event-based metrics
            eventf, iou, counted_events, fake_events, undetected_events = event_metrics(labels, preds, tolerance=9, overlap_threshold=0.75)
            eventf_list.append(eventf)
            iou_list.append(iou)
    return acc_list, framef_list, eventf_list, iou_list

def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")



class AsymmetricalFocalLoss(nn.Module):
    def __init__(self, gamma=0, zeta=0):
        super(AsymmetricalFocalLoss, self).__init__()
        self.gamma = gamma   # balancing between classes
        self.zeta = zeta     # balancing between active/inactive frames

    def forward(self, pred, target):
        losses = - (((1 - pred) ** self.gamma) * target * torch.clamp_min(torch.log(pred), -100) +
                    (pred ** self.zeta) * (1 - target) * torch.clamp_min(torch.log(1 - pred), -100))
        return torch.mean(losses)

def train_FDYSED(model, train_loader, device, num_epochs=10):
    criterion = AsymmetricalFocalLoss(gamma=2, zeta=0.5)
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

# MDFDSED
def obtain_loss(train_cfg, model_outs, labels, weak_labels, mask_strong, mask_weak):
    strong_pred_stud, strong_pred_tch, weak_pred_stud, weak_pred_tch = model_outs
    loss_total = 0

    # loss_class_weak = train_cfg["criterion_class"](weak_pred_stud[mask_weak], weak_labels)
    # loss_cons_weak = train_cfg["criterion_cons"](weak_pred_stud, weak_pred_tch.detach())

    w_cons = train_cfg["w_cons_max"] * train_cfg["scheduler"]._get_scaling_factor()
    loss_class_strong = train_cfg["criterion_class"](strong_pred_stud[:], labels[:]) #strong masked label size = [bs_strong, n_class, frames]
    loss_cons_strong = train_cfg["criterion_cons"](strong_pred_stud, strong_pred_tch.detach())
    loss_total += loss_class_strong + w_cons * (loss_cons_strong) # train_cfg["w_weak"] * loss_class_weak + \  + train_cfg["w_weak_cons"] * loss_cons_weak

    return loss_total #, loss_class_strong, loss_class_weak, loss_cons_strong, loss_cons_weak

def train_MDFDSED(model, train_loader, device, num_epochs=10):
    train_cfg = yaml.load(open("./config_MDFDbest.yaml", "r"), Loader=yaml.Loader)
    criterion = obtain_loss
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
                
            loss = criterion(train_cfg, outputs, labels, None, None, None)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        scheduler.step()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}, LR: {scheduler.get_last_lr()[0]}')

