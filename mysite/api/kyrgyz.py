from fastapi import  UploadFile, HTTPException, File, APIRouter
import os
import io
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from torchaudio import transforms
from torch.utils.data import Dataset

kyrgyz_router = APIRouter(prefix="/kyrgyz_predict", tags=["Kyrgyz Commands"])

class KyrgyzSpeechAudio(nn.Module):
    def __init__(self):
        super().__init__()

        self.first = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )

        self.second = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, len(classes))
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.first(x)
        x = self.second(x)
        return x


classes = torch.load('labels/kyrgyz_classes.pth')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
label_to_index = {word: ind for ind, word in enumerate(classes)}
index_to_label = {ind: word for ind, word in enumerate(classes)}

model = KyrgyzSpeechAudio()
model.load_state_dict(torch.load('mysite/models/kyrgyz_model.pth', map_location=device))
model.to(device)
model.eval()
max_len = 100

transform = transforms.MelSpectrogram(
    sample_rate=16000,
    n_mels=64,
)


class KyrgyzSpeech(Dataset):
    def __init__(self, data_path, transform, max_len):
        self.data_path = data_path
        self.transform = transform
        self.max_len = max_len
        self.audios = []

        for name in os.listdir(data_path):
            name_path = os.path.join(data_path, name)

            if not os.path.isdir(name_path):
                continue

            for audio in os.listdir(name_path):
                if audio.endswith('.wav'):
                    audio_path = os.path.join(name_path, audio)
                    self.audios.append((audio_path, name))

    def __len__(self):
        return len(self.audios)

    def __getitem__(self, index):
        audio_path, name = self.audios[index]
        waveform, sr = torchaudio.load(audio_path)

        if sr != 16000:
            resample = transforms.Resample(orig_freq=sr, new_freq=16000)
            waveform = resample(waveform)

        spec = self.transform(waveform).squeeze(0)

        if spec.shape[1] > self.max_len:
            spec = spec[:, :self.max_len]

        if spec.shape[1] < self.max_len:
            count_len = self.max_len - spec.shape[1]
            spec = F.pad(spec, (0, count_len))

        return spec, label_to_index[name]


def change_audio(waveform, sample_rate):
    if sample_rate != 16000:
        new_sr = transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = new_sr(waveform)

    spec = transform(waveform).squeeze(0)

    if spec.shape[-1] > max_len:
        spec = spec[:, :max_len]

    if spec.shape[-1] < max_len:
        count_len = max_len - spec.shape[-1]
        spec = F.pad(spec, (0, count_len))

    return spec



@kyrgyz_router.post('/')
async def predict_audio(file: UploadFile = File(...)):
    try:
        data = await file.read()
        if not data:
            raise HTTPException(status_code=401, detail='File is empty')

        import soundfile as sf
        import numpy as np
        wf, sr = sf.read(io.BytesIO(data), dtype='float32')
        wf = torch.tensor(wf)
        if wf.dim() == 1:
            wf = wf.unsqueeze(0)
        else:
            wf = wf.T

        spec = change_audio(wf, sr).unsqueeze(0).to(device)

        with torch.no_grad():
            y_pred = model(spec)
            pred_index = torch.argmax(y_pred, dim=1).item()
            pred_class = index_to_label[pred_index]
            return {'Index': pred_index, 'Class': pred_class}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


