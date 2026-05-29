from fastapi import  UploadFile, File, HTTPException, APIRouter
import torch
import torch.nn as nn
from torchaudio import transforms
import io
import torch.nn.functional as F
import soundfile as sf
import os
import torchaudio
import pandas as pd
from torch.utils.data import Dataset

urban_router = APIRouter(prefix='/urban', tags=['urban audio'])


df = pd.read_csv('csv/UrbanSound8K.csv')

class UrbanAudio(nn.Module):
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
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.first(x)
        x = self.second(x)
        return x


label = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 'drilling',
           'engine_idling', 'gun_shot', 'jackhammer', 'siren', 'street_music']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
label_to_index = {word: ind for ind, word in enumerate(label)}
model = UrbanAudio()
model.load_state_dict(torch.load('mysite/models/urban_model.pth', map_location=device))
model.to(device)
model.eval()
max_len = 500


transform = transforms.MelSpectrogram(
    sample_rate=22050,
    n_mels=64
)


class UrbanSoundDataset(Dataset):
    def __init__(self, data_path, transform, max_len):
        self.data_path = data_path
        self.transform = transform
        self.max_len = max_len
        self.audios = []

        for name in os.listdir(data_path):
            name_path = os.path.join(data_path, name)
            if not os.path.isdir(name_path):
                continue
            for file in os.listdir(name_path):
                if file.endswith('.wav'):
                    file_path = os.path.join(name_path, file)

                    try:
                        classes = list(df[df['slice_file_name'] == file]['class'])[0]
                        self.audios.append((file_path, classes))
                    except Exception as e:
                        print(f'Error: {e}')

    def __len__(self):
        return len(self.audios)

    def __getitem__(self, idx):
        file_path, classes = self.audios[idx]
        waveform, sr = torchaudio.load(file_path)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sr != 22050:
            resample = transforms.Resample(sr, 22050)
            waveform = resample(waveform)

        spec = self.transform(waveform)

        spec = (spec - spec.mean()) / (spec.std() + 1e-6)

        if spec.shape[-1] > self.max_len:
            spec = spec[:, :, :self.max_len]
        else:
            pad = self.max_len - spec.shape[-1]
            spec = F.pad(spec, (0, pad))

        return spec, label_to_index[classes]



@urban_router.post('/')
async def urban_audio(file: UploadFile = File(...)):
    try:
        data = await file.read()
        if not data:
            raise HTTPException(status_code=422, detail="File is empty")

        wf, sr = sf.read(io.BytesIO(data), dtype='float32')
        wf = torch.from_numpy(wf).to(torch.float32)

        if wf.ndim > 1:
            wf = wf.T
            wf = wf.mean(dim=0, keepdim=True)
        else:
            wf = wf.unsqueeze(0)

        if sr != 22050:
            resample = transforms.Resample(orig_freq=sr, new_freq=22050)
            wf = resample(wf)

        spec = transform(wf)

        if spec.shape[-1] > max_len:
            spec = spec[:, :, :max_len]
        else:
            pad_amount = max_len - spec.shape[-1]
            spec = F.pad(spec, (0, pad_amount))

        spec = spec.unsqueeze(0).to(device)

        with torch.no_grad():
            y_pred = model(spec)
            pred_ind = torch.argmax(y_pred, dim=1).item()

            return {
                'Index': pred_ind,
                'Class': label[pred_ind]
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

