from fastapi import UploadFile, File, HTTPException, APIRouter
import torch
import torch.nn as nn
from torchaudio import transforms
import io
import torch.nn.functional as F
import soundfile as sf


class CheckAudio(nn.Module):
    def __init__(self):
      super().__init__()

      self.first =  nn.Sequential(
          nn.Conv2d(1,16, kernel_size=3, padding=1),
          nn.ReLU(),
          nn.MaxPool2d(2),
          nn.Conv2d(16,32, kernel_size=3, padding=1),
          nn.ReLU(),
          nn.MaxPool2d(2),
          nn.AdaptiveAvgPool2d((8, 8))
      )
      self.second = nn.Sequential(
          nn.Flatten(),
          nn.Linear(32 * 8 * 8, 128),
          nn.ReLU(),
          nn.Linear(128, 35)
      )


    def forward(self, x):
      x = x.unsqueeze(1)
      x = self.first(x)
      x = self.second(x)
      return x



labels = ['backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight',
           'five', 'follow','forward','four', 'go', 'happy', 'house',
           'learn', 'left', 'marvin', 'nine', 'no', 'off', 'on', 'one',
           'right', 'seven', 'sheila', 'six', 'stop', 'three','tree',
           'two', 'up', 'visual', 'wow', 'yes', 'zero']


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
index_to_labels = {ind: lab for ind, lab in enumerate(labels)}
model = CheckAudio()
model.load_state_dict(torch.load('mysite/models/speech_model.pth', map_location=device))
model.to(device)
model.eval()


transform = transforms.MelSpectrogram(
    sample_rate=16000,
    n_mels=64
)
max_len = 100

def change_audio(waveform, sample_rate):
    if sample_rate != 16000:
        new_sr = transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = new_sr(torch.tensor(waveform))

    spec = transform(waveform).squeeze(0)

    if spec.shape[1] > max_len:
        spec = spec[:, :max_len]

    if spec.shape[1] < max_len:
        count_len = max_len - spec.shape[1]
        F.pad(spec, (0, count_len))

    return spec

speech_router = APIRouter(prefix='/speech_predict', tags=['Specch Commands '])

@speech_router.post('/')
async def predict_audio(file: UploadFile = File(...)):
    try:
        data = await file.read()
        if not data:
            raise HTTPException(status_code=401, detail="File is empty")

        wf, sr = sf.read(io.BytesIO(data),dtype='float32')
        wf = torch.tensor(wf)

        spec = change_audio(wf, sr).unsqueeze(0).to(device)

        with torch.no_grad():
            y_pred = model(spec)
            pred_ind = torch.argmax(y_pred, dim=1).item()
            pred_class = index_to_labels[pred_ind]
            return {'Index': pred_ind,'Class': pred_class}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))








# classes = ['backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight',
#            'five', 'follow','forward','four', 'go', 'happy', 'house',
#            'learn', 'left', 'marvin', 'nine', 'no', 'off', 'on', 'one',
#            'right', 'seven', 'sheila', 'six', 'stop', 'three','tree',
#            'two', 'up', 'visual', 'wow', 'yes', 'zero']