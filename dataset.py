import torch
import torchaudio
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path

#defining a dataset class with label extraction from filename
class AudioDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.files = sorted(list(Path(folder_path).glob("*.wav")))
        self.transform = transform
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_fft=1024,
            hop_length=512, n_mels=64
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        waveform, sr = torchaudio.load(path)
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
        mel = self.mel_spectrogram(waveform)
        mel_db = torchaudio.transforms.AmplitudeToDB()(mel)

        if self.transform:
            mel_db = self.transform(mel_db)

        filename = path.name.lower()
        label = 1 if "anomaly" in filename else 0  #filename should be indication of an anomaly

        return mel_db, torch.tensor(label, dtype=torch.long)


# transformations to ensure data is consistent in shape and has a normalised range
transform = transforms.Compose([
    transforms.Lambda(lambda x: x.squeeze(0)),     # [1, H, W] -> [H, W]
    transforms.Lambda(lambda x: x.unsqueeze(0)),   # [H, W] -> [1, H, W]
    transforms.Resize((64, 64)),
    transforms.Normalize(mean=[-30.0], std=[15.0])
])

#local path on pc
train_path = "/content/unzipped/mixedtrain_fan"
test_path = "/content/unzipped/test_fan - Copy"

train_data = AudioDataset(train_path, transform)
test_data = AudioDataset(test_path, transform)
