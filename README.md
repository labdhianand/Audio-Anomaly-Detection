# Audio-Anomaly-Detection
This project implements an **anomaly detection system for audio signals**, using **Mel-spectrogram features** and a **Random Forest classifier**.  
It detects whether an input audio clip is *normal* or *anomalous* based on patterns in the spectrogram.


---

## Features
-> Converts audio files into Mel-spectrograms (via `torchaudio`)  
-> Flattens spectrograms into feature vectors for ML  
-> Trains a Random Forest classifier on labeled data  
-> Supports hyperparameter tuning with `GridSearchCV`  
-> Prints anomaly scores & model accuracy on test data  

---

## Tech Stack
- Python 3.x
- PyTorch
- torchaudio
- scikit-learn
- numpy



