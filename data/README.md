# Dataset Information

The dataset used to train and test the model was taken from:  
🔗 [Anomaly Detection from Sound Data (Fan) – Kaggle](https://www.kaggle.com/datasets/vuppalaadithyasairam/anomaly-detection-from-sound-data-fan)

---

## Notes
- Only a **small subset of approximately 15–20 labeled audio files** was used for this project.
- The subset consisted of both **normal** and **anomalous** examples, manually selected from the dataset.
- This was sufficient for demonstrating the feasibility of the approach.
---

## Expected Format
This project expects the dataset to be prepared as a **PyTorch Dataset object** that yields the following per sample:
- `mel_spectrogram`: a `torch.Tensor` of shape `(1, 64, 64)` (Mel-scaled, converted to decibels)
- `label`: integer —  
  - `0` → Normal  
  - `1` → Anomalous

