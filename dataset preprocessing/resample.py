import librosa
from pathlib import Path
import soundfile as sf

# Încărcați fișierul audio (în acest exemplu, presupunem că este mono)
semnal, sr = librosa.load(Path(r'C:\Users\coval\OneDrive\Desktop\machinegun.wav'), sr=19980)  # asigurați-vă că 'sr' corespunde ratei de eșantionare inițiale
print(semnal)
# Resample la 16kHz
semnal_resampled = librosa.resample(semnal, orig_sr=sr, target_sr=16000)

# Salvarea semnalului resampled într-un nou fișier audio
sf.write(Path(r'C:\Users\coval\OneDrive\Desktop\machinegun_16.wav'), semnal_resampled, 16000)
