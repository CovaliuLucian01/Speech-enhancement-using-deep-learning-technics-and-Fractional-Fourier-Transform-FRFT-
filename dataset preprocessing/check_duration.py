import librosa
from pathlib import Path

# Încarcă semnalul audio și rata de eșantionare
semnal, sr = librosa.load(Path(r'D:\licenta\Licenta\dataset\noise\1781_blue_noise.wav'))

# Calculează durata semnalului în secunde
durata = len(semnal) / sr

# Afișează durata semnalului
print(f"Durata semnalului audio este de {durata:.4f} secunde.")
