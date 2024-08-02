import librosa
import numpy as np
import soundfile as sf
import os

def remove_silence(audio_path, output_path, top_db=30):
    # Încarcă fișierul audio
    y, sr = librosa.load(audio_path, sr=None)
    # Detectează părțile non-tăcute
    non_silent_intervals = librosa.effects.split(y, top_db=top_db)
    # Concatenează segmentele non-tăcute
    y_non_silent = np.concatenate([y[start:end] for start, end in non_silent_intervals])
    # Salvează fișierul audio procesat
    sf.write(output_path, y_non_silent, sr)

# Calea către directorul cu fișierele audio
directory = r'D:\licenta\Python\semestrul2\dataset\noise'

# Parcurge fiecare fișier din director
for filename in os.listdir(directory):
    if filename.endswith('.wav') or filename.endswith('.mp3'):  # Verifică extensia fișierului
        audio_path = os.path.join(directory, filename)
        output_path = audio_path  # Suprascrie fișierul original
        remove_silence(audio_path, output_path)

print("Toate fișierele au fost procesate.")
