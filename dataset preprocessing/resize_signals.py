import librosa
import numpy as np
import soundfile as sf
import os

# Calea către directorul cu fișierele audio
mode = 2
clean_directory = r'D:\licenta\Python\semestrul2\dataset\clean'
noise_directory = r'C:\Users\coval\OneDrive\Desktop\Noise_test'
# Calea către directorul unde vor fi salvate fișierele ajustate
output_directory = r'C:\Users\coval\OneDrive\Desktop\Noise_test'
os.makedirs(output_directory, exist_ok=True)

target_duration = 6.016  # durata țintă în secunde
sr = 16000  # rata de eșantionare

if mode == 1:
    # Parcurge fiecare fișier din director
    for filename in os.listdir(clean_directory):
        if filename.endswith('.wav') or filename.endswith('.flac'):  # Verifică extensia fișierului
            audio_path = os.path.join(clean_directory, filename)
            output_path = os.path.join(output_directory, filename)

            # Încarcă fișierul audio
            y, sr = librosa.load(audio_path, sr=sr)

            # Calculează numărul de eșantioane necesar pentru durata țintă
            target_samples = int(target_duration * sr)

            # Verifică dacă fișierul are mai multe eșantioane decât ținta și taie excesul
            if len(y) > target_samples:
                y = y[:target_samples]
            # Dacă fișierul are mai puține eșantioane, adaugă padding cu zero
            elif len(y) < target_samples:
                padding = target_samples - len(y)  # Calculează cât padding este necesar
                y = np.pad(y, (0, padding), 'constant')

            # Salvează fișierul ajustat
            sf.write(output_path, y, int(sr))
else:
    # Parcurge fiecare fișier din director
    for filename in os.listdir(noise_directory):
        if filename.endswith('.wav') or filename.endswith('.flac'):  # Verifică extensia fișierului
            audio_path = os.path.join(noise_directory, filename)
            output_path = os.path.join(output_directory, filename)

            # Încarcă fișierul audio
            y, sr = librosa.load(audio_path, sr=sr)

            # Calculează numărul de eșantioane necesar pentru durata țintă
            target_samples = int(target_duration * sr)

            # Dacă fișierul are mai puține eșantioane, adaugă padding cu zero
            if len(y) < target_samples:
                # Calculează de câte ori trebuie să repetăm zgomotul
                repeat_times = np.ceil(target_samples / len(y))
                y = np.tile(y, int(repeat_times))
            # Taie excesul de zgomot repetat pentru a se potrivi exact cu durata țintă
            y = y[:target_samples]
            # Salvează fișierul ajustat
            sf.write(output_path, y, int(sr))
            print(filename)
print("Ajustarea fișierelor audio s-a finalizat.")
