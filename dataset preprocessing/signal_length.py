import librosa
import os

# Calea către directorul cu dataset-ul tău
dataset_path = r'D:\licenta\Python\semestrul2\dataset\clean'

# Inițializează variabila pentru durata maximă
max_duration = 0
min_duration = 40
count_files_under_5_sec = 0
file_with_max_duration = ""
file_with_min_duration = ""

# Parcurge toate fișierele din dataset
for subdir, dirs, files in os.walk(dataset_path):
    for file in files:
        # Completează aici cu extensiile dorite (e.g., '.wav', '.mp3')
        if file.endswith('.wav') or file.endswith('.mp3'):
            file_path = os.path.join(subdir, file)
            # Încarcă fișierul audio pentru a obține durata
            y, sr = librosa.load(file_path, sr=None)
            duration = librosa.get_duration(y=y, sr=sr)

            # Verifică dacă această durată este cea mai mare până acum
            if duration > max_duration:
                max_duration = duration
                file_with_max_duration = file_path
            if duration < min_duration:
                min_duration = duration
                file_with_min_duration = file_path
            if duration < 6:
                count_files_under_5_sec += 1
print(f"Cel mai lung fișier este: {file_with_max_duration} cu o durată de {max_duration} secunde.")
print(f"Cel mai scurt fișier este: {file_with_min_duration} cu o durată de {min_duration} secunde.")
print(f"Numărul de fișiere cu durata sub 5 secunde este: {count_files_under_5_sec}.")
