import librosa
import soundfile as sf
import os
import re
import numpy as np


def adjust_noise_level(clean_signal, noise_signal, desired_snr_db):
    # Calculează puterea semnalului clean
    power_clean = np.mean(clean_signal ** 2)

    # Calculează puterea necesară pentru zgomot bazată pe SNR-ul dorit
    desired_snr = 10 ** (desired_snr_db / 10.0)
    desired_noise_power = power_clean / desired_snr

    # Calculează puterea actuală a zgomotului
    power_noise = np.mean(noise_signal ** 2)

    # Calculează factorul de scalare pentru a ajunge la puterea dorită a zgomotului
    scale_factor = np.sqrt(desired_noise_power / power_noise)

    # Scalează semnalul de zgomot
    adjusted_noise_signal = noise_signal * scale_factor
    return adjusted_noise_signal


def extract_number(filename):
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else 0


def extract_number_from_clean(filename):
    match = re.search(r'(\d+)\D*$', filename)
    return int(match.group(1)) if match else 0


def extract_number_from_noise(filename):
    match = re.search(r'^(\d+)', filename)
    return int(match.group(1)) if match else 0


# Căile către directoarele cu semnale clean și zgomot, și directorul de output
clean_directory = r'D:\licenta\Licenta\dataset\clean'
noise_directory = r'D:\licenta\Licenta\dataset\noise'
output_directory = r'D:\licenta\Licenta\dataset\mixed'

# Crează directorul de output dacă nu există
os.makedirs(output_directory, exist_ok=True)

sr = 16000  # Rata de eșantionare
desired_snr_db = 5  # SNR-ul dorit în decibeli

# Obține și sortează listele de fișiere din directoarele clean și noise
clean_files = sorted(os.listdir(clean_directory), key=extract_number_from_clean)
noise_files = sorted(os.listdir(noise_directory), key=extract_number_from_noise)

if len(clean_files) != len(noise_files):
    raise ValueError("Numărul de fișiere clean nu corespunde cu numărul de fișiere noise")

# Parcurge fiecare pereche de fișiere clean și noise
for i, (filename_clean, filename_noise) in enumerate(zip(clean_files, noise_files), start=1):
    clean_path = os.path.join(clean_directory, filename_clean)
    noise_path = os.path.join(noise_directory, filename_noise)

    # Încarcă semnalele
    clean_signal, _ = librosa.load(clean_path, sr=sr)
    print(filename_noise)
    noise_signal, _ = librosa.load(noise_path, sr=sr)
    print(filename_clean)
    # Ajustează nivelul zgomotului pentru a obține SNR-ul dorit
    adjusted_noise_signal = adjust_noise_level(clean_signal, noise_signal, desired_snr_db)
    # Mixează semnalele
    mixed_signal = clean_signal + adjusted_noise_signal

    # Salvează semnalul mixat
    output_path = os.path.join(output_directory, f'mixed{i}_snr{desired_snr_db}.wav')
    sf.write(output_path, mixed_signal, sr)

print("Mixarea semnalelor clean cu zgomotul a fost finalizată.")
