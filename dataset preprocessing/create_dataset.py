import librosa
import numpy as np
import os
import re
from pathlib import Path


def extract_number(file_name):
    # Extragerea numărului din numele fișierului
    numbers = re.findall(r'\d+', file_name)
    return int(numbers[0]) if numbers else 0


def incarca_si_preproceseaza_semnale(directory, sr=16000, max_signals=None):
    files = sorted(os.listdir(directory), key=extract_number)[:max_signals]
    signals = []
    for filename in files:  # Printează primele 10 fișiere pentru verificare
        print(filename)
    for filename in files:
        filepath = os.path.join(directory, filename)
        signal, _ = librosa.load(filepath, sr=sr)
        signals.append(signal)
    return np.array(signals)


train_ratio = 0.7
max_signals = 1900
snr = 5

# Căile către directoare
clean_directory = r'D:\licenta\Licenta\dataset\clean'
noise_directory = r'D:\licenta\Licenta\dataset\noise'
mixed_directory = r'D:\licenta\Licenta\dataset\mixed'
output_dir = Path(r"C:\Users\coval\OneDrive\Desktop")
# Încărcarea și preprocesarea semnalelor
clean_signals = incarca_si_preproceseaza_semnale(clean_directory, max_signals=max_signals)
noise_signals = incarca_si_preproceseaza_semnale(noise_directory, max_signals=max_signals)
mixed_signals = incarca_si_preproceseaza_semnale(mixed_directory, max_signals=max_signals)
# Salvarea semnalelor în fișiere .npy

num_train_files = int(len(clean_signals) * train_ratio)

clean_data_train = clean_signals[:num_train_files]
clean_data_validate = clean_signals[num_train_files:]
np.save(output_dir / f"clean_data_train_{snr}_{num_train_files}.npy", clean_data_train)
np.save(output_dir / f"clean_data_validate_{snr}_{max_signals-num_train_files}.npy", clean_data_validate)

noise_data_train = noise_signals[:num_train_files]
noise_data_validate = noise_signals[num_train_files:]
np.save(output_dir / f"noise_data_train_{snr}_{num_train_files}.npy", noise_data_train)
np.save(output_dir / f"noise_data_validate_{snr}_{max_signals-num_train_files}.npy", noise_data_validate)

mixed_data_train = mixed_signals[:num_train_files]
moxed_data_validate = mixed_signals[num_train_files:]
np.save(output_dir / f"mixed_data_train_{snr}_{num_train_files}.npy", mixed_data_train)
np.save(output_dir / f"mixed_data_validate_{snr}_{max_signals-num_train_files}.npy", moxed_data_validate)

print("Salvarea semnalelor clean și noise a fost finalizată.")
