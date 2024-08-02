import tkinter as tk
from tkinter import filedialog
import librosa
import numpy as np
from pathlib import Path


def select_files(category):
    root = tk.Tk()
    root.withdraw()  # Ascundem fereastra principală tkinter
    tk.messagebox.showinfo("Select Files", f"Please select {category} files.")
    file_paths = filedialog.askopenfilenames()  # Acesta va deschide dialogul pentru a selecta fișierele
    return list(file_paths)  # Convertim la listă pentru a lucra mai ușor cu calea fiecărui fișier


def incarca_si_preproceseaza_fisiere(file_paths, sr=16000):
    signals = []
    for filepath in file_paths:
        signal, _ = librosa.load(filepath, sr=sr)
        signals.append(signal)
    return np.array(signals)


def salveaza_semnale(signals, output_dir, file_name):
    np.save(output_dir / file_name, signals)


# Setăm directorul de ieșire
output_dir = Path(r"D:\licenta\Licenta\dataset\test")

categories = ["clean", "mixed", "noise"]
for i in range(3):
    file_paths = select_files(categories[i])  # Selectăm fișierele cu mesaj specific pentru fiecare categorie
    signals = incarca_si_preproceseaza_fisiere(file_paths)  # Încărcăm și preprocesăm semnalele selectate
    salveaza_semnale(signals, output_dir, f"test_{categories[i]}_{len(signals)}.npy")

print("Salvarea semnalelor selectate a fost finalizată.")
