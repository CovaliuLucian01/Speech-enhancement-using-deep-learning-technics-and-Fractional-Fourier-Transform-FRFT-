import os
from pathlib import Path

# Setează directorul în care se află fișierele tale
dir_path = Path(r'C:\Users\coval\OneDrive\Desktop\presto')

# Setează indexul de start
start_index = 1614# sau orice alt număr de la care vrei să începi

# Lista toate fișierele din director și sortează-le în ordinea dorită
# Aici presupunem că vrei să le sortezi alfabetic după nume
files = list(dir_path.glob('presto*.wav'))

# Redenumește fișierele
for i, file_path in enumerate(files, start=start_index):
    # Construiește noul nume al fișierului
    new_name = f"{i}_presto.wav"  # Înlocuiește 'nume' cu orice alt prefix dorit
    new_file_path = dir_path.joinpath(new_name)

    # Redenumește fișierul
    os.rename(file_path, new_file_path)
