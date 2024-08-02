import os

# Calea către directorul cu fișierele tale audio
directory = r'D:\licenta\Python\semestrul2\dataset\clean'

# Obține o listă cu toate fișierele din director
files = os.listdir(directory)


# Parcurge fiecare fișier din director
for i, filename in enumerate(files, start=1):
    # Construiește calea completă a fișierului actual
    if filename.endswith('.wav') or filename.endswith('.flac'):
        old_file = os.path.join(directory, filename)


        # Construiește noua denumire a fișierului
        new_filename = f"clean{i}.wav"
        new_file = os.path.join(directory, new_filename)

        # Redenumește fișierul
        os.rename(old_file, new_file)
        print(f"{filename} -> {new_filename}")

# Afișează un mesaj de finalizare
print("Toate fișierele au fost redenumite cu succes!")
