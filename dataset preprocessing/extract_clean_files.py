import os
import shutil


root_dir = r'D:\licenta\Python\semestrul2\dataset\clean\test-clean'
target_dir = r'D:\licenta\Python\semestrul2\dataset\clean'

os.makedirs(target_dir, exist_ok=True)

for subdir, dirs, files in os.walk(root_dir):
    for file in files:
        if file.endswith('.flac'):
            file_path = os.path.join(subdir, file)
            target_path = os.path.join(target_dir, file)

            if not os.path.exists(target_path):
                try:
                    shutil.copy(file_path, target_path)  # Copiază fișierul în directorul țintă
                    print(f"Copiat: {file_path} -> {target_path}")
                except Exception as e:
                    print(f"Eroare la copierea fișierului: {file_path}. Eroare: {e}")
            else:
                print(f"Fișierul există deja: {target_path}")

print("Procesul de copiere a fost finalizat.")

