import os
import soundfile as sf

# Calea către directorul cu fișierele audio
directory = r'D:\licenta\Python\semestrul2\dataset\clean'

# Parcurge fiecare fișier din director
for filename in os.listdir(directory):
    if filename.endswith('.wav') or filename.endswith('.flac'):  # Verifică extensia fișierului
        audio_path = os.path.join(directory, filename)

        # Încarcă informații despre fișierul audio folosind soundfile
        with sf.SoundFile(audio_path) as sound_file:
            duration = len(sound_file) / sound_file.samplerate

        # Verifică dacă durata este mai mică de 1 secundă
        if duration < 3.74:
            print(f"Ștergerea fișierului: {filename} cu durata de {duration} secunde.")
            os.remove(audio_path)

print("Procesul de ștergere a fost finalizat.")
