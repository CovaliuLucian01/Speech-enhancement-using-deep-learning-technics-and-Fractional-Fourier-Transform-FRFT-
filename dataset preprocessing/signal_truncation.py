import librosa
import soundfile as sf
from pathlib import Path

# Încarcă semnalul audio
semnal, sr = librosa.load(Path(r'C:\Users\coval\OneDrive\Desktop\presto.wav'))

# Durata unui segment în secunde
durata_segment = 6.016

# Calculează numărul de eșantioane per segment
esantioane_per_segment = int(durata_segment * sr)

# Calculează numărul total de segmente
numar_segmente = int(len(semnal) / esantioane_per_segment)

# Crează o listă pentru a păstra segmentele
segmente = []

# Împarte semnalul în segmente
for i in range(numar_segmente):
    start = i * esantioane_per_segment
    end = start + esantioane_per_segment
    segment = semnal[start:end]
    segmente.append(segment)

# Dacă vrei să salvezi segmentele ca fișiere separate
for i, segment in enumerate(segmente):
    segment_path = Path(f"C:\\Users\\coval\\OneDrive\\Desktop\\presto\\presto{i}.wav")
    sf.write(segment_path, segment, int(sr))
