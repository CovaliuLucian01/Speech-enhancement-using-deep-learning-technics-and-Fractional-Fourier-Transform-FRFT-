import numpy as np
import scipy.signal
import soundfile as sf

# Setează rata de eșantionare și durata semnalului
sr = 16000  # Rata de eșantionare
durata = 7  # Durata în secunde

# Generează zgomot alb
zgomot_alb = np.random.randn(sr * durata)

# Crează un filtru care să amplifice treptat frecvențele mai înalte
# Pentru blue noise, vom folosi un filtru trece-sus cu o frecvență de tăiere joasă și o panta ascendentă.
b, a = scipy.signal.butter(N=2, Wn=0.01, btype='high', analog=False)
zgomot_albastru = scipy.signal.lfilter(b, a, zgomot_alb)

# Normalizarea zgomotului albastru la intervalul [-1, 1]
zgomot_albastru = zgomot_albastru / np.max(np.abs(zgomot_albastru))

# Salvează zgomotul albastru într-un fișier
sf.write('blue_noise.wav', zgomot_albastru, sr)
