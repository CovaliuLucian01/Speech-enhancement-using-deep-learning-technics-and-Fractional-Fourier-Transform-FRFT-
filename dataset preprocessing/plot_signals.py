import matplotlib.pyplot as plt
from scipy.io import wavfile
from pathlib import Path
# Pentru fișiere non-WAV, folosește:
# import librosa
# import numpy as np

# Citirea primului fișier audio
rate1, data1 = wavfile.read(Path(r"D:\licenta\Licenta\tests\BLSTM_Model_alg2_MAE_Mask_based_0.001_0.7_6.28_200_set2\clean_wav_1.wav"))
# Pentru fișiere non-WAV, folosește:
# data1, rate1 = librosa.load('calea_catre_primul_fisier_audio', sr=None)

# Citirea celui de-al doilea fișier audio
# rate2, data2 = wavfile.read(Path(r"D:\licenta\Licenta\tests\FractionalFeaturesModel_alg1_Mask_based_0.001_0.7_4.10\estimated_wav_3.wav"))
# rate22, data22 = wavfile.read(Path(r"D:\licenta\Licenta\tests\LSTM_Model_alg1_Mask_based_0.0001_0.7_5.12\estimated_wav_3.wav"))
rate23, data23 = wavfile.read(Path(r"D:\licenta\Licenta\tests\BLSTM_Model_alg2_MAE_Mask_based_0.001_0.7_6.28_200_set2\estimated_wav_1.wav"))
# Pentru fișiere non-WAV, folosește:
# data2, rate2 = librosa.load('calea_catre_al_doilea_fisier_audio', sr=None)
rate3, data3 = wavfile.read(Path(r"D:\licenta\Licenta\tests\BLSTM_Model_alg2_MAE_Mask_based_0.001_0.7_6.28_200_set2\mixed_wav_1.wav"))
# Crearea figurii și a axelor
fig, axs = plt.subplots(3, 1, figsize=(10, 6))

# Plotarea primului semnal audio
axs[0].plot(data1)
axs[0].set_title('Semnal curat')

# Plotarea celui de-al doilea semnal audio
# axs[2].plot(data2)
# axs[2].set_title('Semnal estimat cu rețeaua Multistrat')

axs[1].plot(data3)
axs[1].set_title('Semnal curat + strănuturi pe fundal')

# axs[3].plot(data22)
# axs[3].set_title('Semnal estimat cu rețeaua LSTM')

axs[2].plot(data23)
axs[2].set_title('Semnal estimat cu rețeaua BLSTM')
# Afișarea ploturilor
plt.tight_layout()
plt.show()
