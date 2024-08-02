import time
import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch_frft.frft_module import frft
from pystoi import stoi
from pesq import pesq, cypesq
import tkinter as tk
from tkinter import filedialog
import librosa
from fwSNR import fwSNRseg
import torch.nn.functional as f
import matplotlib.pyplot as plt


def adjust_learning_rate(epoch):
    if epoch <= 70:
        return 1.0  # Nu modificăm rata de învățare pentru primele 70 de epoci
    elif 70 < epoch <= 100:
        return 0.5  # Reducem rata de învățare la jumătate după 70 de epoci
    else:
        return 0.1  # Reducem rata de învățare la 0.1 după 150 de epoci



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


def check_data_loader(data_loader):
    for i, (speech_signals, noise_signals) in enumerate(data_loader):
        print(f"Lotul {i+1}")
        print(f"Dimensiunea semnalului de vorbire: {speech_signals.size()}")
        print(f"Dimensiunea semnalului de zgomot: {noise_signals.size()}")
        # Oprește după primele 3 loturi pentru a evita o ieșire prea lungă
        if i == 2:
            break


def pad_to_even_length(signal, target_length):
    if signal.shape[0] < target_length:
        padding_size = target_length - signal.shape[0]
        padding = torch.zeros(padding_size, device=signal.device)
        signal_padded = torch.cat([signal, padding], dim=0)
    else:
        signal_padded = signal
    return signal_padded


def pad_to_segment(signal, segment_length):
    # Adaugă padding doar la semnale pentru a face fiecare divizibil cu segment_length.
    remainder = signal.size(0) % segment_length
    if remainder > 0:
        padding_size = segment_length - remainder
        signal = f.pad(signal, (0, padding_size), 'constant', 0)
    return signal


def custom_collate_fn(batch):
    (speech_signals, noise_signals) = zip(*batch)
    segment_length = 1024
    speech_signals_padded = [pad_to_segment(signal, segment_length) for signal in speech_signals]
    noise_signals_padded = [pad_to_segment(signal, segment_length) for signal in noise_signals]

    # Converteste listele de tensori într-un batch tensorial
    speech_signals_batch = torch.stack(speech_signals_padded)
    noise_signals_batch = torch.stack(noise_signals_padded)

    return speech_signals_batch, noise_signals_batch


def pad_collate(batch):
    (speech_signals, noise_signals) = zip(*batch)
    speech_signals_padded = pad_sequence(speech_signals, batch_first=True, padding_value=0)
    noise_signals_padded = pad_sequence(noise_signals, batch_first=True, padding_value=0)
    return speech_signals_padded, noise_signals_padded


def find_optimal_p(windowed_segment, k):
    optimal_p = 0
    minimal_l1_norm = float('inf')

    # Iterează prin ordinele posibile de la 0 la 1 cu pasul k
    for p in np.arange(0, 1.1, k):
        # Aplică FRFT pentru ordinea curentă p
        frft_result = frft(windowed_segment, p)  # Asigură-te că frft este implementată pentru a opera pe tensori PyTorch

        # Calculează norma L1 a diferenței
        l1_norm = torch.norm(frft_result, p=1)

        # Verifică dacă aceasta este cea mai mică normă L1 până acum
        if l1_norm < minimal_l1_norm:
            minimal_l1_norm = l1_norm
            optimal_p = p

    return optimal_p


def frft_features_alg1(signals, stft_config, device, num_frames):
    batch = signals.size(0)
    # Calculează STFT pe semnal
    # print('\nlast_frame_end:', signals.shape[-1])
    # print(signal.shape)  # noise_signals:torch.Size([batch_size, nr_samples])
    hop_length = stft_config[1]  # 1024
    win_length = stft_config[2]  # 2048
    hamming_window = torch.hamming_window(win_length).to(device).unsqueeze(0).unsqueeze(0)
    # print("hamming", hamming_window.shape)
    # Asigură-te că semnalele sunt zero-padded astfel încât ultimul frame să fie complet.
    # Calculează numărul necesar de zerouri pentru padding la sfârșitul semnalului.
    last_frame_end = (num_frames - 1) * hop_length + win_length
    # print('last_frame_end:', last_frame_end)
    padding_needed = max(0, last_frame_end - signals.shape[-1])
    # print("need", last_frame_end - signals.shape[-1])
    if padding_needed > 0:
        signals = torch.nn.functional.pad(signals, (0, padding_needed), "constant", 0)
    # print(signals.shape)
    # Utilizează `unfold` pentru a crea frame-uri din semnale.
    # signals: tensor de forma [batch_size, signal_length]
    # Returnează un tensor de forma [batch_size, num_frames, win_length]
    # print(signals.unfold(1, win_length, hop_length).shape)
    frames = signals.unfold(-1, win_length, hop_length).mul(hamming_window)
    # print(frames.shape)  # torch.Size([batch_size, N(num_frames), W(2048)])

    # frft_features_matrix = torch.zeros((batch, num_frames, win_length, 7), device=device)
    frft_features_matrix = torch.zeros((batch, num_frames, win_length), device=device)
    # print("frft_features_matrix:"+str(frft_features_matrix.shape))  # frft_features_matrix:torch.Size([batch_size, W(1024),  N(num_frames)])


    p = 0
    frft_result = frft(frames, p, dim=2)
    # print(frft_result.shape)
    frft_features_matrix[:, :, :] = torch.abs(frft_result)

    return frft_features_matrix


def frft_features_alg1_v1(signals, stft_config, device, num_frames, k, M):

    batch = signals.size(0)
    # Calculează STFT pe semnal
    # print(signal.shape)  # noise_signals:torch.Size([batch_size, nr_samples])
    hop_length = stft_config[1]  # 1024
    win_length = stft_config[2]  # 2048
    hamming_window = torch.hamming_window(win_length).to(device)

    frft_features_matrix = torch.zeros((batch, num_frames, win_length, 2 * M + 1), device=device)
    # print("frft_features_matrix:"+str(frft_features_matrix.shape))  # frft_features_matrix:torch.Size([batch_size, N(num_frames), W(1024), 2 * M + 1])
    # Pentru fiecare cadru n, găsește cel mai bun ordin p
    # batch = signals.size(0)
    # print(batch)
    starti = time.time()
    for batch_idx in range(batch):
        for n in range(num_frames):
            # start1 = time.time()

            start = n*hop_length
            end = start + win_length
            signal_segment = signals[batch_idx, start:end]
            # Verificăm dacă segmentul necesită padding
            if signal_segment.shape[0] < win_length:
                # Aplicăm padding pentru a ajunge la dimensiunea win_length
                padding_size = win_length - signal_segment.shape[0]
                # print("trebuie",padding_size)
                signal_segment = f.pad(signal_segment, pad=(0, padding_size), mode='constant', value=0)

            windowed_segment = signal_segment.mul(hamming_window)
            # print(windowed_segment.shape)  # windowed_segment:torch.Size([win_length])

            # print(windowed_segment.shape)  # signal_segment_padded:torch.Size([win_length])
            # end1 = time.time()
            # print(f"Timpul total pentru calcul winodows: {end1 - start1} secunde.")
            # start2 = time.time()
            # Găsește cel mai bun p folosind funcția separată

            optimal_p = find_optimal_p(windowed_segment, k)
            # end2 = time.time()

            # print(f"Timpul total pentru optimalp: {end2 - start2} secunde.")
            # print(optimal_p, l1_norm)
            # După ce găsește cel mai bun p, generează caracteristicile FRFT pentru cadrele învecinate
            # start3 = time.time()
            for i in range(-M, M + 1):
                if 0 <= n + i < num_frames:
                    frft_result = frft(windowed_segment, optimal_p)  # frft_result:torch.Size([1024])
                    # print(frft_result.shape, frft_result)
                    frft_features_matrix[batch_idx, n, :, i + M] = torch.abs(frft_result)
                    # i + M este indicele cadrului care este procesat
                    # : indică că toate punctele de eșantionare în cadru sunt selectate
                    # n este indicele cadrului curent în matricea de caracteristici
            # end3 = time.time()
            # print(f"Timpul total pentru frft_features_matrix: {end3 - start3} secunde.")
    end = time.time()
    print(f"Timpul total pentru frft_features_matrixtotal: {end - starti} secunde.")
    return frft_features_matrix


def frft_features_alg2(signals, stft_config, device, num_frames):
    batch = signals.size(0)
    # Calculează STFT pe semnal
    # print('\nlast_frame_end:', signals.shape[-1])
    # print(signal.shape)  # noise_signals:torch.Size([batch_size, nr_samples])
    hop_length = stft_config[1]  # 1024
    win_length = stft_config[2]  # 2048
    hamming_window = torch.hamming_window(win_length).to(device).unsqueeze(0).unsqueeze(0)
    # Asigură-te că semnalele sunt zero-padded astfel încât ultimul frame să fie complet.
    # Calculează numărul necesar de zerouri pentru padding la sfârșitul semnalului.
    last_frame_end = (num_frames - 1) * hop_length + win_length
    # print('last_frame_end:', last_frame_end)
    padding_needed = max(0, last_frame_end - signals.shape[-1])
    # print("need", last_frame_end - signals.shape[-1])
    if padding_needed > 0:
        signals = torch.nn.functional.pad(signals, (0, padding_needed), "constant", 0)
    # print(signals.shape)
    # Utilizează `unfold` pentru a crea frame-uri din semnale.
    # signals: tensor de forma [batch_size, signal_length]
    # Returnează un tensor de forma [batch_size, num_frames, win_length]
    frames = signals.unfold(-1, win_length, hop_length).mul(hamming_window)
    # print(frames.shape)  # torch.Size([batch_size, W(2048), N(num_frames)])

    p_values = torch.linspace(0, 1, steps=11, device=device)

    frft_features_matrix = torch.zeros((batch, num_frames, win_length, len(p_values)), device=device)
    # print("frft_features_matrix:"+str(frft_features_matrix.shape))  # frft_features_matrix:torch.Size([batch_size, N(num_frames), W(1024), 11])
    # Pentru fiecare cadru n, găsește cel mai bun ordin p
    # batch = signals.size(0)
    # print(batch)

    # Iterează prin ordinele posibile de la 0 la k
    for p_idx, p in enumerate(p_values):
        # print(p)
        frft_result = frft(frames, p)
        frft_features_matrix[:, :, :, p_idx] = torch.abs(frft_result)
    # print(frft_features_matrix.shape)
    return frft_features_matrix


def frft_features_alg2_v1(signals, stft_config, device, num_frames):

    batch = signals.size(0)
    # Calculează STFT pe semnal
    # print(signal.shape)  # noise_signals:torch.Size([batch_size, nr_samples])
    hop_length = stft_config[1]  # 1024
    win_length = stft_config[2]  # 2048
    hamming_window = torch.hamming_window(win_length).to(device)

    frft_features_matrix = torch.zeros((batch, num_frames, win_length, 11), device=device)
    # print("frft_features_matrix:"+str(frft_features_matrix.shape))  # frft_features_matrix:torch.Size([batch_size, N(num_frames), W(1024), 11])
    # Pentru fiecare cadru n, găsește cel mai bun ordin p
    # batch = signals.size(0)
    # print(batch)
    for batch_idx in range(batch):
        for n in range(num_frames):
            start = n * hop_length
            end = start + win_length
            signal_segment = signals[batch_idx, start:end]
            if signal_segment.size(0) < win_length:
                # print(signal_segment.shape)
                signal_segment = pad_to_even_length(signal_segment, win_length)
                # print("o fost nevoie", signal_segment.shape)
            # print(signal_segment.shape)
            windowed_segment = signal_segment * hamming_window
            # print(windowed_segment.shape)  # windowed_segment:torch.Size([win_length])

            # Iterează prin ordinele posibile de la 0 la k
            for p_idx, p in enumerate(np.arange(0, 1.1, 0.1)):
                # print(p)
                frft_result = frft(windowed_segment, p)
                frft_features_matrix[batch_idx, n, :, p_idx] = torch.abs(frft_result)

    return frft_features_matrix


def stft(signal, stft_config):
    # Aplică STFT folosind funcția torch.stft
    # signal - semnalul de intrare
    # n_fft - numărul de puncte folosit în FFT
    # hop_length - numărul de eșantioane între ferestre succesive
    # win_length - dimensiunea ferestrei aplicate semnalului
    n_fft = stft_config[0]
    hop_length = stft_config[1]
    win_length = stft_config[2]
    stft_matrix = torch.stft(signal, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                             window=torch.hamming_window(win_length).to(signal.device), return_complex=True)
    # print("stft_matrix", stft_matrix.shape)  # srft_matrix:torch.Size([batch_size, W(1024)+1, N(num_frames)])

    # real_part = stft_matrix.real
    # imag_part = stft_matrix.imag
    # combined_input = torch.cat([real_part, imag_part], dim=1)
    # print("combined_input:", combined_input.shape, "\nsignal:", stft_matrix.shape)
    _, _, num_frames = stft_matrix.shape  # Extrage numărul de cadre din forma tensorului
    # Returnează magnitudinea și faza STFT
    magnitude, phase = torch.abs(stft_matrix), torch.angle(stft_matrix)

    # phase_cos = torch.cos(torch.angle(stft_matrix))
    # phase_sin = torch.sin(torch.angle(stft_matrix))
    # combined_input2 = torch.cat([magnitude, phase_cos, phase_sin], dim=1)  # Concatenare de-a lungul dimensiunii canalului
    # print("combined_input2:", combined_input2.shape, "\nsignal:", stft_matrix.shape)
    return stft_matrix, magnitude, phase, num_frames


def istft(spectrum, stft_config):
    n_fft = stft_config[0]
    hop_length = stft_config[1]
    win_length = stft_config[2]
    signal = torch.istft(spectrum, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                         window=torch.hamming_window(win_length).to(spectrum.device))
    return signal


#######################################################################
#                           For validation                            #
#######################################################################
def zero_pad_signals(signal1, signal2):
    # Determină lungimea maximă dintre cele două semnale
    max_length = max(signal1.shape[-1], signal2.shape[-1])

    # Inițializează noi semnale cu lungimea maximă și aplică zero-padding unde este necesar
    padded_signal1 = np.zeros((signal1.shape[0], max_length))
    padded_signal2 = np.zeros((signal2.shape[0], max_length))

    # Copiază valorile originale în semnalele noi
    padded_signal1[:, :signal1.shape[-1]] = signal1
    padded_signal2[:, :signal2.shape[-1]] = signal2

    return padded_signal1, padded_signal2


def zero_pad_to_match_frames(signal, num_frames_expected, win_length, hop_length):
    # Calculează numărul total de puncte necesare pentru a avea num_frames_expected cadre
    total_points_needed = (num_frames_expected - 1) * hop_length + win_length
    current_length = signal.size(1)

    if current_length < total_points_needed:
        # Calculează cât zero-padding este necesar
        padding_size = total_points_needed - current_length
        # Adaugă zero-padding la semnal
        padded_signal = f.pad(signal, (0, padding_size), "constant", 0)
        return padded_signal
    else:
        # Dacă semnalul este deja suficient de mare, returnează-l așa cum este
        return signal


def cal_stoi(estimated_speechs, clean_speechs):
    stoi_scores = []
    # print(estimated_speechs.shape[0])
    for i in range(estimated_speechs.shape[0]):
        stoi_score = stoi(clean_speechs[i, :], estimated_speechs[i, :], 16000, extended=False)
        stoi_scores.append(stoi_score)
    return stoi_scores


def cal_fwSNRseg(clean_wavs, estimated_wavs):
    fwSNRseg_scores = []
    # print(clean_wavs.shape[0])
    for i in range(clean_wavs.shape[0]):
        fwSNRseg_score = fwSNRseg(clean_wavs[i, :], estimated_wavs[i, :], fs=16000)
        fwSNRseg_scores.append(fwSNRseg_score)
    return fwSNRseg_scores


def run_pesq_waveforms(dirty_wav, clean_wav):

    # Specificați frecvența de eșantionare (fs). Presupun că fs = 8000, puteți să-l înlocuiți cu valoarea corectă.
    fs = 16000  # se presupune o frecvență de eșantionare de 8 kHz, puteți să o modificați

    # Calculăm scorul PESQ

    return pesq(fs, clean_wav, dirty_wav, 'wb')


def cal_pesq(dirty_wavs, clean_wavs):
    pesq_scores = []
    for i in range(dirty_wavs.shape[0]):
        try:
            # Încercăm să calculăm scorul PESQ pentru perechea curentă de semnale
            score = run_pesq_waveforms(clean_wavs[i, :], dirty_wavs[i, :])
            pesq_scores.append(score)
        except cypesq.NoUtterancesError:
            # Dacă întâmpinăm eroarea NoUtterancesError, folosim o valoare prestabilită sau ultima valoare validă
            # Verificăm dacă avem scoruri PESQ calculate anterior
            if pesq_scores:
                last_valid_score = pesq_scores[-1]  # Utilizăm ultimul scor valid
            else:
                last_valid_score = 1.0  # Sau o valoare prestabilită dacă nu avem niciun scor valid anterior
            pesq_scores.append(last_valid_score)
            print(
                f'NoUtterancesError encountered. Using last valid PESQ score {last_valid_score} for the current pair.')
    return pesq_scores


def calculate_total_params(our_model):
    total_parameters = 0
    for variable in our_model.parameters():
        shape = variable.size()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim
        total_parameters += variable_parameters

    return total_parameters


def fig2np(fig):
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data


def salveaza_spectrograma(signal, fs, save_path, title):
    # Calculul STFT
    D = librosa.stft(signal, n_fft=2048, hop_length=512)
    # Calculul magnitudinii în decibeli
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_db, sr=fs, hop_length=512, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_irm_as_numpy(irm, fs, hop_length, clim=None, save_path='irm_image.png'):
    n_frames = irm.shape[1]
    time_axis = np.linspace(0, n_frames * hop_length / fs, n_frames)
    freq_axis = np.linspace(0, fs / 2, irm.shape[0])

    # Create figure
    fig, ax = plt.subplots()
    cax = ax.imshow(irm, aspect='auto', origin='lower', cmap='jet',
                    extent=[time_axis[0], time_axis[-1], freq_axis[0], freq_axis[-1]])

    if clim is not None:
        cax.set_clim(*clim)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    fig.colorbar(cax)

    fig.canvas.draw()
    data = fig2np(fig)  # Converteste figura in numpy array
    plt.imsave(save_path, data)  # Salvarea imaginii
    plt.close(fig)


# Write training related parameters into the log file.
def write_status_to_log_file(fp1, total_parameters):
    fp1.write('%d-%d-%d %d:%d:%d\n' % (time.localtime().tm_year,
                                       time.localtime().tm_mon, time.localtime().tm_mday,
                                       time.localtime().tm_hour, time.localtime().tm_min,
                                       time.localtime().tm_sec))
    fp1.write('total params   : %d (%.2f M, %.2f MBytes)\n' % (total_parameters,
                                                               total_parameters / 1000000.0,
                                                               total_parameters * 4.0 / 1000000.0))

# PROGRESS BAR


class Bar(object):
    def __init__(self, dataloader):
        # Verifică dacă dataloader are atributele necesare
        if not hasattr(dataloader, 'dataset'):
            raise ValueError('Attribute `dataset` not exists in dataloder.')
        if not hasattr(dataloader, 'batch_size'):
            raise ValueError('Attribute `batch_size` not exists in dataloder.')

        self.dataloader = dataloader  # Referință la DataLoader
        self.iterator = iter(dataloader)  # Creează un iterator pentru dataloader
        self.dataset = dataloader.dataset  # Referință la dataset-ul din DataLoader
        self.batch_size = dataloader.batch_size  # Dimensiunea batch-ului din DataLoader
        self._idx = 0  # Contor pentru numărul de batch-uri procesate
        self._batch_idx = 0  # Contor pentru numărul de exemple procesate
        self._time = []  # Lista pentru măsurarea timpului între batch-uri
        self._DISPLAY_LENGTH = 50  # Lungimea barei de progres

    def __len__(self):
        return len(self.dataloader)  # Returnează numărul de batch-uri în DataLoader

    def __iter__(self):
        return self  # Returnează obiectul Bar ca un iterator

    def __next__(self):
        # Adaugă timpul curent la lista _time pentru măsurarea duratei
        if len(self._time) < 2:
            self._time.append(time.time())

        # Actualizează contorul pentru numărul de exemple procesate
        self._batch_idx += self.batch_size
        if self._batch_idx > len(self.dataset):
            self._batch_idx = len(self.dataset)

        try:
            batch = next(self.iterator)  # Încearcă să obțină următorul batch
            self._display()  # Afișează bara de progres
        except StopIteration:
            raise StopIteration()

        # Actualizează contorul pentru numărul de batch-uri procesate și resetează dacă a ajuns la sfârșit
        self._idx += 1
        if self._idx >= len(self.dataloader):
            self._reset()

        return batch  # Returnează batch-ul curent

    def _display(self):
        # Calculează ETA (Estimated Time of Arrival) pentru finalizarea procesării
        if len(self._time) > 1:
            t = (self._time[-1] - self._time[-2])
            eta = t * (len(self.dataloader) - self._idx)
        else:
            eta = 0

        # Calculează rata de completare și afișează bara de progres
        rate = self._idx / len(self.dataloader)
        len_bar = int(rate * self._DISPLAY_LENGTH)
        bar = ('=' * len_bar + '>').ljust(self._DISPLAY_LENGTH, '.')
        idx = str(self._batch_idx).rjust(len(str(len(self.dataset))), ' ')

        tmpl = '\r{}/{}: [{}] - ETA {:.1f}s'.format(
            idx,
            len(self.dataset),
            bar,
            eta
        )
        print(tmpl, end='')
        if self._batch_idx == len(self.dataset):
            print()

    def _reset(self):
        # Resetează contorii și lista de timpuri pentru o nouă epocă de antrenament
        self._idx = 0
        self._batch_idx = 0
        self._time = []
