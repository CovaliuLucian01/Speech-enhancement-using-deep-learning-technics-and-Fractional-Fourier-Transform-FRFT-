import torch
import numpy as np
from torch_frft.frft_module import frft
import time
import torch.nn.functional as f
from pathlib import Path


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


def frft_features(signals, stft_config, device, num_frames, k, M, alg):
    batch = signals.size(0)
    # Calculează STFT pe semnal
    # print(signal.shape)  # noise_signals:torch.Size([batch_size, nr_samples])
    hop_length = stft_config[1]  # 1024
    win_length = stft_config[2]  # 2048
    hamming_window = torch.hamming_window(win_length).to(device)
    if alg == "alg1":
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
                    # print(f"Lungimea așteptată: {win_length}, Lungimea reală: {signal_segment.shape[0]}")
                    # Aplicăm padding pentru a ajunge la dimensiunea win_length
                    padding_size = win_length - signal_segment.shape[0]
                    # print("trebuie", padding_size)
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
    else:

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
                if signal_segment.shape[0] < win_length:
                    # print(f"Lungimea așteptată: {win_length}, Lungimea reală: {signal_segment.shape[0]}")
                    # Aplicăm padding pentru a ajunge la dimensiunea win_length
                    padding_size = win_length - signal_segment.shape[0]
                    # print("trebuie", padding_size)
                    signal_segment = f.pad(signal_segment, pad=(0, padding_size), mode='constant', value=0)
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


def frft_features_batched(signals, stft_config, device, num_frames, k, M, alg, batch_size):
    num_signals = signals.shape[0]
    # Inițializează matricea de caracteristici FRFT completă ca listă
    frft_features_list = []
    print("s:", signals.shape)
    # Ciclu prin semnale în batch-uri
    for start in range(0, num_signals, batch_size):
        end = min(start + batch_size, num_signals)
        signal_batch = signals[start:end]
        print("b:", signal_batch.shape)
        signal_batch = torch.from_numpy(signal_batch).type(torch.float32).to(device)
        signal_batch = torch.nn.functional.normalize(signal_batch, p=2.0, dim=1, eps=1e-12)
        print("b:", signal_batch.shape)
        # Calculează caracteristicile FRFT pentru batch-ul curent
        frft_features_batch = frft_features(signal_batch, stft_config, device, num_frames, k, M, alg)

        # Adaugă rezultatele la lista de caracteristici
        frft_features_list.append(frft_features_batch.cpu())  # Transferă înapoi pe CPU pentru a economisi memoria GPU
        print("Batch procesat!")
    # Concatenează toate batch-urile într-un singur tensor
    frft_features_matrix = torch.cat(frft_features_list, dim=0)
    print("Frft bactch:", frft_features_matrix.shape)
    return frft_features_matrix


# Încărcarea semnalelor mixate
snr = 5
max_signals = 1900
num_train_files = 1330
algoritms = ["alg1", "alg2"]
alg = algoritms[1]
num_validate_files = max_signals - num_train_files

mode = ["train", "test"]
mode = mode[0]
if mode == "train":
    director = Path(r"D:\licenta\Licenta\dataset")
    # mixed_train_signals_path = Path(director / f"dataset{max_signals}/mixed_data_train_{snr}_{num_train_files}.npy")
    mixed_validate_signals_path = Path(director / f"dataset{max_signals}/mixed_data_validate_{snr}_{num_validate_files}.npy")


    # mixed_train_signals = np.load(mixed_train_signals_path)
    mixed_validate_signals = np.load(mixed_validate_signals_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convertirea în tensori PyTorch și asigurarea tipului float32
    # mixed_train_signals_tensor = torch.from_numpy(mixed_train_signals).type(torch.float32).to(device)
    # mixed_validate_signals_tensor = torch.from_numpy(mixed_validate_signals).type(torch.float32).to(device)

    # Definește parametrii STFT
    n_fft = 2048
    win_length = 2048
    hop_length = 1024
    stft_config = [n_fft, hop_length, win_length]
    # print("Fisiere antrenare:", mixed_train_signals_tensor.shape)
    # print("Fisiere validare:", mixed_validate_signals_tensor.shape)
    num_frames = 94  # Numărul de cadre pentru care să extragi caracteristicile
    k = 0.1  # pas
    M = 3
    batch_size = 100

    # Extragerea caracteristicilor
    # frft_features_train = frft_features_batched(mixed_train_signals, stft_config, device, num_frames, k, M, alg, batch_size)
    # print("Frft antrenare:", frft_features_train.shape)
    # frft_features_train = frft_features_train.view(frft_features_train.size(0), frft_features_train.size(1), -1)
    # Salvarea caracteristicilor extrase
    # print("Frft antrenare:", frft_features_train.shape)

    # np.save(director / f"dataset{max_signals}" / f"frft_features_{alg}_train_{snr}_{num_train_files}.npy", frft_features_train.numpy())

    frft_features_validate = frft_features_batched(mixed_validate_signals, stft_config, device, num_frames, k, M, alg, batch_size)
    print("Frft validare:", frft_features_validate.shape)
    frft_features_validate = frft_features_validate.view(frft_features_validate.size(0), frft_features_validate.size(1), -1)
    # Salvarea caracteristicilor extrase
    print("Frft validare:", frft_features_validate.shape)

    np.save(director / f"dataset{max_signals}" / f"frft_features_{alg}_validate_{snr}_{num_validate_files}.npy", frft_features_validate.numpy())
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_fft = 2048
    win_length = 2048
    hop_length = 1024
    stft_config = [n_fft, hop_length, win_length]
    num_test = 10
    num_frames = 94
    k = 0.1  # pas
    M = 3
    batch_size = 100

    director = Path(r"D:\licenta\Licenta\dataset")
    mixed_test_signals_path = Path(director / f"test/var2/test_mixed_{num_test}.npy")

    mixed_test_signals = np.load(mixed_test_signals_path)
    # Extragerea caracteristicilor
    frft_features_test = frft_features_batched(mixed_test_signals, stft_config, device, num_frames, k, M, alg, batch_size)
    print("Frft antrenare:", frft_features_test.shape)
    frft_features_test = frft_features_test.view(frft_features_test.size(0), frft_features_test.size(1), -1)
    # Salvarea caracteristicilor extrase
    print("Frft antrenare:", frft_features_test.shape)
    np.save(director / f"test" / f"frft_features_{alg}_test_{num_test}.npy",
            frft_features_test.numpy())
print("Extracția și salvarea caracteristicilor au fost finalizate.")
