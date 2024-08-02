import torch
import numpy as np
from functii import (stft, istft, cal_pesq, cal_stoi, cal_fwSNRseg, zero_pad_signals, Bar, plot_irm_as_numpy, salveaza_spectrograma)
from pystoi import stoi
from pesq import pesq
from fwSNR import fwSNRseg
import soundfile as sf


def train_model(model, train_loader, criterion, optimizer, device, loss_option, stft_config):
    model.train()  # Setează modelul în modul de antrenare
    train_loss = 0.0
    batch_num = 0
    for speech_signals, noise_signals, mixed_signals, frft_features in Bar(train_loader):
        batch_num += 1
        speech_signals = speech_signals.to(device)  # speech_signals:torch.Size([batch_size, nr_samples])
        noise_signals = noise_signals.to(device)  # noise_signals:torch.Size([batch_size, nr_samples])
        mixed_signals = mixed_signals.to(device)  # mixed_signals:torch.Size([batch_size, nr_samples])
        frft_features = frft_features.to(device)
        # frft_features:torch.Size([batch_size, N(num_frames), 1024*(2 * M + 1)])

        speech_signals = torch.nn.functional.normalize(speech_signals, p=2.0, dim=1, eps=1e-12)
        noise_signals = torch.nn.functional.normalize(noise_signals, p=2.0, dim=1, eps=1e-12)
        mixed_signals = torch.nn.functional.normalize(mixed_signals, p=2.0, dim=1, eps=1e-12)

        # print("speech_signals:" + str(speech_signals.shape))
        # print("\nnoise_signals:" + str(noise_signals.shape))
        # print("\nmixed_signals:" + str(mixed_signals.shape))
        # print("\nfrft_features:" + str(frft_features.shape))
        _, speech_magnitude, _, _ = stft(speech_signals, stft_config)  # speech_magnitude:torch.Size([batch_size, W(1024)+1, N(num_frames)])
        _, noise_magnitude, _, _ = stft(noise_signals, stft_config)  # noise_magnitude:torch.Size([batch_size, W(1024)+1, N(num_frames)])
        _, mixed_magnitude, _, _ = stft(mixed_signals, stft_config)  # noise_magnitude:torch.Size([batch_size, W(1024)+1, N(num_frames)])


        # print("speech_magnitude:" + str(speech_magnitude.shape))
        # print("noise_magnitude:" + str(noise_magnitude.shape))
        # print("mixed_magnitude:" + str(mixed_magnitude.shape))
        noise_nyquist_freq = mixed_magnitude[:, -1:, :-1]
        # Calculează IRM
        irm = torch.sqrt(speech_magnitude[:, :, :-1] ** 2 / (speech_magnitude[:, :, :-1] ** 2 + noise_magnitude[:, :, :-1] ** 2))  # irm:torch.Size([batch_size, W(1024)+1, N(num_frames)])
        # print("irm:"+str(irm.shape))
        # Extrage frecvența Nyquist
        # irm:torch.Size([batch_size, W(1024), N(num_frames)])
        # print("irm new:" + str(irm.shape))
        outputs = model(frft_features)
        # print("output:"+str(output.shape))  #output: torch.Size([batch_size, N(num_frames), DNN_output_size = 1024])
        output_permuted = outputs.permute(0, 2, 1)  # Redimensionăm output pentru a se potrivi cu irm
        # print("output_permuted:" + str(output_permuted.shape)) # irm:torch.Size([batch_size, DNN_output_size = 1024, N(num_frames)])
        output_permuted = torch.cat((output_permuted, noise_nyquist_freq), dim=1)
        if loss_option == 'Mask_based':
            loss = criterion(output_permuted, irm)
        else:
            magnitude_est = output_permuted * mixed_magnitude[:, :, :-1]
            loss = criterion(magnitude_est, speech_magnitude[:, :, :-1])

        # print("loss:", loss.item())
        optimizer.zero_grad()
        loss.backward()
        # Cliparea gradientilor pentru a preveni explozia gradientilor
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += loss
    train_loss /= batch_num

    return train_loss


def model_validate(model, validation_loader, criterion, writer, dir_to_save, epoch, device, loss_option, stft_config):
    model.eval()  # Setam pe modul evaluare
    # Initializari
    validation_loss = 0
    batch_num = 0
    avg_pesq = 0
    avg_stoi = 0
    avg_fwSNR = 0
    f_score = open(dir_to_save + '/Epoch_' + '%d_SCORES' % epoch, 'a')

    with torch.no_grad():
        for speech_signals, noise_signals, mixed_signals, frft_features in Bar(validation_loader):
            batch_num += 1

            speech_signals = speech_signals.to(device)  # speech_signals:torch.Size([batch_size, nr_samples])
            noise_signals = noise_signals.to(device)  # noise_signals:torch.Size([batch_size, nr_samples])
            mixed_signals = mixed_signals.to(device)  # mixed_signals:torch.Size([batch_size, nr_samples])
            frft_features = frft_features.to(device)

            speech_signals = torch.nn.functional.normalize(speech_signals, p=2.0, dim=1, eps=1e-12)
            noise_signals = torch.nn.functional.normalize(noise_signals, p=2.0, dim=1, eps=1e-12)
            mixed_signals = torch.nn.functional.normalize(mixed_signals, p=2.0, dim=1, eps=1e-12)

            # speech_signals = torch.nn.functional.normalize(speech_signals, p=2.0, dim=1, eps=1e-12)
            # noise_signals = torch.nn.functional.normalize(noise_signals, p=2.0, dim=1, eps=1e-12)
            _, speech_magnitude, _, _ = stft(speech_signals, stft_config)  # speech_magnitude:torch.Size([batch_size, W(1024)+1, N(num_frames)])
            _, noise_magnitude, _, _ = stft(noise_signals, stft_config)  # noise_magnitude:torch.Size([batch_size, W(1024)+1, N(num_frames)])
            mixed_stft, mixed_magnitude, mixed_phase, num_frames = stft(mixed_signals, stft_config)  # noise_magnitude:torch.Size([batch_size, W(1024)+1, N(num_frames)])

            # Extrage frecvența Nyquist
            # speech_nyquist_freq = speech_magnitude[:, -1:, :]  # speech_nyquist_freq:torch.Size([batch_size, 1, N(num_frames)])
            noise_nyquist_freq = mixed_magnitude[:, -1:, :-1]  # noise_nyquist_freq:torch.Size([batch_size, 1, N(num_frames)])
            # Calculează IRM
            irm = torch.sqrt(speech_magnitude[:, :, :-1] ** 2 / (speech_magnitude[:, :, :-1] ** 2 + noise_magnitude[:, :, :-1] ** 2))  # irm:torch.Size([batch_size, W+1 = 1025, N(num_frames)])
            # Tăiem frecvență Nyquist din tensorul de magnitudine
            # irm = irm[:, :-1, :-1]  # irm:torch.Size([batch_size, W(win_len), N(num_frames)])

            outputs = model(frft_features)
            # print("output:"+str(outputs.shape))  #output: torch.Size([batch_size, N(num_frames), DNN_output_size = 1024])
            output_permuted = outputs.permute(0, 2, 1)  # Redimensionăm output pentru a se potrivi cu irm
            # print("output_permuted:" + str(output_permuted.shape)) # irm:torch.Size([batch_size, DNN_output_size = 1024, N(num_frames)])
            output_permuted = torch.cat((output_permuted, noise_nyquist_freq), dim=1)

            # print(output_permuted.shape, noise_magnitude.shape)
            # Adăugăm frecvența Nyquist presupunând că este similară cu ultima frecvență existentă
            # print(output_permuted.shape, noise_nyquist_freq.shape)
            # zero = torch.zeros((outputs.shape[0], 1, num_frames-1), device=device)
            # output_permuted_with_nyquist = torch.cat((output_permuted, zero), dim=1)
            # print(output_permuted_with_nyquist.shape)  # output_permuted_with_nyquist:torch.Size([batch_size, W(1024)+1, N(num_frames)])

            estimated_magnitude = output_permuted * mixed_magnitude[:, :, :-1]
            estimated_complex_spectrum = estimated_magnitude * torch.exp(1j * mixed_phase[:, :, :-1])
            # print(estimated_complex_spectrum.shape)  # estimated_complex_spectrum:torch.Size([batch_size, W(1024)+1, N(num_frames)])
            # estimated_complex_spectrum = torch.cat((estimated_complex_spectrum, noise_nyquist_freq), dim=1)
            estimated_signals = istft(estimated_complex_spectrum, stft_config)

            if loss_option == 'Mask_based':
                loss = criterion(output_permuted, irm)
            else:
                magnitude_est = output_permuted * mixed_magnitude[:, :, :-1]
                loss = criterion(magnitude_est, speech_magnitude[:, :, :-1])

            validation_loss += loss.item()
            # print(estimated_signals.shape, speech_signals.shape)  # torch.Size([batch_size, num_samples]) torch.Size([batch_size, num_samples])
            # Convert tensors to numpy arrays for metric calculations
            estimated_wavs = estimated_signals.cpu().detach().numpy().astype(np.float32)
            clean_wavs = speech_signals.cpu().detach().numpy().astype(np.float32)
            mixed_wavs = mixed_signals.cpu().detach().numpy().astype(np.float32)
            noise_wavs = noise_signals.cpu().detach().numpy().astype(np.float32)
            irm_mask = irm.cpu().detach().numpy().astype(np.float32)
            irm_features = output_permuted.cpu().detach().numpy().astype(np.float32)


            # print(estimated_wavs.shape, clean_wavs.shape)
            clean_wavs, estimated_wavs = zero_pad_signals(clean_wavs, estimated_wavs)
            # print(estimated_wavs.shape, clean_wavs.shape)
            stoif = cal_stoi(clean_wavs, estimated_wavs)
            pesqf = cal_pesq(clean_wavs, estimated_wavs)
            fwSNR = cal_fwSNRseg(clean_wavs, estimated_wavs)

            for i in range(len(pesqf)):
                f_score.write(f'PESQ {pesqf[i]:.6f} | STOI {stoif[i]:.6f} | fwSNR {fwSNR[i]:.6f}\n')

            avg_pesq += np.mean(pesqf)
            avg_stoi += np.mean(stoif)
            avg_fwSNR += np.mean(fwSNR)

        if epoch % 10 == 0:
            writer.log_wav(mixed_wavs[0], clean_wavs[0], noise_wavs[0], estimated_wavs[0], epoch)
            writer.log_spectrogram(mixed_wavs[0], clean_wavs[0], noise_wavs[0], estimated_wavs[0], irm_mask[0], irm_features[0], epoch)

        validation_loss /= batch_num
        avg_pesq /= batch_num
        avg_stoi /= batch_num
        avg_fwSNR /= batch_num

        # Optionally log to tensorboard or console
        print(f'Validation Loss: {validation_loss}, PESQ: {avg_pesq}, STOI: {avg_stoi}, fwSNR: {avg_fwSNR}')

    return validation_loss, avg_pesq, avg_stoi, avg_fwSNR


def test_model(model, test_loader, device, stft_config, save_path, num_signals):
    model.eval()  # Setam pe modul evaluare
    # Initializari

    for speech_signals, noise_signals, mixed_signals, frft_features in Bar(test_loader):
        speech_signals = speech_signals.to(device)  # speech_signals:torch.Size([batch_size, nr_samples])
        noise_signals = noise_signals.to(device)  # noise_signals:torch.Size([batch_size, nr_samples])
        mixed_signals = mixed_signals.to(device)  # mixed_signals:torch.Size([batch_size, nr_samples])
        frft_features = frft_features.to(device)  # mixed_signals:torch.Size([batch_size, nr_samples])
        # print(speech_signals.shape)
        speech_signals = torch.nn.functional.normalize(speech_signals, p=2.0, dim=1, eps=1e-12)
        noise_signals = torch.nn.functional.normalize(noise_signals, p=2.0, dim=1, eps=1e-12)
        mixed_signals = torch.nn.functional.normalize(mixed_signals, p=2.0, dim=1, eps=1e-12)

        batch = num_signals
        # print(batch)
        hop_length = stft_config[1]  # 1024
        # win_length = stft_config[2]  # 2048

        _, speech_magnitude, _, num_frames = stft(speech_signals, stft_config)  # speech_magnitude:torch.Size([batch_size, W(1024)+1, N(num_frames)])
        _, noise_magnitude, _, _ = stft(noise_signals, stft_config)  # noise_magnitude:torch.Size([batch_size, W(1024)+1, N(num_frames)])
        mixed_stft, mixed_magnitude, mixed_phase, _ = stft(mixed_signals, stft_config)  # noise_magnitude:torch.Size([batch_size, W(1024)+1, N(num_frames)])



        with torch.no_grad():
            # noise_nyquist_freq = mixed_magnitude[batch_idx, -1:, :-1]  # noise_nyquist_freq:torch.Size([batch_size, 1, N(num_frames)])
            # print("noise_nyquist_freq:" + str(noise_nyquist_freq.shape))
            # Calculează IRM
            noise_nyquist_freq = mixed_magnitude[:, -1:, :-1]
            irm = torch.sqrt(speech_magnitude[:, :, :-1] ** 2 / (
                        speech_magnitude[:, :, :-1] ** 2 + noise_magnitude[:, :, :-1] ** 2))  # irm:torch.Size([batch_size, W+1 = 1025, N(num_frames)])
            # Tăiem frecvență Nyquist din tensorul de magnitudine
            # print("irm:" + str(irm.shape))
            outputs = model(frft_features)
            print("output:"+str(outputs.shape))  # output: torch.Size([ N(num_frames), DNN_output_size = 1024])
            # print("output_permuted:" + str(output_permuted.shape)) # irm:torch.Size([DNN_output_size = 1024, N(num_frames)])
            output_permuted = outputs.permute(0, 2, 1)
            # print(output_permuted.shape, noise_magnitude.shape)
            # Adăugăm frecvența Nyquist presupunând că este similară cu ultima frecvență existentă
            # print(output_permuted.shape, noise_nyquist_freq.shape)

            output_permuted_with_nyquist = torch.cat((output_permuted, noise_nyquist_freq), dim=1)
            # print(mixed_magnitude[:, :, :-1].shape)  # output_permuted_with_nyquist:torch.Size([batch_size, W(1024)+1, N(num_frames)])

            estimated_magnitude = output_permuted_with_nyquist * mixed_magnitude[:, :, :-1]
            # print(estimated_magnitude.shape)
            estimated_complex_spectrum = estimated_magnitude * torch.exp(1j * mixed_phase[:, :, :-1])
            # print(estimated_complex_spectrum.shape)
            # zero = torch.zeros((outputs.shape[0], 1, 94), device=device)
            # estimated_complex_spectrum = torch.cat((estimated_complex_spectrum, zero), dim=1)
            # estimated_complex_spectrum:torch.Size([batch_size, W(1024)+1, N(num_frames)])

            # print(estimated_complex_spectrum.unsqueeze(0).shape, speech_signals.shape)
            estimated_signal = istft(estimated_complex_spectrum, stft_config)
            # print(estimated_signal.shape, speech_signals.shape)  # torch.Size([batch_size, num_samples]) torch.Size([batch_size, num_samples])
            for batch_idx in range(batch):
                # Convert tensors to numpy arrays for metric calculations
                estimated_wav = estimated_signal[batch_idx, :].cpu().detach().numpy().astype(np.float32)
                clean_wav = speech_signals[batch_idx, :].cpu().detach().numpy().astype(np.float32)
                mixed_wav = mixed_signals[batch_idx, :].cpu().detach().numpy().astype(np.float32)
                # print("est:", estimated_wav.shape)

                irm_mask = irm[batch_idx, :, :].cpu().detach().numpy().astype(np.float32)
                irm_features = output_permuted[batch_idx, :, :].cpu().detach().numpy().astype(np.float32)
                # print("irm:", irm_mask.shape)
                # print("irm f:", irm_features.shape)
                # print(estimated_wavs.shape, clean_wavs.shape)

                if len(clean_wav) > len(estimated_wav):
                    estimated_wav = np.pad(estimated_wav, (0, len(clean_wav) - len(estimated_wav)), 'constant', constant_values=0)
                elif len(estimated_wav) > len(clean_wav):
                    clean_wav = np.pad(clean_wav, (0, len(estimated_wav) - len(clean_wav)), 'constant', constant_values=0)

                # print(estimated_wavs.shape, clean_wavs.shape)
                stoi_score = stoi(clean_wav, estimated_wav, 16000, extended=False)
                pesq_score = pesq(16000, clean_wav, estimated_wav, 'wb')
                fwSNR_score = fwSNRseg(clean_wav, estimated_wav, fs=16000)

                sf.write(f'{save_path}/estimated_wav_{batch_idx}.wav', estimated_wav * 1.5, 16000)
                sf.write(f'{save_path}/clean_wav_{batch_idx}.wav', clean_wav, 16000)
                sf.write(f'{save_path}/mixed_wav_{batch_idx}.wav', mixed_wav, 16000)
                plot_irm_as_numpy(irm_mask, 16000, hop_length, save_path=f'{save_path}/irm{batch_idx}.png')
                plot_irm_as_numpy(irm_features, 16000, hop_length, save_path=f'{save_path}/irm_features{batch_idx}.png')
                salveaza_spectrograma(clean_wav, 16000, save_path=f'{save_path}/clean_spec{batch_idx}.png', title="Clean Signal Spectrogram")
                salveaza_spectrograma(estimated_wav, 16000, save_path=f'{save_path}/est_spec{batch_idx}.png', title="Estimated Signal Spectrogram")
                salveaza_spectrograma(mixed_wav, 16000, save_path=f'{save_path}/mixed_spec{batch_idx}.png', title="Mixed Signal Spectrogram")
                # Optionally log to tensorboard or console
                print(f'PESQ: {pesq_score}, STOI: {stoi_score}, fwSNR: {fwSNR_score}')
