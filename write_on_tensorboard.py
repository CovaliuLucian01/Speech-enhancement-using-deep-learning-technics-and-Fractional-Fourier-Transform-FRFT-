"""
For observing the results using tensorboard

1. wav
2. spectrogram
3. loss
"""
# tensorboard --logdir=logs
from tensorboardX import SummaryWriter
import numpy as np
import matplotlib.pylab as plt


def fig2np(fig):
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data


def plot_spectrogram_to_numpy(input_wav, fs, n_fft, n_overlap, mode, clim, label):
    # cuda to cpu
    # if isinstance(input_wav, torch.Tensor):
    #     input_wav = input_wav.cpu().detach().numpy()
    fig, ax = plt.subplots(figsize=(12, 3))

    pxx, freq, t, cax = plt.specgram(input_wav, NFFT=n_fft, Fs=fs, window=np.hamming(n_fft), noverlap=n_overlap,
                                     cmap='jet', mode=mode)
    if clim is not None:
        cax.set_clim(*clim)

    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.tight_layout()
    plt.clim(clim)

    if label is None:
        fig.colorbar(cax)
    else:
        fig.colorbar(cax, label=label)

    fig.canvas.draw()
    data = fig2np(fig)
    plt.close()
    return data


def plot_irm_as_numpy(irm, fs, hop_length, clim=None):
    """
    Plot an Ideal Ratio Mask (IRM) as a spectrogram and convert it to a numpy array.

    Parameters:
    - irm: Ideal Ratio Mask, a numpy array or tensor of shape (frequency bins, time frames)
    - fs: Sampling frequency
    - n_fft: FFT window size
    - hop_length: Hop length for STFT
    - clim: Color limits for the plot, as a tuple (min, max)

    Returns:
    - A numpy array representing the plotted IRM.
    """

    # Determine time and frequency axes.
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
    # Convert the figure to a numpy array and close the plot to free memory
    data = fig2np(fig)
    plt.close(fig)

    return data


class Writer(SummaryWriter):
    def __init__(self, logdir):
        super(Writer, self).__init__(logdir)

    def log_loss(self, train_loss, vali_loss, step):
        self.add_scalar('train_loss', train_loss, step)
        self.add_scalar('vali_loss', vali_loss, step)


    def log_score(self, vali_pesq, vali_stoi, vali_fwSNR, step):
        self.add_scalar('vali_pesq', vali_pesq, step)
        self.add_scalar('vali_stoi', vali_stoi, step)
        self.add_scalar('vali_fwSNR', vali_fwSNR, step)

    def log_score_test(self, vali_pesq, vali_stoi, vali_fwSNR):
        self.add_scalar('vali_pesq', vali_pesq)
        self.add_scalar('vali_stoi', vali_stoi)
        self.add_scalar('vali_fwSNR', vali_fwSNR)

    def log_wav(self, mixed_wav, clean_wav, noise_wav, est_wav, step):
        # <Audio>
        self.add_audio('mixed_wav', mixed_wav, step, 16000)
        self.add_audio('clean_target_wav', clean_wav, step, 16000)
        self.add_audio('noise_wav', noise_wav, step, 16000)
        self.add_audio('estimated_wav', est_wav, step, 16000)


    def log_wav_test(self, mixed_wav, clean_wav, est_wav, batch_index):
        # <Audio>
        self.add_audio('mixed_wav', mixed_wav, global_step=batch_index, sample_rate=16000)
        self.add_audio('clean_target_wav', clean_wav, global_step=batch_index, sample_rate=16000)
        self.add_audio('estimated_wav', est_wav, global_step=batch_index, sample_rate=16000)


    def log_spectrogram(self, mixed_wav, clean_wav, noise_wav, est_wav,irm, irm_features, step):
        # power spectral density.
        self.add_image('spectograms/clean_spectrogram',
                       plot_spectrogram_to_numpy(clean_wav, fs=16000, n_fft=2048, n_overlap=1024, mode='psd',
                                                 clim=None, label='dB'), step, dataformats='HWC')

        self.add_image('spectograms/mixed_spectrogram',
                       plot_spectrogram_to_numpy(mixed_wav, fs=16000, n_fft=2048, n_overlap=1024, mode='psd',
                                                 clim=None, label='dB'), step, dataformats='HWC')

        self.add_image('spectograms/noise_spectrogram',
                       plot_spectrogram_to_numpy(noise_wav, fs=16000, n_fft=2048, n_overlap=1024, mode='psd',
                                                 clim=None, label='dB'), step, dataformats='HWC')

        self.add_image('spectograms/estimated_spectrogram',
                       plot_spectrogram_to_numpy(est_wav, fs=16000, n_fft=2048, n_overlap=1024, mode='psd',
                                                 clim=None, label='dB'), step, dataformats='HWC')

        # magnitude spectrum
        self.add_image('magnitudes/clean_magnitude',
                       plot_spectrogram_to_numpy(clean_wav, fs=16000, n_fft=2048, n_overlap=1024, mode='magnitude',
                                                 clim=None, label='dB'), step, dataformats='HWC')

        self.add_image('magnitudes/mixed_magnitude',
                       plot_spectrogram_to_numpy(mixed_wav, fs=16000, n_fft=2048, n_overlap=1024, mode='magnitude',
                                                 clim=None, label='dB'), step, dataformats='HWC')

        self.add_image('magnitudes/noise_magnitude',
                       plot_spectrogram_to_numpy(noise_wav, fs=16000, n_fft=2048, n_overlap=1024, mode='magnitude',
                                                 clim=None, label='dB'), step, dataformats='HWC')

        self.add_image('magnitudes/estimated_magnitude',
                       plot_spectrogram_to_numpy(est_wav, fs=16000, n_fft=2048, n_overlap=1024, mode='magnitude',
                                                 clim=None, label='dB'), step, dataformats='HWC')

        # phase spectrum with unwrap

        self.add_image('unwrap_phases/clean_unwrap_phase',
                       plot_spectrogram_to_numpy(clean_wav, fs=16000, n_fft=2048, n_overlap=1024, mode='phase',
                                                 clim=None, label=None), step, dataformats='HWC')

        self.add_image('unwrap_phases/mixed_unwrap_phase',
                       plot_spectrogram_to_numpy(mixed_wav, fs=16000, n_fft=2048, n_overlap=1024, mode='phase',
                                                 clim=None, label=None), step, dataformats='HWC')

        self.add_image('unwrap_phases/noise_unwrap_phase',
                       plot_spectrogram_to_numpy(noise_wav, fs=16000, n_fft=2048, n_overlap=1024, mode='phase',
                                                 clim=None, label=None), step, dataformats='HWC')

        self.add_image('unwrap_phases/estimated_unwrap_phase',
                       plot_spectrogram_to_numpy(est_wav, fs=16000, n_fft=2048, n_overlap=1024, mode='phase',
                                                 clim=None, label=None), step, dataformats='HWC')


        # phase spectrum without unwrap

        self.add_image('phases/clean_phase',
                       plot_spectrogram_to_numpy(clean_wav, fs=16000, n_fft=2048, n_overlap=1024, mode='angle',
                                                 clim=None, label=None), step, dataformats='HWC')

        self.add_image('phases/mixed_phase',
                       plot_spectrogram_to_numpy(mixed_wav, fs=16000, n_fft=2048, n_overlap=1024, mode='angle',
                                                 clim=None, label=None), step, dataformats='HWC')

        self.add_image('phases/noise_phase',
                       plot_spectrogram_to_numpy(noise_wav, fs=16000, n_fft=2048, n_overlap=1024, mode='angle',
                                                 clim=None, label=None), step, dataformats='HWC')

        self.add_image('phases/estimated_phase',
                       plot_spectrogram_to_numpy(est_wav, fs=16000, n_fft=2048, n_overlap=1024, mode='angle',
                                                 clim=None, label=None), step, dataformats='HWC')



        self.add_image('IRM/IRM', plot_irm_as_numpy(irm, fs=16000, hop_length=1024, clim=None), step, dataformats='HWC')

        self.add_image('IRM/IRM_features', plot_irm_as_numpy(irm_features, fs=16000, hop_length=1024, clim=None), step, dataformats='HWC')
