import os
import time
import torch
import shutil
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from pathlib import Path
from write_on_tensorboard import Writer
from functions import calculate_total_params  # , adjust_learning_rate
from models import model_validate, train_model, test_model
# from torch.optim.lr_scheduler import LambdaLR


# Definirea setului de date
class SpeechDataset(Dataset):
    def __init__(self, speech_path, noise_path, mixed_path, frft_features_path=None):
        # Încarcă fișierele .npy folosind numpy.load
        self.speech_signals = np.load(speech_path, allow_pickle=True)
        self.noise_signals = np.load(noise_path, allow_pickle=True)
        self.mixed_signals = np.load(mixed_path, allow_pickle=True)
        self.frft_features = np.load(frft_features_path, allow_pickle=True)

    def __len__(self):
        # Lungimea datasetului va fi dată de numărul de exemple în oricare din seturi
        return len(self.speech_signals)

    def __getitem__(self, idx):
        # Extrage semnalele și caracteristicile pentru indexul specificat
        speech_signal = torch.from_numpy(self.speech_signals[idx]).float()
        noise_signal = torch.from_numpy(self.noise_signals[idx]).float()
        mixed_signal = torch.from_numpy(self.mixed_signals[idx]).float()
        frft_feature = torch.from_numpy(self.frft_features[idx]).float()

        return speech_signal, noise_signal, mixed_signal, frft_feature


# Definire arhitectura rețea neurale pentru îmbunătățirea vorbirii
# 5 straturi neuronale dintre care 3 ascunse
# dropout de 0.2 intre straturile ascunse
# functia de activare pentru input si straturile ascunse este relu
# functia de activare pentru output si este sigmoid


class LSTMEnhancementModel(nn.Module):
    def __init__(self, algorithm, hidden_dim=512, num_layers=3, output_dim=1024, dropout=0.3):
        super(LSTMEnhancementModel, self).__init__()
        if algorithm == 'alg1':
            input_dim = 14336
        else:
            input_dim = 22528
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.r_projection = nn.Linear(hidden_dim, hidden_dim)
        self.p_projection = nn.Linear(hidden_dim, output_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.tan = nn.Tanh()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        # print("lstm:", out)
        r_out = self.r_projection(out)
        p_out = self.p_projection(r_out)
        # print("p_out:", p_out)
        final_output = self.tan(p_out)
        # print("f_out:", final_output)
        return final_output


class BLSTMEnhancementModel(nn.Module):
    def __init__(self, algorithm, hidden_dim=512, num_layers=3, dropout=0.4):
        super(BLSTMEnhancementModel, self).__init__()
        if algorithm == 'alg1':
            input_dim = 14336
            output_dim = 1024
        else:
            input_dim = 22528
            output_dim = 1024

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.blstm = nn.LSTM(input_size=input_dim,
                             hidden_size=hidden_dim,
                             num_layers=num_layers,
                             batch_first=True,
                             bidirectional=True,
                             dropout=dropout)
        self.linear = nn.Linear(hidden_dim * 2,  output_dim)  # De 2 ori hidden_dim pentru că este bidirecțional
        self.tan = nn.Tanh()

    def forward(self, x):
        blstm_output, _ = self.blstm(x)
        linear_output = self.linear(blstm_output)
        final_output = self.tan(linear_output)
        # print("f_out:", final_output)
        return final_output


class SpeechEnhancementDNN(nn.Module):
    def __init__(self, algorithm):
        super(SpeechEnhancementDNN, self).__init__()
        # Calcul alg1v1 input_size = W(win_len)*(2 * M + 1) = 2048*7 = 14336
        # Calcul alg2 input_size = W(win_len)*1 = 2048*11= 22528
        if algorithm == 'alg1':
            self.input_layer = nn.Linear(in_features=14336, out_features=2048)
        else:
            self.input_layer = nn.Linear(in_features=22528, out_features=2048)

        self.hidden_layer1 = nn.Linear(in_features=2048, out_features=2048)
        self.hidden_layer2 = nn.Linear(in_features=2048, out_features=2048)
        self.hidden_layer3 = nn.Linear(in_features=2048, out_features=2048)
        self.output_layer = nn.Linear(in_features=2048, out_features=1024)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()
        self.tan = nn.Tanh()

    def forward(self, x):
        # x = self.relu(self.input_layer(x))
        x = self.input_layer(x)
        x = self.relu(self.hidden_layer1(x))
        x = self.dropout(x)
        x = self.relu(self.hidden_layer2(x))
        x = self.dropout(x)
        x = self.relu(self.hidden_layer3(x))
        x = self.sigmoid(self.output_layer(x))

        return x


###############################################################################
#         Parameter Initialization and Setting for model training             #
###############################################################################
def main():
    # Inițializarea și setarea parametrilor pentru antrenamentul modelului
    stage = ["train", "test"]
    max_sig = ["20", "1900"]
    snr_opt = ["3", "5"]
    devices = ['cuda', 'cpu']
    loss_options = ['Mask_based', 'Magnitude_based']
    algorithms = ['alg1', 'alg2']

    stage = stage[1]
    device = devices[0]
    algorithm = algorithms[1]
    loss_option = loss_options[0]
    max_signals = max_sig[1]
    snr = snr_opt[1]
    num_epochs = 200
    lr_rate = 0.001
    batch = 50
    train_ratio = 0.7

    # Definește parametrii STFT
    n_fft = 2048
    win_length = 2048
    hop_length = 1024
    stft_config = [n_fft, hop_length, win_length]

    # Initializare model
    models = ["FractionalFeaturesModel", "LSTM_Model", "BLSTM_Model"]
    model_selected = models[2]
    if model_selected == "FractionalFeaturesModel":
        model = SpeechEnhancementDNN(algorithm).to(device)
    elif model_selected == "BLSTM_Model":
        model = BLSTMEnhancementModel(algorithm).to(device)
    else:
        model = LSTMEnhancementModel(algorithm).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr_rate)
    # scheduler = LambdaLR(optimizer, lr_lambda=adjust_learning_rate)

    # scheduler = StepLR(optimizer, step_size=50, gamma=0.1)
    criterions = ["MSE", "MAE"]
    criterion_selected = criterions[1]
    if criterion_selected == "MSE":
        criterion = nn.MSELoss()
    else:
        criterion = nn.L1Loss()

    # Activează detectarea de anomalii
    torch.autograd.set_detect_anomaly(True)

    # Procentul de date folosit pentru antrenament

    train_signals = round(int(max_signals)*train_ratio)
    validate_signals = round(int(max_signals)-train_signals)

    director = Path(r"D:\licenta\Licenta\dataset")
    speech_path_train = Path(director / f"dataset{max_signals}" / f"clean_data_train_{snr}_{train_signals}.npy")
    mixed_path_train = Path(director / f"dataset{max_signals}" / f"mixed_data_train_{snr}_{train_signals}.npy")
    noise_path_train = Path(director / f"dataset{max_signals}" / f"noise_data_train_{snr}_{train_signals}.npy")
    frft_path_train = Path(director / f"dataset{max_signals}" /
                           f"frft_features_{algorithm}_train_{snr}_{train_signals}.npy") if algorithm != 'mag' else None

    speech_path_validate = Path(director / f"dataset{max_signals}" / f"clean_data_validate_{snr}_{validate_signals}.npy")
    mixed_path_validate = Path(director / f"dataset{max_signals}" / f"mixed_data_validate_{snr}_{validate_signals}.npy")
    noise_path_validate = Path(director / f"dataset{max_signals}" / f"noise_data_validate_{snr}_{validate_signals}.npy")
    frft_path_validate = Path(director / f"dataset{max_signals}" /
                              f"frft_features_{algorithm}_validate_{snr}_{validate_signals}.npy") if algorithm != 'mag' else None

    total_params = calculate_total_params(model)
    ###############################################################################
    #                        Confirm model information                            #
    ###############################################################################

    print('%d-%d-%d %d:%d:%d\n' %
          (time.localtime().tm_year, time.localtime().tm_mon,
           time.localtime().tm_mday, time.localtime().tm_hour,
           time.localtime().tm_min, time.localtime().tm_sec))
    print('total params   : %d (%.2f M, %.2f MBytes)\n' %
          (total_params,
           total_params / 1000000.0,
           total_params * 4.0 / 1000000.0))


###############################################################################
###############################################################################
#                             Main program start !!                           #
###############################################################################
###############################################################################

    if stage == "train":
        #######################################################################
        #                                 path                                #
        #######################################################################
        job_dir = './models/'

        logs_dir = './logs/'
        chkpt_model = None  # 'LSTM_Model_alg2_MAE_Mask_based_0.0001_0.7_6.26'  # 'FILE PATH (if you have pretrained model..)'
        chkpt = str("100")
        chkpt_path = None
        if chkpt_model is not None:
            chkpt_path = job_dir + str(chkpt_model) + '/chkpt_' + chkpt + '.pt'
        ###############################################################################
        #                        Set a log file to store progress.                    #
        #               Set a hps file to store hyper-parameters information.         #
        ###############################################################################

        # Verificare și încărcare checkpoint dacă există, altfel începe antrenamentul de la zero
        if chkpt_model is not None:  # Load the checkpoint
            print('Reincepere de la un checkpoint: %s' % chkpt_path)

            # Set a log file to store progress.
            dir_to_save = job_dir + str(chkpt_model)
            dir_to_logs = logs_dir + str(chkpt_model)

            checkpoint = torch.load(chkpt_path)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            epoch_start_idx = checkpoint['epoch'] + 1
            mse_vali_total = np.load(str(dir_to_save + '/mse_vali_total.npy'))
            # if the loaded length is shorter than I expected, extend the length
            if len(mse_vali_total) < num_epochs:
                plus = num_epochs - len(mse_vali_total)
                mse_vali_total = np.concatenate((mse_vali_total, np.zeros(plus)), 0)
        else:  # First learning
            print('Antrenare noua!')

            # make the file directory to save the models
            if not os.path.exists(job_dir):
                os.mkdir(job_dir)
            if not os.path.exists(logs_dir):
                os.mkdir(logs_dir)

            epoch_start_idx = 1
            mse_vali_total = np.zeros(num_epochs)

            # Set a log file to store progress.
            dir_to_save = os.path.join(job_dir + model_selected + '_' + algorithm + '_' + str(criterion_selected) + '_'
                                       + loss_option + '_' + str(lr_rate) + '_' + str(train_ratio)
                                       + '_%d.%d' % (time.localtime().tm_mon, time.localtime().tm_mday))

            dir_to_logs = os.path.join(logs_dir + model_selected + '_' + algorithm + '_' + str(criterion_selected) + '_'
                                       + loss_option + '_' + str(lr_rate) + '_' + str(train_ratio)
                                       + '_%d.%d' % (time.localtime().tm_mon, time.localtime().tm_mday))

        # make the file directory
        if not os.path.exists(dir_to_save):
            os.makedirs(dir_to_save)
            os.makedirs(dir_to_logs)

        # logging
        log_fname = str(dir_to_save + '/log.txt')
        if not os.path.exists(log_fname):
            fp = open(log_fname, 'w')
            # write_status_to_log_file(fp, total_params)
        else:
            fp = open(log_fname, 'a')

        # Inițializarea scriitorului pentru TensorBoard, unde vor fi logate progresul și rezultatele
        writer = Writer(dir_to_logs)
        estimator = model_validate
        # Crearea unei instanțe a clasei CustomDataset

        train_dataset = SpeechDataset(speech_path_train, noise_path_train, mixed_path_train, frft_path_train)
        validate_dataset = SpeechDataset(speech_path_validate, noise_path_validate, mixed_path_validate, frft_path_validate)

        # Crearea unui DataLoader pentru încărcarea datelor
        # data_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True, collate_fn=pad_collate)
        # validation_loader = DataLoader(validate_dataset, batch_size=batch, shuffle=True, collate_fn=pad_collate)
        data_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
        validation_loader = DataLoader(validate_dataset, batch_size=batch, shuffle=True)

        # Apelarea funcției pentru a verifica DataLoader-ul
        # check_data_loader(data_loader)
        # Inițializarea modelului
        for epoch in range(epoch_start_idx, num_epochs + 1):
            start_time = time.time()

            train_loss = train_model(model, data_loader, criterion, optimizer, device, loss_option, stft_config)

            # if epoch <= 50:
            #     scheduler.step()
            # Afișarea ratei de învățare după fiecare actualizare a parametrilor
            # print('Learning rate after epoch {} update: {}'.format(epoch, optimizer.param_groups[0]['lr']))

            # save checkpoint file to resume training
            save_path = str(dir_to_save + '/' + ('chkpt_%d.pt' % epoch))
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch
            }, save_path)
            # Validation
            vali_loss, vali_pesq, vali_stoi, vali_fwSNR = estimator(model, validation_loader, criterion, writer, dir_to_save, epoch, device, loss_option, stft_config)
            # write the loss on tensorboard
            # scheduler.step()
            writer.log_loss(train_loss, vali_loss, epoch)
            writer.log_score(vali_pesq, vali_stoi, vali_fwSNR, epoch)
            print('Epoch [{}] | Train_loss {:.6f} | Validation_loss {:.6} takes {:.2f} seconds\n'
                  .format(epoch, train_loss, vali_loss, time.time() - start_time))
            print('          | V PESQ: {:.6f} | STOI: {:.6f} | fwSNR: {:.6f} '.format(vali_pesq, vali_stoi, vali_fwSNR))
            # log file save
            fp.write('Epoch [{}] | Train_loss {:.6f} | Validation_loss {:.6} takes {:.2f} seconds\n'
                     .format(epoch, train_loss, vali_loss, time.time() - start_time))
            fp.write('          | V PESQ: {:.6f} | STOI: {:.6f} | fwSNR: {:.6f}\n'.format(vali_pesq, vali_stoi, vali_fwSNR))

            mse_vali_total[epoch - 1] = vali_loss
            np.save(str(dir_to_save + '/mse_vali_total.npy'), mse_vali_total)
            # Eliberează memoria GPU după fiecare epocă

        fp.close()
        print('Antrenare terminata!')

        # Copy optimum model that has minimum MSE.
        print('Salvare model optim...')
        min_index = np.argmin(mse_vali_total)
        print('Minimum validation loss is at ' + str(min_index + 1) + '.')
        src_file = str(dir_to_save + '/' + ('chkpt_%d.pt' % (min_index + 1)))
        tgt_file = str(dir_to_save + '/chkpt_opt.pt')
        shutil.copy(src_file, tgt_file)

    elif stage == "test":
        num_signals = 10
        speech_path_test = Path(r'D:\licenta\Licenta\dataset\test\var1\test_clean_' + str(num_signals) + ".npy")
        noise_path_test = Path(r'D:\licenta\Licenta\dataset\test\var1\test_noise_' + str(num_signals) + ".npy")
        mixed_path_test = Path(r'D:\licenta\Licenta\dataset\test\var1\test_mixed_' + str(num_signals) + ".npy")
        frft_path_train = Path(r'D:\licenta\Licenta\dataset\test\var1\frft_features_' + algorithm + '_test_' + str(num_signals) + ".npy")

        chkpt_modelt = 'BLSTM_Model_alg2_MAE_Mask_based_0.001_0.7_6.28'  # 'FILE PATH (if you have pretrained model..)'

        job_dir_traint = './models/'
        save_path = str('./tests/' + chkpt_modelt)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_path = str('./tests/' + chkpt_modelt)
        chkptt = str("200")
        chkpt_patht = job_dir_traint + str(chkpt_modelt) + '/chkpt_' + chkptt + '.pt'

        checkpointt = torch.load(chkpt_patht)
        # rint(checkpointt['model'].keys())
        model.load_state_dict(checkpointt['model'])
        optimizer.load_state_dict(checkpointt['optimizer'])

        test_dataset = SpeechDataset(speech_path_test, noise_path_test, mixed_path_test, frft_path_train)
        test_loader = DataLoader(test_dataset, batch_size=batch)
        estimator = test_model

        # Testing
        estimator(model, test_loader, device, stft_config, save_path, num_signals)

        print('Testare terminata!')


if __name__ == '__main__':
    main()
