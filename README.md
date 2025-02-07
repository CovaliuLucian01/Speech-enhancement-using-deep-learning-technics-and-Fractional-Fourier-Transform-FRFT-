# Speech Enhancement using Deep Learning Techniques and Fractional Fourier Transform (FRFT)

This repository contains the implementation of my bachelor’s thesis, which focuses on enhancing vocal signal quality using deep learning techniques and Fractional Fourier Transform (FRFT). The project explores advanced signal processing methods combined with deep learning models to improve the clarity and integrity of vocal signals.

## Requirements
This repository is tested on Windows 11 and requires:
- Python 3.10
- CUDA 11.8
- PyTorch 2.2.1

### Development Environment
- PyCharm Community Edition 2023.3.5

## Getting Started

### 1. Install the necessary libraries
### 2. Make a dataset for train and validation
You will need three sets of signals:

- Clean signals
- Noise signals
- Noisy signals (clean signals with added noise)
  
The signals are sampled at 16 kHz and have a duration of 6 seconds. The naming convention used for the files is as follows:

Clean signals: clean*i* where *i* is a number from 1 to n (e.g., clean1, clean2, clean3, ...).  
Noise signals: *i*_Babble *i* is a number from 1 to n (e.g., 1_Babble, 2_Babble, 3_Babble, ...).  
Noisy signals: mixed*i*_snr5 where *i* is a number from 1 to n (e.g., mixed1_snr5, mixed2_snr5, mixed3_snr5, ...).  

Data Preparation Example:

To create the noisy signals dataset, you can use the noise_add+snr.py script to combine the clean signals with the noise signals, and then use the create_dataset.py script to compile the datasets into .npy format, ensuring that the files are named correctly to maintain order.
### 3. Features extraction

Use the feature_extraction.py script to calculate the FRFT characteristics from the noisy dataset for two separate algorithms. The script will save the extracted features directly into .npy files for training.

### 4. Configuration  
Configuration can be done from the network.py file. Below is an example configuration setup:
```bash
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
```
### 5. Train network
You can start training from the same script. The neural networks are defined in models.py.
### 6. TensorBoard
You can visualize the results throughout the training process using TensorBoard. Among other things, you can track the evolution of the loss functions for both training and validation, as well as the three quality metrics used: PESQ, STOI and fwSNR. 

To open TensorBoard, use the following command in your terminal:
```bash
tensorboard --logdir .\logs
```
![image](https://github.com/user-attachments/assets/0ef7b8ed-3e79-4733-ae54-6c95ad3f0ea8)
### 7. Testing
To test a trained model, change the stage variable in the network.py file to test. When testing a trained model, ensure that the network settings match those used during training to avoid errors.
### **8. References**

1. **Inspiration and starting point for the code:**
   - [DNN-based Speech Enhancement in the Frequency Domain](https://github.com/seorim0/DNN-based-Speech-Enhancement-in-the-frequency-domain)

2. **Method used in this work:**
   - Liyun Xu, Tong Zhang, "Fractional feature-based speech enhancement with deep neural network," *Speech Communication*, vol. 153, 2023. Available at: [Elsevier](https://doi.org/10.1016/j.specom.2023.102971).

3. **FRFT Module:**
   - [torch-frft](https://pypi.org/project/torch-frft/)

4. **Datasets used:**
   - [LibriSpeech ASR corpus](https://www.openslr.org/12)
   - [Signal Processing Information Base (SPIB)](http://spib.linse.ufsc.br/noise.html)
   - [DEMAND: a collection of multi-channel recordings of acoustic noise in diverse environments](https://zenodo.org/records/1227121)

