import os
### kss ###
dataset = "kss"
data_path = os.path.join("/home/minsu/dataset/", dataset)
meta_name = "transcript.v.1.3.txt"
textgrid_name = "TextGrid.zip"


# Text
text_cleaners = ['korean_cleaners']

# Audio and mel
### kss ###
sampling_rate = 22050
filter_length = 1024
hop_length = 256
win_length = 1024

### kss ###
max_wav_value = 32768.0
n_mel_channels = 80
mel_fmin = 20
mel_fmax = 7800

# FastSpeech 2
encoder_layer = 4
encoder_head = 2
encoder_hidden = 256
decoder_layer = 4
decoder_head = 2
decoder_hidden = 256
fft_conv1d_filter_size = 1024
fft_conv1d_kernel_size = (9, 1)
encoder_dropout = 0.2
decoder_dropout = 0.2

variance_predictor_filter_size = 256
variance_predictor_kernel_size = 3
variance_predictor_dropout = 0.5

max_seq_len = 1000


# Quantization for F0 and energy
### kss ###
f0_min = 71.0
f0_max = 768.2
energy_min = 0.0
energy_max = 0.008
n_bins = 256


# Checkpoints and synthesis path
preprocessed_path = os.path.join("./preprocessed/", dataset)
checkpoint_path = os.path.join("./ckpt/", dataset)
synth_path = os.path.join("./synth/", dataset)
eval_path = os.path.join("./eval/", dataset)
log_path = os.path.join("./log/", dataset)
test_path = "./results"
waveglow_path = "./waveglow/pretrained_model/waveglow_256channels.pt"


# Optimizer
batch_size = 16
epochs = 1000
n_warm_up_step = 4000
grad_clip_thresh = 1.0
acc_steps = 1

betas = (0.9, 0.98)
eps = 1e-9
weight_decay = 0.


# Vocoder
vocoder = 'waveglow' # 'waveglow' or 'melgan'

# Log-scaled duration
log_offset = 1.


# Save, log and synthesis
save_step = 10000
#save_step=1000
synth_step = 1000
eval_step = 1000
eval_size = 256
log_step = 1000
clear_Time = 20
