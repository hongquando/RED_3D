#!/bin/sh
"""
Denoise all the sequences existent in a given folder using FastDVDnet.

@author: Matias Tassano <mtassano@parisdescartes.fr>
"""
import os
import argparse
import time
import cv2
import torch
import torch.nn as nn
import glob
import scipy.io
import numpy as np
from fastdvdnet import denoise_seq_fastdvdnet
from models import FastDVDnet
from utils_fastdvdnet import remove_dataparallel_wrapper

# from RED_3D_Osirim_v3.fastdvdnet import denoise_seq_fastdvdnet
# from RED_3D_Osirim_v3.models import FastDVDnet
# from RED_3D_Osirim_v3.utils_fastdvdnet import remove_dataparallel_wrapper

NUM_IN_FR_EXT = 5  # temporal size of patch
MC_ALGO = 'DeepFlow'  # motion estimation algorithm

def test_fastdvdnet(model_file, x_est, noise_sigma, ):
    """Denoises all sequences present in a given folder. Sequences must be stored as numbered
    image sequences. The different sequences must be stored in subfolders under the "test_path" folder.

    Inputs:
            args (dict) fields:
                    "model_file": path to model
                    "test_path": path to sequence to denoise
                    "suffix": suffix to add to output name
                    "max_num_fr_per_seq": max number of frames to load per sequence
                    "noise_sigma": noise level used on test set
                    "dont_save_results: if True, don't save output images
                    "no_gpu": if True, run model on CPU
                    "save_path": where to save outputs as png
                    "gray": if True, perform denoising of grayscale images instead of RGB
    """
    # Start time
    start_time = time.time()

    # Sets data type according to CPU or GPU modes
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Create models
    print('Loading models ...')
    model_temp = FastDVDnet(num_input_frames=NUM_IN_FR_EXT)

    # Load saved weights
    state_temp_dict = torch.load(model_file)
    if torch.cuda.is_available():
        device_ids = [0]
        model_temp = nn.DataParallel(model_temp, device_ids=device_ids).cuda()
    else:
        # CPU mode: remove the DataParallel wrapper
        state_temp_dict = remove_dataparallel_wrapper(state_temp_dict)
    model_temp.load_state_dict(state_temp_dict)

    # Sets the model in evaluation mode (e.g. it removes BN)
    model_temp.eval()

    with torch.no_grad():
        seq = np.reshape(x_est, (x_est.shape[0], -1, x_est.shape[1], x_est.shape[2]))
        if len(seq.shape) == 3:
            seq = np.expand_dims(seq, 1)
        seq = torch.from_numpy(seq).to(device)

    seqn = seq
    noisestd = torch.FloatTensor([noise_sigma]).to(device)

    denframes = denoise_seq_fastdvdnet(seq=seqn, \
                                       noise_std=noisestd, \
                                       temp_psz=NUM_IN_FR_EXT, \
                                       model_temporal=model_temp)

    denframes = denframes.cpu().numpy()
    f_x_est = np.reshape(denframes, (denframes.shape[0], denframes.shape[2], denframes.shape[3], 3))

    return f_x_est