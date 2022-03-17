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

NUM_IN_FR_EXT = 5  # temporal size of patch
MC_ALGO = 'DeepFlow'  # motion estimation algorithm

def test_fastdvdnet(model_temp, x_est, noise_sigma):
    """Denoises all sequences present in a given folder. Sequences must be stored as numbered
    image sequences. The different sequences must be stored in subfolders under the "test_path" folder.
    """
    # Sets data type according to CPU or GPU modes
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

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