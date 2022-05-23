
import re
import sys
import unittest
import importlib
from pathlib import Path

from Proj_287630_282604_288453.Miniproject_2.model import *

import torch
import torch.nn.functional as F

# Import tqdm if installed
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x

verbose=True

def compute_psnr(x, y, max_range=1.0):
        assert x.shape == y.shape and x.ndim == 4
        return 20 * torch.log10(torch.tensor(max_range)) - 10 * torch.log10(((x-y) ** 2).mean((1,2,3))).mean()

if __name__ == '__main__':

    model = Model()
    train_path="train_data.pkl"
    val_path = "val_data.pkl"
    train_input0, train_input1 = torch.load(train_path)
    val_input, val_target = torch.load(val_path)

    train_input0 = train_input0.float() / 255.0
    train_input1 = train_input1.float() / 255.0
    val_input = val_input.float() / 255.0
    val_target = val_target.float() / 255.0

    output_psnr_before = compute_psnr(val_input, val_target)
    print(f"[PSNR before: {output_psnr_before:.2f} dB]")

    model.train(train_input0, train_input1, verbose)

    mini_batch_size = 100
    model_outputs = model.predict(val_input)

    output_psnr_after = compute_psnr(model_outputs, val_target)
    
    print(f"[PSNR: {output_psnr_after:.2f} dB]")
