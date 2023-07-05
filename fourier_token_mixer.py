import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def real_imaginary_relu(z):
    return F.relu(z.real) + 1.j * F.relu(z.imag)

def phase_amplitude_relu(z):
    return F.relu(torch.abs(z)) * torch.exp(1.j * torch.angle(z))

class mod_relu(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.b = nn.Parameter(torch.tensor(0.))
        self.b.requires_grad = True

    def forward(self, z):
        return F.relu(torch.abs(z) + self.b) * torch.exp(1.j * torch.angle(z)) 

def FFTPerChunk(x: torch.Tensor, length: int, inverse=False, n=None):
    if not inverse:
        n = x.shape[-1]
    # calculate the pad_n of x(spatial)
    pad_n = n + (length - (n % length)) % length
    assert pad_n % length == 0
    # pre-processing x
    if not inverse:
        x = torch.cat([x, torch.zeros(x.shape[:-1] + ((length - (n % length)) % length, )).to(x.device)], dim=-1)
        x = x.reshape(x.shape[:-1] + (pad_n//length, length))
    else:
        ffted_length = length // 2 + 1
        assert x.shape[-1] % ffted_length == 0
        x = x.reshape(x.shape[:-1] + (x.shape[-1]//ffted_length, ffted_length))
    # main fft(ifft)
    out = torch.fft.rfft(x) if not inverse else torch.fft.irfft(x, n=length)
    out = out.reshape(out.shape[:-2] + (-1,))
    if inverse:
        assert out.shape[-1] == pad_n
        out = out[:, :, :n]
    return out


class FourierTokenMixer(nn.Module):
    def __init__(self, dim, d_factor, window_size):
        super(FourierTokenMixer, self).__init__()

        self.conv_in = nn.Conv1d(in_channels=dim, out_channels=dim//d_factor, kernel_size=1)

        self.conv_spec = nn.Conv1d(in_channels=dim//d_factor, out_channels=dim//d_factor, kernel_size=1, dtype=torch.complex64)
        #self.acti_spec = real_imaginary_relu
        #self.acti_spec = phase_amplitude_relu
        self.acti_spec = mod_relu()

        self.conv_out = nn.Conv1d(in_channels=dim//d_factor, out_channels=dim, kernel_size=1)
        self.gelu = nn.GELU()

        self.ws = window_size
    
    def forward(self, x):
        x = self.conv_in(x)
        ffted_x = FFTPerChunk(x, length=self.ws)

        ffted_x = self.acti_spec(self.conv_spec(ffted_x))

        x = FFTPerChunk(ffted_x, length=self.ws, inverse=True, n=x.shape[-1])
        return self.gelu(self.conv_out(x))
