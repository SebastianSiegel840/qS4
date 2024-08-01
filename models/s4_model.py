import sys
import os
import argparse
import torch 
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict

import torch.backends.cudnn as cudnn
from tqdm.auto import tqdm
from copy import deepcopy

import precision_tools as pt

from os.path import expanduser
home = expanduser("~")
path_s4 = os.path.join(home, 'state-spaces')
if not(os.path.isdir(path_s4)):
    os.system('git clone git@github.com:YounesBouhadjar/state-spaces.git ~/state-spaces')
sys.path.append(path_s4)

from src.models.s4.s4 import S4
from src.models.s4.s4d import S4D
#from models_base.s4.s4 import S4Block as S4
#from models_base.s4.s4d import S4D

# Dropout broke in PyTorch 1.11
if tuple(map(int, torch.__version__.split('.')[:2])) == (1, 11):
    print("WARNING: Dropout is bugged in PyTorch 1.11. Results may be worse.")
    dropout_fn = nn.Dropout
if tuple(map(int, torch.__version__.split('.')[:2])) >= (1, 12):
    dropout_fn = nn.Dropout1d
else:
    dropout_fn = nn.Dropout2d

### Network Parameters ###
class return_args(object):
    def __init__(self, parser_args) -> None:
        dataset = parser_args.__dict__.get('dataset')
        args = self.return_args_dict(dataset=dataset)        
        for k, v in args.items():
            setattr(self, k, v) # Dictonary to arguments        
               
        for k, v in parser_args.__dict__.items():
            setattr(self, k, v) # Update config from config file using parsed arguments
        
        dim_in = int(self.n_fft/2)+1
        window_size = 480000 / self.splitting_factor
        #self.window_size = int(window_size/(dim_in//2)+1) if self.spectrogram else int(window_size)
        self.window_size = int(window_size)

        #padding = int((self.conv_kernel_size-self.conv_stride)/2)
        #self.n_tokens = int((self.window_size-self.conv_kernel_size+2*padding)/self.conv_stride+1) 

    def return_args_dict(self, dataset=None):
        if dataset is None:
            args = dict(
                n_layers = 6, #4 6 
                d_model = 256, #128 256
                dropout = 0.0, #0.1 0.0
                prenorm = 'store_true',
                )   
        elif dataset == 'pathfinder':
            args = dict(
                n_layers = 6, #4 6 
                d_model = 256, #128 256
                dropout = 0.0, #0.1 0.0
                prenorm = 'store_true',
                ) 
        else:
            args = dict(
                n_layers = 4, #4 6 
                d_model = 128, #128 256
                dropout = 0.1, #0.1 0.0
                prenorm = 'store_true',
                ) 
        return args        

class S4Model(nn.Module):

    def __init__(self, params, d_input, d_output, weight_noise=None, **model_args):
        super(S4Model, self).__init__()

        d_model = params['d_model']
        d_state = params['d_state']
        n_layers = params['n_layers']
        dropout = params['dropout']
        lr = params['lr']
        prenorm = params['prenorm']
        dataset = params['dataset']

        self.prenorm = prenorm
        
        if 'coder_quant' in model_args and model_args['coder_quant'] is not None:
            self.coder_quant = int(model_args['coder_quant'])
        else:
            self.coder_quant = None

        # Linear encoder (d_input = 1 for grayscale and 3 for RGB)
        if self.coder_quant is not None:
            self.encoder = pt.QuantizedLinear(d_input, d_model, quant_levels=self.coder_quant, quant_fn=pt.max_quant_fn)
        else:
            self.encoder = pt.BaseLinear(d_input, d_model)

        # Stack S4 layers as residual blocks
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        for _ in range(n_layers):
            if dataset == 'pathfinder':
                self.s4_layers.append(
                    S4(d_model, dropout=dropout, transposed=True, **model_args)  ## , lr=min(0.001, lr)
                )
                self.norms.append(nn.BatchNorm1d(1024))
            else:
                self.s4_layers.append(
                    S4D(d_model, d_state=d_state, dropout=dropout, transposed=True, weight_noise=weight_noise, **model_args, lr=min(0.001, lr))
                )
                self.norms.append(nn.LayerNorm(d_model))
            self.dropouts.append(dropout_fn(dropout))

        # Linear decoder
        if self.coder_quant is not None:
            self.decoder = pt.QuantizedLinear(d_model, d_output, quant_levels=self.coder_quant, quant_fn=pt.max_quant_fn)
        else:
            self.decoder = pt.BaseLinear(d_model, d_output)

    def forward(self, x, analysis=False):
        """
        Input x is shape (B, L, d_input)
        """
        #x = self.encoder(x.permute(0,2,1))  # (B, L, d_input) -> (B, L, d_model)
        x = self.encoder(x)  # (B, L, d_input) -> (B, L, d_model)
        #if self.coder_quant is not None:
        if analysis:
            y_encoder = deepcopy(x)
            if self.coder_quant is not None:
                x = x - (x - max_quant_fn(x, quant_levels=self.coder_quant))

        y_layers = []
        x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)
        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)

            z = x
            if self.prenorm:
                # Prenorm
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)

            # Apply S4 block: we ignore the state input and output
            z, _ = layer(z)
            if analysis:
                y_layers.append(deepcopy(z))

            # Dropout on the output of the S4 block
            z = dropout(z)

            # Residual connection
            x = z + x

            if not self.prenorm:
                # Postnorm
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)

        x = x.transpose(-1, -2)

        # Pooling: average pooling over the sequence length
        x = x.mean(dim=1)

        # Decode the outputs
        x = self.decoder(x)  # (B, d_model) -> (B, d_output)
        #if self.coder_quant is not None:
        #    x = x - (x - max_quant_fn(x, quant_levels=self.coder_quant))
        if analysis:
            return x, y_encoder, y_layers#.permute(0,2,1)
        else:
            return x

if __name__ == "__main__":

    params = {}
    params['d_input'] = 1
    params['d_output'] = 10
    params['n_layers'] = 1
    params['d_model'] = 2 #128
    params['dropout'] = 0.1
    params['prenorm'] = 'store_true'
    params['lr'] = 0.01

    # Model
    print('==> Building model..')
    model = S4Model(
        params
    )


# taken from bitnet 1.58b
def max_quant_fn(a, quant_levels=2):
        # scaling parameter to get an estimate of the magnitude of the activations. 
    # clamp to avoid division by zero
    #import pdb
    #pdb.set_trace()
    scale = quant_levels / 2 / torch.clamp(torch.max(a.abs().flatten(), dim=-1, keepdim=True)[0], min=1e-5) 

    # a * scale normalizes a. rounding brings them to the next integer. 
    # clamping to cut off values above the quantization limits. / scale to undo normalization
    a_out = torch.clamp((a * scale).round(), min=-quant_levels // 2, max=quant_levels // 2) / scale
    return a_out

# taken from bitnet 1.58b
def mean_quant_fn(w, quant_levels=2):
    # scaling parameter to get an estimate of the magnitude of the weights. 
    # clamp to avoid division by zero
    scale = quant_levels / 2 / w.abs().flatten().mean().clamp(min=1e-5) 

    # w * scale normalizes w. rounding brings them to the next integer. 
    # clamping to cut off values above the quantization limits. / scale to undo normalization
    w_out = (w * scale).round().clamp(-quant_levels // 2, quant_levels // 2) / scale
    return w_out