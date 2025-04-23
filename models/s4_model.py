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
                d_state = 64,
                dropout = 0.0, #0.1 0.0
                prenorm = 'store_true',
                norm_fn = 'batch', #### CHANGED!!!####
                weight_decay = 0.05,
                epochs = 100,
                batch_size = 64
                )   
        elif dataset == 'pathfinder':
            args = dict(
                n_layers = 6, #4 6 
                d_model = 256, #128 256
                d_state = 64,
                dropout = 0.0, #0.1 0.0
                prenorm = 'store_true',
                norm_fn = 'batch',
                epochs = 200,
                batch_size = 64,
                weight_decay = 0.05
                ) 
        elif dataset == 'cifar10':
            args = dict(
                n_layers = 4, #4 6 
                d_model = 128, #128 256
                d_state = 128,
                dropout = 0.1, #0.1 0.0
                prenorm = 'store_true',
                norm_fn = 'layer', #### CHANGED!!! layer is original ######
                epochs = 200,
                batch_size = 64,
                weight_decay = 0.05
                ) 
        elif dataset == 'text':
            args = dict(
                n_layers = 6, #4 6 
                d_model = 256, #128 256
                dropout = 0.0, #0.1 0.0
                prenorm = 'store_true',
                norm_fn = 'batch',
                batch_size = 16,
                epochs = 32,
                weight_decay = 0.05
                ) 
        elif dataset == 'list':
            args = dict(
                n_layers = 8, #4 6 
                d_model = 128, #128 256
                dropout = 0.0, #0.1 0.0
                prenorm = 'store_false',
                norm_fn = 'batch',
                batch_size = 50,
                epochs = 50,
                weight_decay = 0.05
                ) 
        else:
            args = dict(
                n_layers = 4, #4 6 
                d_model = 128, #128 256
                dropout = 0.1, #0.1 0.0
                prenorm = 'store_true',
                norm_fn = 'batch', #### CHANGED!!!######
                weight_decay = 0.05,
                epochs = 100,
                batch_size = 64
                ) 
        return args        

class S4Model(nn.Module):

    def __init__(self, params, d_input, d_output, weight_noise=None, **model_args):
        super(S4Model, self).__init__()

        n_layers = params['n_layers']
        if type(params['d_model']) is list or type(params['d_model']) is tuple: # or True
            d_model = params['d_model']
        else:
            d_model = [params['d_model'] for _ in range(n_layers)]

        if type(params['d_state']) is list or type(params['d_state']) is tuple: # or True
            d_state = params['d_state']
        else:
            d_state = [params['d_state'] for _ in range(n_layers)]
        
        dropout = params['dropout']
        lr = params['lr']
        prenorm = params['prenorm']
        dataset = params['dataset']
        norm_fn = params['norm_fn']

        self.prenorm = prenorm
        self.norm_fn = norm_fn
        
        if 'coder_quant' in model_args and model_args['coder_quant'] is not None:
            self.coder_quant = int(model_args['coder_quant'])
        else:
            self.coder_quant = None

        if 'model' in model_args and model_args['model'] is not None:
            spec_model = model_args['model']
        else:
            spec_model = None

        # Linear encoder (d_input = 1 for grayscale and 3 for RGB)
        if self.coder_quant is not None:
            self.encoder = pt.QuantizedLinear(d_input, d_model[0], quant_levels=self.coder_quant, quant_fn=pt.max_quant_fn)
        else:
            self.encoder = pt.BaseLinear(d_input, d_model[0])

        # Stack S4 layers as residual blocks
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        for l in range(n_layers):
            pass_args = deepcopy(model_args)
            for p in ['A_quant', 'C_quant', 'dt_quant', 'linear_quant', 'act_quant', 'state_quant']:

                if model_args[p] is None or model_args[p] is 'None':
                    pass_args[p] = None
                elif not type(model_args[p]) is list:
                    pass_args[p] = int(model_args[p])
                else: # if list of quant values provided
                    pass_args[p] = int(model_args[p][l])

            if spec_model == 'S4': #or (spec_model is None and dataset in ['pathfinder', 'text', 'list', 'hd']):
                self.s4_layers.append(
                    S4(d_model[l], d_state=d_state[l], dropout=dropout, transposed=True , **pass_args)  ##, lr=min(0.001, lr)
                )
            else:
                if l < n_layers - 1:
                    self.s4_layers.append(
                        S4D(d_model[l], d_state=d_state[l], dropout=dropout, d_model_next=d_model[l+1], transposed=True, weight_noise=weight_noise, **pass_args, lr=min(0.001, lr))
                    )
                else:
                    self.s4_layers.append(
                        S4D(d_model[l], d_state=d_state[l], dropout=dropout, transposed=True, weight_noise=weight_noise, **pass_args, lr=min(0.001, lr))
                    )
                
            if norm_fn == 'layer':
                if l < n_layers - 1:
                    self.norms.append(nn.LayerNorm(d_model[l+1]))
                else:
                    self.norms.append(nn.LayerNorm(d_model[l]))
            else:
                if l < n_layers - 1:
                    self.norms.append(nn.BatchNorm1d(d_model[l+1]))
                else:
                    self.norms.append(nn.BatchNorm1d(d_model[l]))

            self.dropouts.append(dropout_fn(dropout))

        # Linear decoder
        if self.coder_quant is not None:
            self.decoder = pt.QuantizedLinear(d_model[n_layers-1], d_output, quant_levels=self.coder_quant, quant_fn=pt.max_quant_fn)
        else:
            self.decoder = pt.BaseLinear(d_model[n_layers-1], d_output)

    def forward(self, x, analysis=False): # normal forward
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

            #import pdb
            #pdb.set_trace()
            z = x
            if self.prenorm:
                # Prenorm
                if self.norm_fn == 'layer':
                    z = norm(z.transpose(-1, -2)).transpose(-1, -2)                
                elif self.norm_fn == 'batch':
                    z = norm(z)
                    #z = norm(z.transpose(-1, -2)).transpose(-1, -2)

            # Apply S4 block: we ignore the state input and output
            z, _ = layer(z)
            if analysis:
                y_layers.append(deepcopy(z))

            # Dropout on the output of the S4 block
            z = dropout(z)

            # Residual connection
            x = z# + x

            if not self.prenorm:
                # Postnorm
                if self.norm_fn == 'layer':
                    x = norm(x.transpose(-1, -2)).transpose(-1, -2)
                elif self.norm_fn == 'batch':
                    x = norm(x)
                    #x = norm(x.transpose(-1, -2)).transpose(-1, -2)
        
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

    def forwardignore(self, inputs, analysis=False): # for hessian ->directly calculates loss
        """
        Input x is shape (B, L, d_input)
        """
        x, target = inputs
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

            #import pdb
            #pdb.set_trace()
            z = x
            if self.prenorm:
                # Prenorm
                if self.norm_fn == 'layer':
                    z = norm(z.transpose(-1, -2)).transpose(-1, -2)                
                elif self.norm_fn == 'batch':
                    z = norm(z)
                    #z = norm(z.transpose(-1, -2)).transpose(-1, -2)

            # Apply S4 block: we ignore the state input and output
            z, _ = layer(z)
            if analysis:
                y_layers.append(deepcopy(z))

            # Dropout on the output of the S4 block
            z = dropout(z)

            # Residual connection
            x = z# + x

            if not self.prenorm:
                # Postnorm
                if self.norm_fn == 'layer':
                    x = norm(x.transpose(-1, -2)).transpose(-1, -2)
                elif self.norm_fn == 'batch':
                    x = norm(x)
                    #x = norm(x.transpose(-1, -2)).transpose(-1, -2)
        
        x = x.transpose(-1, -2)

        # Pooling: average pooling over the sequence length
        x = x.mean(dim=1)

        # Decode the outputs
        x = self.decoder(x)  # (B, d_model) -> (B, d_output)
        #if self.coder_quant is not None:
        #    x = x - (x - max_quant_fn(x, quant_levels=self.coder_quant))
        return nn.CrossEntropyLoss(x, target)

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