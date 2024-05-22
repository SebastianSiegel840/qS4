import sys
import torch
import time
import torch.nn as nn

from mamba_ssm import Mamba

import snntorch as snn
from snntorch import surrogate


### Network Parameters ###
class return_args(object):
    def __init__(self, parser_args) -> None:
        args = self.return_args_dict()        
        for k, v in args.items():
            setattr(self, k, v) # Dictonary to arguments        
               
        for k, v in parser_args.__dict__.items():
            setattr(self, k, v) # Update config from config file using parsed arguments
            
        dim_in = int(self.n_fft/2)+1
        window_size = 480000 / self.splitting_factor
        self.window_size = int(window_size/(dim_in//2)+1) if self.spectrogram else int(window_size)
        #padding = int((self.conv_kernel_size-self.conv_stride)/2)
        #self.n_tokens = int((self.window_size-self.conv_kernel_size+2*padding)/self.conv_stride+1) 

    def return_args_dict(self):
        args = dict(
            project="IntelDNS",
            name="test",
            group="TESTS",

            #window_size = 480000,  #3751, #16000,#32000, 
            # splitting_factor= 30,#15,#1,
            # window_size = 16000,  #3751, #16000,#32000, 
            #splitting_factor= 30,#30,#15,#1,

            d_input = 784,
            d_output = 8,
            d_model = 128,
            d_state = 256

            #lr = 0.01,
            #wandb_status="disabled" # "online", "offline"
            )   
        return args        


class MAMBAModel(nn.Module):

    def __init__(self, params, d_input, d_output, params_scan=None):
        super(MAMBAModel, self).__init__()

        self.d_input = d_input
        self.d_output = d_output
        self.d_model = params['d_model']
        self.d_state = params['d_state']

        if params_scan is not None and params_scan:
            self.std_noise = params_scan['std_noise']
            self.is_variable_B = params_scan['is_variable']
            self.is_variable_C = params_scan['is_variable']
            self.enable_conv = params_scan['enable_conv']
        else:
            # self.std_noise = params['std_noise']
            self.is_variable_B = True
            self.is_variable_C = True
            self.enable_conv = True

        # Linear decoder
        self.encoder = nn.Linear(self.d_input, 
                                 self.d_model
                                 )

        self.mamba = Mamba(
                       # This module uses roughly 3 * expand * d_model^2 parameters
                       d_model=self.d_model, # Model dimension d_model
                       d_state=self.d_state, # SSM state expansion factor
                       d_conv=4,    # Local convolution width
                       expand=2,    # Block expansion factor
                       #std_noise=self.std_noise,
                       #is_variable_B=self.is_variable_B,
                       #is_variable_C=self.is_variable_C,
                       #enable_conv=self.enable_conv
                    )

        # Linear decoder
        self.decoder = nn.Linear(self.d_model,
                                 self.d_output)

    def forward(self, x):
        """
        Input x is shape (B, L, d_input)
        """

        # Encode input
        x = self.encoder(x.permute(0,2,1))

        y = self.mamba(x)

        # Decode the outputs
        y = self.decoder(y)

        return y.permute(0,2,1)


if __name__ == "__main__":
           
    params = {}
    params['d_input'] = 1
    params['d_output'] = 10
    params['n_layers'] = 4
    params['d_model'] = 128
    params['dropout'] = 0.1

    # Model
    print('==> Building model..')
    model = MAMBAModel(
        params
    )
