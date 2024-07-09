#import matplotlib.pyplot as plt
import numpy as np
import copy

N = 64
H = 256
d_model = 4
d_in = 1
d_out = 2
r_A = 32
r_B = 32
r_C = 32
r_lin = 32
r_coder = 32
r_act = 32

model_params = (N, H, d_model, d_in, d_out)
resolutions = np.array((r_A, r_B, r_C, r_lin, r_coder, r_act))

s4 = False # if  not then S4D

def compute_ace(model_params, resolutions, s4=False):
    (N, H, d_model, d_in, d_out) = model_params
    (r_A, r_B, r_C, r_lin, r_coder, r_act) = resolutions
    
    if s4:
        ace_Ax = N * N *  r_A * r_act
    else:
        ace_Ax = N * r_A * r_act

    ace_Bu = N * r_B * r_act
    ace_Cx = N * r_C * r_act


    ace_kernel = ace_Ax + ace_Bu + ace_Cx
    ace_kernels = H * ace_kernel
    ace_mixing = H * H * r_act * r_lin

    ace_layer = ace_kernels + ace_mixing
    ace_layers = d_model * ace_layer

    ace_encoder = H * d_in * r_coder * r_act
    ace_decoder = H * d_out * r_coder * r_act

    ace_total = ace_layers + ace_encoder + ace_decoder
    #print(ace_total)
    return ace_total

def compute_complexity_impact(model_params, resolutions):
    ace_changes = np.zeros(len(resolutions))
    for ires, res in enumerate(resolutions):
        if resolutions[ires] > 1:
            lowered_resolutions = copy.copy(resolutions)
            lowered_resolutions[ires] = lowered_resolutions[ires] - 1
            ace_changes[ires] = compute_ace(model_params, resolutions) - compute_ace(model_params, lowered_resolutions)
    ilower = np.argwhere(ace_changes == np.max(ace_changes))
    delete_list = []
    for i, elem in enumerate(ilower):
        if resolutions[elem] < 1:
            delete_list.append(i)
    ilower = np.delete(ilower, delete_list)
    return ilower, ace_changes


for i in range(256):
    print(format(resolutions) + "\t" + format(compute_ace(model_params, resolutions)))

    ilower, _ = compute_complexity_impact(model_params, resolutions)
    for elem in ilower:
        resolutions[elem] = resolutions[elem] - 1

