import subprocess
from datetime import datetime
import os
import numpy as np
import copy

resume = False
resume_timestamp = ""

start_all = False

low_to_high = True

config_S4D_cifar10_init_high = {
    'N': 128,
    'H': 128,
    'd_model': 4,
    'd_in': 1,
    'd_out': 10,
    'r_A': 32,
    'r_B': 0,
    'r_C': 32,
    'r_state' : 32,
    'r_lin': 32,
    'r_coder': 32,
    'r_act': 32,
    'r_dt': 32,
    's4': False,
    'savepath': 'S4D_cifar10'
}

config_S4D_cifar10_init_low = {
    'N': 128,
    'H': 128,
    'd_model': 4,
    'd_in': 1,
    'd_out': 10,
    'r_A': 1,
    'r_B': 0,
    'r_C': 1,
    'r_state' : 1,
    'r_lin': 1,
    'r_coder': 1,
    'r_act': 1,
    'r_dt': 1,
    's4': False,
    'savepath': 'S4D_cifar10'
}

if low_to_high:
    config_S4D_cifar10 = config_S4D_cifar10_init_low
else:
    config_S4D_cifar10 = config_S4D_cifar10_init_high

optim_params = ['r_A', 'r_C', 'r_state', 'r_act', 'r_lin', 'r_coder', 'r_dt']

init_config = config_S4D_cifar10

if not resume:
    time_stamp = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
else:
    time_stamp = resume_timestamp
res_folder = "/Users/ssiegel/mem-hippo/optimization/exp_" + time_stamp
if not resume:
    os.mkdir(res_folder)
    os.mkdir(res_folder + "/checkpoints")
    summary_file = open(res_folder + "/summary", "w")

    for param in optim_params:
        summary_file.write(param + "\t")
    summary_file.write("cost\tbest_acc\tlast_acc\n")
    summary_file.close()


def get_ACE_metric(config, model_type='ssm'):
    if model_type == 'ssm':
        if config['s4']:
            ace_Ax = config['N'] * config['N'] *  config['r_A']* config['r_state'] * config['r_dt']
        else:
            ace_Ax = config['N'] * config['r_A']* config['r_state'] * config['r_dt']

        ace_Bu = config['N'] * config['r_B'] * config['r_act']
        ace_Cx = config['N'] * config['r_C'] * config['r_state']

        ace_kernel = ace_Ax + ace_Bu + ace_Cx
        ace_kernels = config['H'] * ace_kernel
        ace_mixing = config['H'] * config['H'] * config['r_act'] * config['r_lin']

        ace_layer = ace_kernels + ace_mixing
        ace_layers = config['d_model'] * ace_layer

        ace_encoder = config['H'] * config['d_in'] * config['r_coder'] * config['r_act']
        ace_decoder = config['H'] * config['d_out'] * config['r_coder'] * config['r_act']

        ace_total = ace_layers + ace_encoder + ace_decoder
        print("\n" + config['savepath'])
        print(ace_total / 1e6)

        kernel = np.array([ace_Ax / 1e6, ace_Bu / 1e6, ace_Cx / 1e6])
        kernel_labels = np.array(("Ax", "Bu", "Cx"), dtype=object)

        layer = np.append(kernel * config['H'], ace_mixing / 1e6, axis=None)
        layer_labels = np.append(kernel_labels, "mixing", axis=None)
        model = np.append(layer * config['d_model'], np.array((ace_encoder / 1e6, ace_decoder / 1e6)), axis=None)
        model_labels = np.append(layer_labels, np.array(("encoder", "decoder")))
    return model

def get_modelSize_metric(config, model_type='ssm'):
    if config['s4']:
        A = config['N'] * config['N'] * config['H'] * config['d_model'] * config['r_A']
        B = config['N'] * config['H'] * config['d_model'] * config['r_B']
    else:
        A = config['N'] * config['H'] * config['d_model'] * config['r_A']
        B = 0
    C = config['N'] * config['H'] * config['d_model'] * config['r_C']
    mixing = config['H'] * config['H'] * config['d_model'] * config['r_lin']
    encoder = config['H'] * config['d_in'] * config['r_coder']
    decoder = config['H'] * config['d_in'] * config['r_coder']
    return A, B, C, mixing, encoder, decoder

def get_ADC_metric(config, model_type='ssm'):
    kernel = (config['N'] * config['r_state'] + config['r_act']) * config['H'] * config['d_model']
    mixing = config['H'] * config['r_lin'] * config['d_model']
    coders = config['H'] * config['r_act'] + config['d_out'] * config['r_act']
    return kernel, mixing, coders


def calculate_next_steps(base_args, step_parameters, model_metric=None):
    global start_all
    steps = []
    step_costs = []

    for param in step_parameters:
        step_args = copy.deepcopy(base_args)
        if low_to_high:
            step_args[param] += 1
        else:
            if start_all:
                for p in optim_params:
                    step_args[param] -= 1
            else:
                step_args[param] -= 1
        if step_args[param] < 17 and step_args[param] > 0:
            steps.append(step_args)
            step_costs.append(np.sum(model_metric(step_args)))
    
    return steps, step_costs

def evaluate_model_step(run_config, cutoff_acc=80., max_cost_step=None, model_metric=None):
    global start_all
    tab_data = np.genfromtxt(res_folder + "/summary", delimiter="\t")

    if len(tab_data.shape) > 1:
        for line in tab_data[1:, :]:
            already_computed = True
            for i, param in enumerate(optim_params):
                if int(line[i]) != run_config[param]:
                    already_computed = False
            if already_computed:
                val_acc = line[-2]
                break
    else:
        already_computed = False

    if not already_computed:
        summary_file = open(res_folder + "/summary", "a+")
        for param in optim_params:
            summary_file.write(format(run_config[param]) + "\t")
        summary_file.write(format(np.sum(model_metric(run_config))) + "\t")
        summary_file.close()

        subprocess.run(["python", "training.py", "--dataset", "cifar10", "--grayscale",
                        "--A_quant", format(int(2**run_config['r_A'])),
                        "--C_quant", format(int(2**run_config['r_C'])),
                        "--state_quant", format(int(2**run_config['r_state'])),
                        "--act_quant", format(int(2**run_config['r_act'])),
                        "--linear_quant", format(int(2**run_config['r_lin'])),
                        "--coder_quant", format(int(2**run_config['r_coder'])),
                        "--dt_quant", format(int(2**run_config['r_dt'])),
                        "--check_path", time_stamp,
                        "--summary_file", res_folder + "/summary",
                        "--gpu", "4",
                        "--debug",
                        "--epochs", "50"
                        ]
        )

        tab_data = np.genfromtxt(res_folder + "/summary", delimiter="\t")
        val_acc = tab_data[-1, -2]

    if val_acc > cutoff_acc and not low_to_high or val_acc < cutoff_acc and low_to_high:
        steps, step_costs = calculate_next_steps(run_config, optim_params, model_metric=model_metric)
        sort_indices = np.argsort(np.array(step_costs))

        for ind in sort_indices:
            evaluate_model_step(steps[ind], cutoff_acc=cutoff_acc, model_metric=model_metric)
    else:
        if start_all:
            start_all = False


evaluate_model_step(init_config, model_metric=get_ACE_metric)