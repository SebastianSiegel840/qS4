print("Started!")

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import torchaudio.transforms as audio_T

import os
import argparse
import time
import wandb
import numpy as np

from tqdm.auto import tqdm
import precision_tools as pt

import audio_dataset
#from audio_dataloader_split_samples import DNSAudio

import sys
sys.path.append("models")

print("Modules load!")

dataset_folder = '/Data/pgi-15/datasets/intel_dns/'

# Dropout broke in PyTorch 1.11
if tuple(map(int, torch.__version__.split('.')[:2])) == (1, 11):
    print("WARNING: Dropout is bugged in PyTorch 1.11. Results may be worse.")
    dropout_fn = nn.Dropout
if tuple(map(int, torch.__version__.split('.')[:2])) >= (1, 12):
    dropout_fn = nn.Dropout1d
else:
    dropout_fn = nn.Dropout2d


parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='test', help='wandb run name')
parser.add_argument('--project', type=str, default='test', help='wandb project name')
parser.add_argument('--wandb_status', type=str, default='disabled', help='wandb mode: online, offline, disabled')
# Optimizer
parser.add_argument('--lr', default=0.01, type=float, help='Learning rate')
parser.add_argument('--weight_decay', default=0.05, type=float, help='Weight decay')
# Scheduler
# parser.add_argument('--patience', default=10, type=float, help='Patience for learning rate scheduler')
parser.add_argument('--epochs', default=100, type=int, help='Training epochs')
# Dataset
parser.add_argument('--dataset', default='hd', choices=['mnist', 'cifar10', 'hd', 'dn', 'pathfinder'], type=str, help='Dataset')
parser.add_argument('--grayscale', action='store_true', help='Use grayscale CIFAR10')
parser.add_argument('--subsample', default=8,type=int, help='specify subsampling ratio')
# Dataloader
parser.add_argument('--num_workers', default=4, type=int, help='Number of workers to use for dataloader')
parser.add_argument('--batch_size', default=64, type=int, help='Batch size')
# Model # SEE models/
parser.add_argument('--n_layers_m', default=1, type=int, help='Number of layers')
parser.add_argument('--d_model_m', default=None, type=int, help='Model dimension')
parser.add_argument('--d_state', default=64, type=int, help='State dimension')
parser.add_argument('--dropout_m', default=None, type=float, help='Dropout')
parser.add_argument('--prenorm', action='store_true', help='Prenorm')
# General
parser.add_argument('--resume', '-r', action='store_true', help='Resume from checkpoint')
parser.add_argument('--model_file', type=str, default='s4_model',  help='which model file to use')
parser.add_argument('--net', default="S4Model", type=str, help='which model class to use')
parser.add_argument('--splitting_factor', type=int, default=1, help='Number of chunks to divide the initial samples')
parser.add_argument('--gpu', type=int, default=[1], help='which gpu(s) to use', nargs='+')
parser.add_argument('--n_fft', type=int, default=512, help='number of FFT specturm, hop is n_fft // 4')

parser.add_argument('--energy', action='store_true', help='Activate energy monitoring via Zeus')

parser.add_argument('--kernel_quant', default=None)
parser.add_argument('--linear_quant', default=None)
parser.add_argument('--A_quant', default=None)
parser.add_argument('--B_quant', default=None)
parser.add_argument('--C_quant', default=None)
parser.add_argument('--dt_quant', default=None)
parser.add_argument('--act_quant', default=None)
parser.add_argument('--coder_quant', default=None)
parser.add_argument('--all_quant', default=None)
parser.add_argument('--state_quant', default=None)

parser.add_argument('--nonlin', default='glu')

parser.add_argument('--debug', action='store_true')
parser.add_argument('--hd_small', action='store_true')

parser.add_argument('--defmax', default=None)
parser.add_argument('--defmax_train', action='store_true')

parser.add_argument('--check_path', default=None)

parser_args = parser.parse_args()

model_lib = __import__(parser_args.model_file)

### Save manual model configs ###
n_layers = parser_args.n_layers_m
d_model = parser_args.d_model_m
dropout = parser_args.dropout_m
prenorm = parser_args.prenorm

model_lib = __import__(parser_args.model_file)

args = getattr(model_lib, 'return_args')(parser_args) # Network specific configs


### Enter manual configs if present ###
if n_layers is not None:
    args.n_layers = n_layers
if d_model is not None:
    args.d_model = d_model
if dropout is not None:
    args.dropout = dropout


setattr(args, 'device', args.gpu[0]) if len(args.gpu)==1 else None

device = torch.device('cuda:{}'.format(args.gpu[0]))
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# create a wandb session 
wandb_run = wandb.init(project=args.project,
                        name=args.name,
                        config=args,
                        mode=args.wandb_status)
# change args dictonary to a wandb config object and allow wandb to track it
args = wandb_run.config  

# Data
print(f'==> Preparing {args.dataset} data..')

def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_nontrain_parameters(model):
    return sum(p.numel() for p in model.parameters() if not p.requires_grad)

def split_train_val(train, val_split):
    train_len = int(len(train) * (1.0-val_split))
    train, val = torch.utils.data.random_split(
        train,
        (train_len, len(train) - train_len),
        generator=torch.Generator().manual_seed(42),
    )
    return train, val

if args.dataset == 'cifar10':

    if args.grayscale:
        transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize(mean=122.6 / 255.0, std=61.0 / 255.0),
            transforms.Lambda(lambda x: x.view(1, 1024).t())
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            transforms.Lambda(lambda x: x.view(3, 1024).t())
        ])

    # S4 is trained on sequences with no data augmentation!
    transform_train = transform_test = transform

    trainset = torchvision.datasets.CIFAR10(
        root='./data/cifar/', train=True, download=True, transform=transform_train)
    trainset, _ = split_train_val(trainset, val_split=0.1)

    valset = torchvision.datasets.CIFAR10(
        root='./data/cifar/', train=True, download=True, transform=transform_test)
    _, valset = split_train_val(valset, val_split=0.1)

    testset = torchvision.datasets.CIFAR10(
        root='./data/cifar/', train=False, download=True, transform=transform_test)

    d_input = 3 if not args.grayscale else 1
    d_output = 10
    collate_fn=None

    print("d_input :" + format(d_input))

elif args.dataset == "pathfinder":
    from pathfinder import PathFinderDataset
    trainset = PathFinderDataset(transform=transforms.ToTensor())
    valset = PathFinderDataset(transform=transforms.ToTensor())
    testset = PathFinderDataset(transform=transforms.ToTensor())
    d_input = 1
    d_output = 2
    collate_fn = None


elif args.dataset == 'mnist':

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(1, 784).t())
    ])
    transform_train = transform_test = transform

    trainset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform_train)
    trainset, _ = split_train_val(trainset, val_split=0.1)

    valset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform_test)
    _, valset = split_train_val(valset, val_split=0.1)

    testset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform_test)

    d_input = 1
    d_output = 10
    collate_fn = None

elif args.dataset == "hd":
    #resampler = audio_T.Resample(48000, 1000)
    
    #transform = resampler
    transform_train = transform_test = None
    datapath = "/Users/ssiegel/datasets"
    trainset = audio_dataset.HD(
        path=datapath + "/hd_audio", 
        subset="train", 
        language="english", 
        transform=transform_train,
        subsample=args.subsample)
    print("Passed")
    valset = audio_dataset.HD(
        path=datapath + "/hd_audio", 
        subset="train", 
        language="english", 
        transform=transform_test,
        subsample=args.subsample)

    testset = audio_dataset.HD(
        path=datapath + "/hd_audio", 
        subset="test", 
        language="english", 
        transform=transform_test,
        subsample=args.subsample)

    d_input = 1
    d_output = 10
    collate_fn = None

elif args.dataset == "dn": # denoising task
    trainset = DNSAudio(args, root=dataset_folder + 'training_set/')
    valset = DNSAudio(args, root=dataset_folder + 'validation_set/')
    testset = DNSAudio(args, root=dataset_folder + 'validation_set/') #TODO test set

    d_input = 1
    d_output = 1
    def collate_fn(batch):
        noisy, clean, noise = [], [], []

        for sample in batch:
            noisy += [torch.FloatTensor(sample[0])]
            clean += [torch.FloatTensor(sample[1])]
            #noise += [torch.FloatTensor(sample[2])]

        #return torch.stack(noisy), torch.stack(clean), torch.stack(noise)
        return torch.stack(noisy), torch.stack(clean)


else: raise NotImplementedError

def custom_collate(batch):
    #max_length = torch.max([t.shape[0] for t in batch])
    batch_padded = []
    max_length = 0
    for t in batch:
        if t[0].shape[1] > max_length:
            max_length = t[0].shape[1]

    for t in batch:
        sample_padded = torch.zeros((1, max_length))
        sample_padded[0, 0:t[0].shape[1]] = t[0][0, :]
        batch_padded.append((sample_padded, t[1]))
    #batch_padded = torch.tensor(batch_padded).to(device)
    return batch_padded

# Dataloaders
testloader = torch.utils.data.DataLoader(
    testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

if device == 'cuda':
    cudnn.benchmark = True


###############################################################################
# initialize model and test
###############################################################################

def eval(model, dataloader, save_path=None, num_batches=None, noise_mag=0.0, noise_std=0.0):
    global best_acc
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        pbar = tqdm(enumerate(dataloader), disable=True)
        for batch_idx, (inputs, targets) in pbar:
            if num_batches is not None and batch_idx >= num_batches:
                break

            inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs + noise_mag * torch.normal(0., noise_std, inputs.shape).to(device)

            outputs = model(inputs)
            
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        acc = 100.*correct/total
    return acc

if args.all_quant is not None:
    model_args = {
        'A_quant': args.all_quant,
        'B_quant': args.all_quant,
        'C_quant': args.all_quant,
        'dt_quant': args.all_quant,
        'kernel_quant': args.all_quant,
        'linear_quant': args.all_quant,
        'act_quant': args.all_quant,
        'coder_quant': args.all_quant,
        'state_quant': args.all_quant,
        'nonlin': args.nonlin,
        'defmax': args.defmax,
        'defmax_train': args.defmax_train
    }
else:
    model_args = {
        'A_quant': args.A_quant,
        'B_quant': args.B_quant,
        'C_quant': args.C_quant,
        'dt_quant': args.dt_quant,
        'kernel_quant': args.kernel_quant,
        'linear_quant': args.linear_quant,
        'act_quant': args.act_quant,
        'coder_quant': args.coder_quant,
        'state_quant': args.state_quant,
        'nonlin': args.nonlin,
        'defmax': args.defmax,
        'defmax_train': args.defmax_train
    }

for arg in ['A_quant', 'C_quant', 'B_quant', 'dt_quant', 'kernel_quant', 'linear_quant', 'act_quant', 'coder_quant', 'state_quant']:
    if model_args[arg] == 'None':
        model_args[arg] = None

def evaluate_model(check_path, model_args, quantization, levels, num_batches=None, res_file=None): # quantization is list of tuples ('quant_param', value)
    if check_path is None:
        if args.dataset == 'cifar10':
            check_point_test_path = './checkpoint/baseline_gr/ckpt0.pth'
            save_path = 'baseline_gr/ckpt0.pth'
        elif args.dataset == 'hd':
            check_point_test_path = './checkpoint/baseline_hd_1layer/ckpt0.pth'
            save_path = 'baseline_hd_1layer/ckpt0.pth'
        elif args.dataset == 'pathfinder':
            check_point_test_path = './checkpoint/baseline_S4D_path_6l/ckpt0.pth'
            save_path = 'baseline_S4D_path_6l/ckpt0.pth'
    else:
        check_point_test_path = './checkpoint/' + check_path
        save_path = check_path
    check_point = torch.load(check_point_test_path)

    print(f"{check_path : <40}", end="")
    if res_file is not None:
        res_file.write(check_path + "\t")
    
    for elem in quantization:
        if elem[0] == 'all_quant':
            for arg in ['A_quant', 'C_quant', 'B_quant', 'dt_quant', 'kernel_quant', 'linear_quant', 'act_quant', 'coder_quant', 'state_quant']:
                model_args[arg] = elem[1]
        else:
            model_args[elem[0]] = elem[1]
    
    print(f"{'  ' : <15}", end="")
    if res_file is not None:
        res_file.write("\t\t")

    model_test = getattr(model_lib, args.net)(args, d_input, d_output, **model_args).to(device)

    print(f"{'%.2f' : <10}" % check_point['acc'], end="")
    if res_file is not None:
        res_file.write("%.2f\t" % check_point['acc'])
    model_test.load_state_dict(check_point['model'])

    acc = eval(model_test, testloader, save_path=save_path, num_batches=num_batches, noise_mag=0.0, noise_std=0.)
    print(f"{'%.2f' : <10}" % acc, end="")
    if res_file is not None:
        res_file.write("%.2f\t" % acc)

    for std in levels:
        acc = 0
        for _ in range(5):
            acc = acc + eval(model_test, testloader, save_path=save_path, num_batches=num_batches, noise_mag=1.0, noise_std=std)
        acc /= 5
        print(f"{'%.2f' : <13}" % acc, end="")
        if res_file is not None:
            res_file.write("%.2f\t" % acc)
    if res_file is not None:
        res_file.write("\n")
    print("\n")

def print_header(levels, res_file=None):
    print(f"{'Chechpoint' : <40}", end="")
    print(f"{'quant' : <15}", end="")
    print(f"{'saved' : <10}", end="")
    print(f"{'no noise' : <10}", end="")
    if res_file is not None:
        res_file.write("Checkpoint\tquant\tsaved acc\tno noise\t")

    for lev in levels:
        print(f"{'%.2E' : <10}" % lev, end="")
        if res_file is not None:
            res_file.write(format(lev))
    if res_file is not None:
        res_file.write("\n")
    
    print("\n\n")


quantized_param = 'act'

low = 0.
high = 1e-1
step = 1e-2
levels = np.arange(low, high, step)

#high = 0
#low = -4
#steps_per_dec=1
#levels = np.logspace(low, high, num=(high - low)*steps_per_dec+1, base=10)

res_file = open("./evaluation/noise/input_noise_" + quantized_param + "_hd.txt", "w")

base_path = quantized_param + "_quant_max_hd_S4D_1l_noise/" + quantized_param
quant_levels = [16384, 1024, 512, 256, 128, 64, 16, 8, 4, 2]

print_header(levels, res_file=res_file)

for iq, quant_level in enumerate(quant_levels):
    if quant_level is not None:
        #add = " " + format(iq + 1) if iq > 0 else " 0"
        add = ""
        evaluate_model(base_path + format(quant_level) + add + ".pth", model_args, [(quantized_param + '_quant', quant_level)], levels, res_file=res_file)
    else:
        evaluate_model(None, model_args, [(quantized_param + '_quant', quant_level)], levels, res_file=res_file)
    
res_file.close()
print("\n")

'''
#high_exp = 0
#low_exp = -9
#steps_per_dec=1
#levels = np.logspace(low_exp, high_exp, num=(high_exp - low_exp)*steps_per_dec+1, base=10)
levels = np.linspace(1e-2, 1e-1, 10)
res_file = open("./evaluation/noise/hd_1l/input_noise_act.txt", "w")
#base_path = "all_quant_max_gr_rerun/all"
base_path = "act_quant_max_S4D_hd_1l_rerun_2/act"
quant_levels = [16384,  256, 64, 16, 8, 4, 2]

print_header(levels, res_file=res_file)

for quant_level in quant_levels:
    if quant_level is not None:
        evaluate_model(base_path + format(quant_level) + ".pth", model_args, [('act_quant', quant_level)], levels, res_file=res_file)
    else:
        evaluate_model(None, model_args, [('act_quant', quant_level)], levels, res_file=res_file)
    
res_file.close()
print("\n")
'''