'''
Train an S4 model on sequential CIFAR10 / sequential MNIST with PyTorch for demonstration purposes.
This code borrows heavily from https://github.com/kuangliu/pytorch-cifar.

This file only depends on the standalone S4 layer
available in /models/s4/

* Train standard sequential CIFAR:
    python -m example
* Train sequential CIFAR grayscale:
    python -m example --grayscale
* Train MNIST:
    python -m example --dataset mnist --d_model 256 --weight_decay 0.0

The `S4Model` class defined in this file provides a simple backbone to train S4 models.
This backbone is a good starting point for many problems, although some tasks (especially generation)
may require using other backbones.

The default CIFAR10 model trained by this file should get
89+% accuracy on the CIFAR10 test set in 80 epochs.

Each epoch takes approximately 7m20s on a T4 GPU (will be much faster on V100 / A100).
'''
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
import sys
import copy

from tqdm.auto import tqdm
    
import audio_dataset

from os.path import expanduser
home = expanduser("~")
path_s4 = os.path.join(home, 'state-spaces')
sys.path.append(path_s4)
#from src.dataloaders import lra

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
parser.add_argument('--name', type=str, default='example', help='wandb run name')
parser.add_argument('--project', type=str, default='quant SSM', help='wandb project name')
parser.add_argument('--wb', type=str, default='disabled', help='wandb mode: online, offline, disabled')
parser.add_argument('--sw', dest='run_sweep', type=bool, default=False, help="Activate wb sweep run")

# Optimizer
parser.add_argument('--lr', default=0.01, type=float, help='Learning rate')
parser.add_argument('--weight_decay', default=0.05, type=float, help='Weight decay')
# Scheduler
# parser.add_argument('--patience', default=10, type=float, help='Patience for learning rate scheduler')
parser.add_argument('--epochs', default=100, type=int, help='Training epochs')
# Dataset
parser.add_argument('--dataset', default='cifar10', choices=['mnist', 'cifar10', 'hd', 'dn', 'pathfinder', 'sc', 'list', 'text'], type=str, help='Dataset')
parser.add_argument('--grayscale', action='store_false', help='Use grayscale CIFAR10')
parser.add_argument('--subsample', default=1, type=int, help='specify subsampling ratio')
# Dataloader
parser.add_argument('--num_workers', default=4, type=int, help='Number of workers to use for dataloader')
parser.add_argument('--batch_size', default=64, type=int, help='Batch size')
# Model # SEE models/
parser.add_argument('--n_layers_m', default=None, type=int, help='Number of layers')
parser.add_argument('--d_model_m', default=128, type=int, help='Model dimension')
parser.add_argument('--d_state', default=64, type=int, help='State dimension')
parser.add_argument('--dropout_m', default=None, type=float, help='Dropout')
parser.add_argument('--prenorm', action='store_true', help='Prenorm')
# General
parser.add_argument('--resume', '-r', action='store_true', help='Resume from checkpoint')
parser.add_argument('--model_file', type=str, default='s4_model',  help='which model file to use')
parser.add_argument('--net', default="S4Model", type=str, help='which model class to use')
parser.add_argument('--splitting_factor', type=int, default=1, help='Number of chunks to divide the initial samples')
parser.add_argument('--gpu', type=int, default=[0], help='which gpu(s) to use', nargs='+')
parser.add_argument('--n_fft', type=int, default=512, help='number of FFT specturm, hop is n_fft // 4')

parser.add_argument('--energy', action='store_true', help='Activate energy monitoring via Zeus')
# Quantization
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
parser.add_argument('--model', default=None)
parser.add_argument('--mode', default=None)
parser.add_argument('--measure', default=None)

parser.add_argument('--debug', action='store_true')
parser.add_argument('--hd_small', action='store_true')

parser.add_argument('--defmax', default=None)
parser.add_argument('--defmax_train', action='store_true')
parser.add_argument('--weight_noise', default=None, type=float, help='Weight noise std')

parser.add_argument('--check_path', default=None)
parser.add_argument('--summary_file', default=None)

parser.add_argument('--p_ckpt', default=None)

parser_args = parser.parse_args()

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

if not(args.run_sweep):

    setattr(args, 'device', args.gpu[0]) if len(args.gpu)==1 else None

    device = torch.device('cuda:{}'.format(args.gpu[0]))

    if args.energy:
        from zeus.monitor import ZeusMonitor
        monitor = ZeusMonitor(gpu_indices=[args.gpu[0]])
else:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("\nCUDA enabled")
    else:
        device = torch.device("cpu")
        print("\nCUDA not available")

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# create a wandb session 
if args.run_sweep:
    wandb.init(  # name=args.name,
        mode=args.wb,
        config=args)

    # config_wb = wandb.config
else:
    wandb.init(project=args.project,
               name=args.name,
               mode=args.wb,
               config=args)

# change args dictonary to a wandb config object and allow wandb to track it
args = wandb.config 

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
    if args.hd_small:
        classes = ['0', '1']
    else:
        classes = None
    
    #transform = resampler
    transform_train = transform_test = None
    datapath = "/Local/ssiegel/datasets"
    trainset = audio_dataset.HD(
        path=datapath + "/hd_audio", 
        subset="train", 
        language="english", 
        transform=transform_train,
        subsample=args.subsample,
        classes=classes)
    print("Passed")
    valset = audio_dataset.HD(
        path=datapath + "/hd_audio", 
        subset="test", 
        language="english", 
        transform=transform_test,
        subsample=args.subsample,
        classes=classes)

    testset = audio_dataset.HD(
        path=datapath + "/hd_audio", 
        subset="test", 
        language="english", 
        transform=transform_test,
        subsample=args.subsample,
        classes=classes)

    d_input = 1
    if args.hd_small:
        d_output = 2
    else:
        d_output = 10
    collate_fn = None

elif args.dataset == "sc":
    gsc_path = "/Data/pgi-15/datasets/GSC/"

    transform_train = transform_test = None
    datapath = "/Users/ssiegel/datasets"
    trainset = audio_dataset.SC(
        path=gsc_path, 
        subset="training",
        transform=transform_train,
        subsample=args.subsample)
    print("Passed")
    valset = audio_dataset.SC(
        path=gsc_path, 
        subset="validation", 
        transform=transform_test,
        subsample=args.subsample)

    testset = audio_dataset.SC(
        path=gsc_path, 
        subset="testing", 
        transform=transform_test,
        subsample=args.subsample)

    d_input = 1
    d_output = 36
    collate_fn = None

elif args.dataset == "pathfinder":
    from pathfinder import PathFinderDataset
    trainset = PathFinderDataset(transform=transforms.ToTensor())

    len_dataset = len(trainset)

    val_split = 0.1
    test_split = 0.1
    val_len = int(val_split * len_dataset)
    test_len = int(test_split * len_dataset)
    train_len = len_dataset - val_len - test_len

    (trainset,
     valset,
     testset) = torch.utils.data.random_split(
             trainset,
             [train_len, val_len, test_len],
             generator=torch.Generator().manual_seed(42))

    d_input = 1
    d_output = 2

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
else: 
    raise NotImplementedError 


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
if args.dataset == "pathfinder":
    trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    valloader = torch.utils.data.DataLoader(
            valset, batch_size=args.batch_size, shuffle=False, drop_last=True) ### shuffle true does not work
    testloader = torch.utils.data.DataLoader(
            testset, batch_size=args.batch_size, shuffle=False, drop_last=True) ### shuffle true does not work

else:
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn)
    valloader = torch.utils.data.DataLoader(
        valset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn)

model_args = {
    'A_quant': args.A_quant,
    'B_quant': args.B_quant,
    'C_quant': args.C_quant,
    'dt_quant': args.dt_quant,
    'kernel_quant': args.kernel_quant,
    'linear_quant': args.linear_quant,
    'act_quant': args.act_quant,
    'coder_quant': args.coder_quant,
    'nonlin': args.nonlin
}

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

checkname = ""
if args.all_quant is not None:
    checkname = checkname + "all" + format(args.all_quant)
if args.A_quant is not None:
    checkname = checkname + "A" + format(args.A_quant)
if args.B_quant is not None:
    checkname = checkname + "B" + format(args.B_quant)
if args.C_quant is not None:
    checkname = checkname + "C" + format(args.C_quant)
if args.dt_quant is not None:
    checkname = checkname + "dt" + format(args.dt_quant)
if args.kernel_quant is not None:
    checkname = checkname + "kernel" + format(args.kernel_quant)
if args.linear_quant is not None:
    checkname = checkname + "linear" + format(args.linear_quant)
if args.act_quant is not None:
    checkname = checkname + "act" + format(args.act_quant)
if args.coder_quant is not None:
    checkname = checkname + "coder" + format(args.coder_quant)
if args.state_quant is not None:
    checkname = checkname + "state" + format(args.state_quant)
if args.weight_noise is not None:
    checkname = checkname + "weight_noise" + format(args.weight_noise)

if checkname == "":
    checkname = "baseline"

# Model
print('==> Building model..')
if args.debug:
    model = getattr(model_lib, args.net)(args, d_input, d_output, **model_args, weight_noise=args.weight_noise).to(device)
else:
    try:
        model = getattr(model_lib, args.net)(args, d_input, d_output, **model_args, weight_noise=args.weight_noise).to(device) # Please ensure that your model takes arguments (args, dim_in, dim_out) with args a class object with network config., model_args=model_args
    except:
        exit()
##############################################
#### Save initial state ######################
state = {
    'model': model.state_dict(),
    'args': args,
}

if not os.path.isdir('checkpoint'):
    os.mkdir('checkpoint')
if args.check_path is not None and not os.path.isdir('checkpoint/' + args.check_path):
    os.mkdir('checkpoint/' + args.check_path)

###############################################
###############################################

if device == 'cuda':
    cudnn.benchmark = True

if args.resume:
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()

def interpolate_checkpoints(ckpt1, ckpt2, alpha):
    ckpt_new = copy.deepcopy(ckpt1)
    for param in ckpt1['model']:
        ckpt_new['model'][param].to(device)
        ckpt_new['model'][param] = ckpt1['model'][param].to(device) * (1-alpha) + ckpt2['model'][param].to(device) * alpha
    return ckpt_new


###############################################################################
# Everything after this point is standard PyTorch training!
###############################################################################

t_param = count_trainable_parameters(model)
print("Trainable parameters:\t" + format(t_param))
nt_param = count_nontrain_parameters(model)
print("Non-trainable parameters:\t" + format(nt_param))

if not os.path.isdir('checkpoint'):
    os.mkdir('checkpoint')
if args.check_path is not None:
    num_ckpt = len(os.listdir('./checkpoint/' + args.check_path + '/'))
else:
    num_ckpt = len(os.listdir('./checkpoint/'))

def eval(dataloader, test=False, checkpoint=False):
    global best_acc
    model.eval()
    eval_loss = 0
    correct = 0
    total = 0

    from functorch import make_functional, jacfwd, jacrev, vmap
    fnet, params = make_functional(model) #functorch requires a functional form of the model (fnet) with parameters (params) as an input to the model as well.
    per_sample_hessian = vmap(jacfwd(jacrev(fnet, argnums=0), argnums=0), in_dims=(None, 0)) #(params, x)

    with torch.no_grad():
        pbar = tqdm(enumerate(dataloader))
        for batch_idx, elem in pbar:
            if args.dataset in ['list', 'text']:
                inputs, targets, _ = elem
                inputs = inputs.float()[:, :, None]
                targets = targets#.int()
            else:
                inputs, targets = elem
            inputs, targets = inputs.to(device), targets.to(device)
            if args.dataset == 'dn':
                inputs = inputs.unsqueeze(2)
                targets = targets.unsqueeze(2)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            import pdb
            pdb.set_trace()

            for param_part in params:
                hessian = per_sample_hessian([param_part], inputs)

            eval_loss += loss.item()
            if args.dataset != 'dn':
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                pbar.set_description(
                    'Batch Idx: (%d/%d) | Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                    (batch_idx, len(dataloader), eval_loss/(batch_idx+1),
                     100.*correct/total, correct, total))
            else:
                checkpoint = False
                acc = 0
 
        wandb.log({'val_loss': eval_loss/(batch_idx+1), 
                   'val_acc': 100*correct/total})

        if test:
            wandb.log({'test_loss': eval_loss/(batch_idx+1), 
                        'test_acc': 100*correct/total})
        else:
            wandb.log({'val_loss': eval_loss/(batch_idx+1), 
                       'val_acc': 100*correct/total})

    acc = 100.*correct/total  
    return eval_loss/(batch_idx+1), acc

if args.p_ckpt is None:
    checkpoint_start = "/Users/ssiegel/mem-hippo/checkpoint/for_prune_0/baseline.pth"
else:
    checkpoint_start = "/Users/ssiegel/mem-hippo/checkpoint/" + args.p_ckpt

check1 = torch.load(checkpoint_start, map_location=device)

param_part = "s4_layer"



for n, ckpt in enumerate([check1]):

    if args.p_ckpt is None:
        res_file_name = "/Users/ssiegel/mem-hippo/evaluation/hessian/hessian_test"
    else:
        res_file_name = "/Users/ssiegel/mem-hippo/evaluation/hessian/hessian_" + args.p_ckpt
        if not os.path.isdir("/Users/ssiegel/mem-hippo/evaluation/hessian/"):
            os.mkdir("/Users/ssiegel/mem-hippo/evaluation/hessian/")

    res_file = open(res_file_name, "w")
    model.load_state_dict(ckpt['model'])
    loss, acc = eval(testloader, test=True)
    baseline_acc = acc
    res_file.close()

    from functorch import make_functional, jacfwd, jacrev, vmap

    #net = Model() #model instance
    fnet, params = make_functional(model) #functorch requires a functional form of the model (fnet) with parameters (params) as an input to the model as well.

    per_sample_hessian = vmap(jacfwd(jacrev(fnet, argnums=0), argnums=0), in_dims=(None, 0))(params, x)






