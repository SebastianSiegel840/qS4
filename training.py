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

ENERGY_MONITOR = True

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
parser.add_argument('--name', type=str, default='example', help='wandb run name')
parser.add_argument('--project', type=str, default='quant SSM', help='wandb project name')
parser.add_argument('--wb', type=str, default='disabled', help='wandb mode: online, offline, disabled')
# Optimizer
parser.add_argument('--lr', default=0.01, type=float, help='Learning rate')
parser.add_argument('--weight_decay', default=0.05, type=float, help='Weight decay')
# Scheduler
# parser.add_argument('--patience', default=10, type=float, help='Patience for learning rate scheduler')
parser.add_argument('--epochs', default=100, type=int, help='Training epochs')
# Dataset
parser.add_argument('--dataset', default='hd', choices=['mnist', 'cifar10', 'hd', 'dn', 'pathfinder'], type=str, help='Dataset')
parser.add_argument('--grayscale', action='store_true', help='Use grayscale CIFAR10')
parser.add_argument('--subsample', default=1,type=int, help='specify subsampling ratio')
# Dataloader
parser.add_argument('--num_workers', default=4, type=int, help='Number of workers to use for dataloader')
parser.add_argument('--batch_size', default=64, type=int, help='Batch size')
# Model # SEE models/
#parser.add_argument('--n_layers', default=4, type=int, help='Number of layers')
#parser.add_argument('--d_model', default=128, type=int, help='Model dimension')
#parser.add_argument('--dropout', default=0.1, type=float, help='Dropout')
#parser.add_argument('--prenorm', action='store_true', help='Prenorm')
# General
parser.add_argument('--resume', '-r', action='store_true', help='Resume from checkpoint')
parser.add_argument('--model_file', type=str, default='s4_model',  help='which model file to use')
parser.add_argument('--net', default="S4Model", type=str, help='which model class to use')
parser.add_argument('--splitting_factor', type=int, default=1, help='Number of chunks to divide the initial samples')
parser.add_argument('--gpu', type=int, default=[3], help='which gpu(s) to use', nargs='+')
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

parser.add_argument('--check_path', default=None)

parser_args = parser.parse_args()

model_lib = __import__(parser_args.model_file)

args = getattr(model_lib, 'return_args')(parser_args) # Network specific configs

setattr(args, 'device', args.gpu[0]) if len(args.gpu)==1 else None

device = torch.device('cuda:{}'.format(args.gpu[0]))

if args.enery:
    from zeus.monitor import ZeusMonitor
    monitor = ZeusMonitor(gpu_indices=[args.gpu[0]])

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# create a wandb session 
wandb_run = wandb.init(project=args.project,
                        name=args.name,
                        config=args,
                        mode=args.wb)
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
if args.dataset == "pathfinder":
    trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    valloader = torch.utils.data.DataLoader(
            valset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    testloader = torch.utils.data.DataLoader(
            testset, batch_size=args.batch_size, shuffle=True, drop_last=True)
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
    'coder_quant': args.coder_quant
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
        'coder_quant': args.all_quant
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
        'coder_quant': args.coder_quant
    }
'''  
if args.A_quant is not None:
    model_args['A_quant'] = args.A_quant
if args.B_quant is not None:
    model_args['B_quant'] = args.B_quant
if args.C_quant is not None:
    model_args['C_quant'] = args.C_quant
if args.dt_quant is not None:
    model_args['dt_quant'] = args.dt_quant
if args.kernel_quant is not None:
    model_args['kernel_quant'] = args.kernel_quant
if args.linear_quant is not None:
    model_args['linear_quant'] = args.linear_quant
if args.act_quant is not None:
    model_args['act_quant'] = args.act_quant
if args.coder_quant is not None:
    model_args['coder_quant'] = args.coder_quant'''

for arg in ['A_quant', 'C_quant', 'B_quant', 'dt_quant', 'kernel_quant', 'linear_quant', 'act_quant', 'coder_quant']:
    if model_args[arg] == 'None':
        model_args[arg] = None

# Model
print('==> Building model..')
model = getattr(model_lib, args.net)(args, d_input, d_output, **model_args).to(device) # Please ensure that your model takes arguments (args, dim_in, dim_out) with args a class object with network config., model_args=model_args

##############################################
#### Save initial state ######################
state = {
    'model': model.state_dict(),
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
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    model.load_state_dict(checkpoint['model'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

def setup_optimizer(model, lr, weight_decay, epochs):
    """
    S4 requires a specific optimizer setup.

    The S4 layer (A, B, C, dt) parameters typically
    require a smaller learning rate (typically 0.001), with no weight decay.

    The rest of the model can be trained with a higher learning rate (e.g. 0.004, 0.01)
    and weight decay (if desired).
    """

    # All parameters in the model
    all_parameters = list(model.parameters())

    # General parameters don't contain the special _optim key
    params = [p for p in all_parameters if not hasattr(p, "_optim")]

    # Create an optimizer with the general parameters
    optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    # Add parameters with special hyperparameters
    hps = [getattr(p, "_optim") for p in all_parameters if hasattr(p, "_optim")]
    hps = [
        dict(s) for s in sorted(list(dict.fromkeys(frozenset(hp.items()) for hp in hps)))
    ]  # Unique dicts
    for hp in hps:
        params = [p for p in all_parameters if getattr(p, "_optim", None) == hp]
        optimizer.add_param_group(
            {"params": params, **hp}
        )

    # Create a lr scheduler
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=0.2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    # Print optimizer info
    keys = sorted(set([k for hp in hps for k in hp.keys()]))
    for i, g in enumerate(optimizer.param_groups):
        group_hps = {k: g.get(k, None) for k in keys}
        print(' | '.join([
            f"Optimizer group {i}",
            f"{len(g['params'])} tensors",
        ] + [f"{k} {v}" for k, v in group_hps.items()]))

    return optimizer, scheduler

criterion = nn.CrossEntropyLoss()
#optimizer, scheduler = setup_optimizer(
#    model, lr=args.lr, weight_decay=args.weight_decay, epochs=args.epochs
#)
optimizer = torch.optim.Adam(model.parameters(),
                             lr=args.lr,
                             weight_decay=args.weight_decay)  # 1e-2

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

# Training
def train():
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    pbar = tqdm(enumerate(trainloader))
    for batch_idx, (inputs, targets) in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        if args.dataset == 'dn':
            inputs = inputs.unsqueeze(2)
            targets = targets.unsqueeze(2)

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        if args.dataset != 'dn':
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pbar.set_description(
                'Batch Idx: (%d/%d) | Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                (batch_idx, len(trainloader), train_loss/(batch_idx+1),
                 100.*correct/total, correct, total))
            
            wandb.log({'loss': train_loss/(batch_idx+1), 
                       'acc': 100*correct/total})


def eval(epoch, dataloader, checkpoint=False):
    global best_acc
    model.eval()
    eval_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        pbar = tqdm(enumerate(dataloader))
        for batch_idx, (inputs, targets) in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            if args.dataset == 'dn':
                inputs = inputs.unsqueeze(2)
                targets = targets.unsqueeze(2)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

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

    # Save checkpoint.
    if checkpoint:
        acc = 100.*correct/total
        if acc > best_acc:
            state = {
                'model': model.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if args.check_path is None:
                torch.save(state, './checkpoint/ckpt' + format(num_ckpt) + '.pth') #_sub' + format(args.subsample) + '_batch' + format(args.batch_size) + '.pth')
            else:
                torch.save(state, './checkpoint/' + args.check_path + '/ckpt' + format(num_ckpt) + '.pth') #_sub' + format(args.subsample) + '_batch' + format(args.batch_size) + '.pth')
            
            best_acc = acc

            if not os.path.isdir('checkpoint/layerstxt'):
                os.mkdir('checkpoint/layerstxt')
            for layer in state['model']:
                    #print(layer)
                    #print(state['model'][layer])
                    save = state['model'][layer].cpu().numpy()
                    if len(save.shape) < 3 and len(save.shape) > 0:
                        np.savetxt('checkpoint/layerstxt/' + layer + '.vcsv', save, fmt='%5.4f')
                    elif len(save.shape) == 3:
                        for i in range(len(save[0, 0, :])):
                            np.savetxt('checkpoint/layerstxt/' + layer + '_' + format(i) + '.vcsv', save[:, :, i], fmt='%5.4f')
                    elif len(save.shape) == 4:
                        for i in range(len(save[0, 0, 0, :])):
                            np.savetxt('checkpoint/layerstxt/' + layer + '_' + format(i) + '.vcsv', save[0, :, :, i], fmt='%5.4f')


        return acc

if args.energy:
    monitor.begin_window("training")

pbar = tqdm(range(start_epoch, args.epochs))
best_acc = 0
for epoch in pbar:

    print("Epoch", epoch)
    wandb.log({'epoch': epoch})

    start = time.time()
    train()
    print("train time", time.time() - start)
    print("Validation ...")
    val_acc = eval(epoch, valloader, checkpoint=True)
    if val_acc > best_acc:
        best_acc = val_acc
    print("Testing ...")
    eval(epoch, testloader)
    # scheduler.step()
    # print(f"Epoch {epoch} learning rate: {scheduler.get_last_lr()}")

if args.check_path is not None:
    file = open('./checkpoint/' + args.check_path + '/val_acc', "a+")
    if args.all_quant is not None:
        file.write("all\t" + format(args.all_quant) + "\t" + format(best_acc) + "\n")
        file.close()
        exit()
    else:
        for param in model_args:
            if model_args[param] is not None:
                file.write(param + "\t" + format(model_args[param]) + "\t" + format(best_acc) + "\n")
                file.close()
                exit()
    file.write("baseline\tfloat\t" + format(best_acc) + "\n")
    file.close()

if args.energy:
    meas_total = monitor.end_window("training")
    print(f"Total energy consumption of training run: {meas_total.total_energy / 3.6e6} kWh")
