from typing import Tuple
import torchaudio
import os
from torch.utils.data import Dataset
import torch
import torchaudio.transforms as T

import numpy as np

import os

class HD(Dataset):
    def __init__(self, path, transform=None, target_transform=None, language="english", subset="all", subsample=1, classes=None):
        self.data_path = path + "/audio"
        self.transform = transform
        self.target_transform = target_transform
        self.subsample = subsample
        if subset=="test":
            self.sample_info = []
            with open(path + "/test_filenames.txt") as file:
                for line in file.readlines():
                    self.sample_info.append(line.replace("\n", ""))
        elif subset=="train":
            self.sample_info = []
            with open(path + "/train_filenames.txt") as file:
                for line in file.readlines():
                    self.sample_info.append(line.replace("\n", ""))
        elif subset=="all":
            self.sample_info = []
            with open(path + "/train_filenames.txt") as file:
                for line in file.readlines():
                    self.sample_info.append(line.replace("\n", ""))
            with open(path + "/test_filenames.txt") as file:
                for line in file.readlines():
                    self.sample_info.append(line.replace("\n", ""))

        if language=="all":
            pass
        else:
            self.sample_info = list(filter(lambda a: language in a, self.sample_info))

        if classes is not None:
            self.sample_info = list(filter(lambda a: any(a.endswith("-" + cl + ".flac") for cl in classes), self.sample_info))

        ### Find max sampe length ###
        
        self.max_lenght = int(6965*8/self.subsample + 1)
        '''
        for sample_name in self.sample_info:
            sample, sample_rate = torchaudio.load(self.data_path + "/" + sample_name)
            sample = sample[:, ::self.subsample]
            if sample.shape[1] > self.max_lenght:
                self.max_lenght = sample.shape[1]'''

        print("Max sample length:\t" + format(self.max_lenght))

        # padding


    def __len__(self):
        return len(self.sample_info)

    def __getitem__(self, idx):
        sample, sample_rate = torchaudio.load(self.data_path + "/" + self.sample_info[idx])
        label = int(self.sample_info[idx].rsplit("_")[-1].replace("digit-", "").replace(".flac", ""))
        if self.transform:
            sample = self.transform(sample)
        if self.target_transform:
            sample = self.target_transform(label)

        sample = sample[:, ::self.subsample]

        sample_res = torch.zeros((self.max_lenght, 1))
        sample_res[0:sample.shape[1], 0] = sample[0, :]

        max_abs = torch.max(torch.abs(sample))
        sample_res = sample_res / max_abs
        return sample_res, label


datapath = "/Users/ssiegel/datasets"


def export_to_numpy(dataset, folder):
    if not os.path.isdir(folder):
        os.mkdir(folder)
    for i in range(len(dataset)):
        item = dataset.__getitem__(i)
        np.save(folder + "/" + format(i) + "_raw_" + format(item[1]) + ".npy", item[0].numpy())
