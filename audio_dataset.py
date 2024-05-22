import torchaudio
import os
from torch.utils.data import Dataset
import torch
import torchaudio.transforms as T

class HD(Dataset):
    def __init__(self, path, transform=None, target_transform=None, language="english", subset="all", subsample=1):
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

        ### Find max sampe length ###
        
        self.max_lenght = 6965
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
  
        #resampler = T.Resample(sample_rate, sample_rate / 10)
        #return resampler(sample_res), label
        return sample_res, label
    

#dataset = HD("./data/hd_audio", subset="test", language="english")
#dataset.__getitem__(12)

class HDold(Dataset):
    def __init__(self, audio_dir, transform=None, target_transform=None, language="english"):
        self.audio_dir = audio_dir
        dirlist = os.listdir(self.audio_dir)
        self.sample_info = []
        self.classes = []
        example_waveform, self.sample_rate = torchaudio.load(self.audio_dir + "/" + dirlist[0])
        print("Length sample: " + format(len(example_waveform[0]) / self.sample_rate) + " s")
        
        for elem in os.listdir(self.audio_dir):
            if language is not None:
                if not language in elem:
                    continue
            if elem.endswith("flac"):
                info = elem.replace(".flac", "")
                self.sample_info.append(info)
                self.classes.append(info.rsplit("-")[-1])

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.sample_info)

    def __getitem__(self, idx):
        audio_path = os.path.join(self.audio_dir, self.sample_info[idx] + ".flac")
        sample, sample_rate = torchaudio.load(audio_path)
        label = self.classes[idx]
        if self.transform:
            sample = self.transform(sample)
        if self.target_transform:
            sample = self.target_transform(label)
        return sample, label
