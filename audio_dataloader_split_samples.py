import os
import glob
import torch
import numpy as np
import re
import soundfile as sf
from typing import Tuple, Dict, Any


class DNSAudio:
    """
        Customizable Audio dataset loader for N-DNS
        based on DNSAudio
        
        Parameters
        ----------
        root : str, optional
            Path of the dataset location, by default './'.
    """
    
    def __init__(self, args = None, root: str = './') -> None:
        self.root = root
        self.noisy_files = glob.glob(root + 'noisy/**.wav')
        self.file_id_from_name = re.compile('fileid_(\d+)')
        self.snr_from_name = re.compile('snr(-?\d+)')
        self.target_level_from_name = re.compile('tl(-?\d+)')
        self.source_info_from_name = re.compile('^(.*?)_snr')
        self.splitting_factor = args.splitting_factor
        self.window_size = args.window_size
        
    def _get_filenames(self, n: int) -> Tuple[str, str, str, Dict[str, Any]]:
        noisy_file = self.noisy_files[n % self.__len__()]
        filename = noisy_file.split(os.sep)[-1]
        file_id = int(self.file_id_from_name.findall(filename)[0])
        clean_file = self.root + f'clean/clean_fileid_{file_id}.wav'
        noise_file = self.root + f'noise/noise_fileid_{file_id}.wav'
        snr = int(self.snr_from_name.findall(filename)[0])
        target_level = int(self.target_level_from_name.findall(filename)[0])
        source_info = self.source_info_from_name.findall(filename)[0]
        metadata = {'snr': snr,
                    'target_level': target_level,
                    'source_info': source_info}
        return noisy_file, clean_file, noise_file, metadata

    def __getitem__(self, n: int) -> Tuple[np.ndarray,
                                           np.ndarray,
                                           np.ndarray,
                                           Dict[str, Any]]:
        """Gets the nth sample from the dataset.

        Parameters
        ----------
        n : int
            Index of the dataset sample.

        Returns
        -------
        np.ndarray
            Noisy audio sample.
        np.ndarray
            Clean audio sample.
        np.ndarray
            Noise audio sample.
        Dict
            Sample metadata.
        """
        i = n // self.splitting_factor
        j = n % self.splitting_factor
        noisy_file, clean_file, noise_file, metadata = self._get_filenames(i)
        noisy_audio, sampling_frequency = sf.read(noisy_file)
        clean_audio, _ = sf.read(clean_file)
        noise_audio, _ = sf.read(noise_file)
        num_samples = 30 * sampling_frequency // self.splitting_factor  # 30 sec data
        metadata['fs'] = sampling_frequency        
       
        if (j+1) * self.window_size < len(noisy_audio):
            noisy_audio = noisy_audio[j * self.window_size : (j+1) * self.window_size]
        else:
            noisy_audio = noisy_audio[j * self.window_size :]

        if (j+1) * self.window_size < len(clean_audio):
            clean_audio = clean_audio[j * self.window_size : (j+1) * self.window_size]            
        else:
            clean_audio = clean_audio[j * self.window_size :]
            
        if (j+1) * self.window_size < len(noise_audio):
            noise_audio = noise_audio[j * self.window_size : (j+1) * self.window_size]            
        else:
            noise_audio = noise_audio[j * self.window_size :]
            
        return noisy_audio, clean_audio, noise_audio, metadata

    def __len__(self) -> int:
        """Length of the dataset.
        """
        return len(self.noisy_files) * self.splitting_factor


if __name__ == '__main__':
    root = '/Data/pgi-15/datasets/intel_dns/datasets_fullband/'
    train_set = DNSAudio(
        root='../../data/MicrosoftDNS_4_ICASSP/training_set/')
    validation_set = DNSAudio(
        root='../../data/MicrosoftDNS_4_ICASSP/validation_set/')
