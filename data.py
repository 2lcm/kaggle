import os
import glob
import time
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

def print_tensor(x, desc=""):
    if desc:
        print(f"{desc}: {x.shape} {x.dtype} {x.min()} {x.max()}")
    else:
        print(f"{x.shape} {x.dtype} {x.min()} {x.max()}")


class ASLDataset(Dataset):
    def __init__(self, dirpath, split_path, n_pad_x_seq, n_pad_x_kps, n_pad_y):
        super().__init__()
        self.dirpath = dirpath
        self.split_path = split_path
        self.n_pad_x_seq = n_pad_x_seq
        self.n_pad_x_kps = n_pad_x_kps
        self.n_pad_y = n_pad_y

        try:
            with open(split_path, 'r') as f:
                lines = f.read().strip().split("\n")
            self.fnames = []
            for line in lines:
                fname, n_frames, n_words = line.split(" ")
                if int(n_frames) < 512 and int(n_words) < 64:
                    self.fnames.append(fname)
        except Exception as e:
            raise ValueError(f"File read fail : {self.split_path}", e)
    
    def __len__(self):
        return len(self.fnames)
    
    def __getitem__(self, index):
        fname = self.fnames[index]
        kps, phrase = torch.load(os.path.join(self.dirpath, fname))
        kps = torch.concat([kps[:,469:544], kps[:,1012:1087]], dim=1)
        pad_size_seq = self.n_pad_x_seq - (kps.size(0) + 1)
        pad_size_kps = self.n_pad_x_kps - kps.size(1)

        kps = F.pad(kps, (0, pad_size_kps), "constant", -1)
        kps = F.pad(kps, (0, 0, 1, 0), "constant", 2)
        kps = F.pad(kps, (0, 0, 0, pad_size_seq), "constant", 3)

        pad_size_y = self.n_pad_y - (len(phrase) + 2)
        phrase += 3
        phrase = F.pad(phrase, (1, 0), "constant", 1) # <sos>
        phrase = F.pad(phrase, (0, 1), "constant", 2) # <eos>
        phrase = F.pad(phrase, (0, pad_size_y), "constant", 0) # padding

        return kps, phrase

if __name__ == "__main__":
    train_dataset = ASLDataset(
            "/data/asl-fingerspelling", 
            split_path='data_info.txt',
            n_pad_x_seq=512, 
            n_pad_x_kps=152, 
            n_pad_y=64
        )
    a, b = train_dataset[0]
    
