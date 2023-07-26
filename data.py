import os
import random
import numpy as np
import json

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

def print_tensor(x, desc=""):
    if desc:
        print(f"{desc}: {x.shape} {x.dtype} {x.min()} {x.max()}")
    else:
        print(f"{x.shape} {x.dtype} {x.min()} {x.max()}")


class ASLDataset(Dataset):
    def __init__(self, dirpath, split_path, x_len, y_len, aug=False):
        super().__init__()
        self.dirpath = dirpath
        self.split_path = split_path
        self.x_len = x_len
        self.y_len = y_len
        self.aug = aug

        try:
            with open(split_path, 'r') as f:
                lines = f.read().strip().split("\n")
            self.fnames = []
            for line in lines:
                fname, n_frames, n_words = line.split(" ")
                if int(n_frames) < self.x_len-1 and int(n_words) < self.y_len-1:
                    self.fnames.append(fname)
        except Exception as e:
            raise ValueError(f"File read fail : {self.split_path}", e)
    
    def __len__(self):
        return len(self.fnames)
    
    def __getitem__(self, index):
        fname = self.fnames[index]
        kps, phrase = torch.load(os.path.join(self.dirpath, fname))

        # kps
        x_kps = torch.cat([kps[:,469:490], kps[:,523:544]], dim=-1)
        y_kps = torch.cat([kps[:,1012:1033], kps[:,1066:1087]], dim=-1)

        # Data Augmentation
        if self.aug:
            # Horizontal Flip
            if random.random() < 0.5: 
                x_kps = 1- x_kps
            # Shift
            x_d = random.random() * 0.5 - 0.25
            y_d = random.random() * 0.5 - 0.25
            x_kps += x_d
            y_kps += y_d

        x_kps[x_kps>1] = -1
        x_kps[x_kps<0] = -1
        y_kps[y_kps>1] = -1
        y_kps[y_kps<0] = -1

        kps = torch.concat([x_kps, y_kps], dim=1)

        # if self.aug:
        #     kps[torch.rand_like(kps)<0.1] = -1

        pad_size_seq = self.x_len - kps.size(0)
        kps = F.pad(kps, (0, 0, 0, pad_size_seq), "constant", -1) # padding
        
        # phrase
        # ---------------------------------------------------------------#
        len_phrase = (len(phrase) + 2)
        pad_size_y = (self.y_len+1) - len_phrase
        phrase += 3

        phrase = F.pad(phrase, (1, 0), "constant", 1) # <sos>
        phrase = F.pad(phrase, (0, 1), "constant", 2) # <eos>
        phrase = F.pad(phrase, (0, pad_size_y), "constant", 0) # padding

        phrase_in = phrase[:-1].clone().detach()
        phrase_out = phrase[1:].clone().detach()
        
        return kps, phrase_in, phrase_out
    

def get_index2word(word2index_fpath):
    # make index2word
    with open(word2index_fpath, 'r') as f:
        w2i = json.load(f)

    i2w = dict()
    i2w[0] = ""
    i2w[1] = "<sos>"
    i2w[2] = "<eos>"
    for k, v in w2i.items():
        i2w[v+3] = k
    
    return i2w

if __name__ == "__main__":
    train_dataset = ASLDataset(
            "/data/asl-fingerspelling", 
            split_path='data_info.txt',
            x_len=256, 
            y_len=32,
            aug=True
        )
    a, b, c = train_dataset[0]
    print(len(train_dataset))
    
    print_tensor(a)
    print_tensor(b)
    print_tensor(c)