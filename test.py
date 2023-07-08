import os
import tqdm
import argparse
import json

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from model import ASLTransformer
from data import ASLDataset
from Levenshtein import distance

device = 'cpu'

def print_tensor(x, desc=""):
    if desc:
        print(f"{x.shape} {x.dtype} {x.min()} {x.max()}")
    else:
        print(f"{desc}: {x.shape} {x.dtype} {x.min()} {x.max()}")

def to_phrase(dic, index_lst):
    ret = ""
    for val in index_lst:
        val = val.item()
        if val < 2:
            pass
        else:
            ret += dic[val]
    return ret


def calculate_eval_val(out, gt, iw2):
    out = out.cpu()
    gt = gt.cpu()
    ld_val = 0
    word_cnt = 0
    for i in range(out.size(0)):
        gt_phrase = to_phrase(iw2, gt[i])
        out_phrase = to_phrase(iw2, out[i])
        levenshtein_distance = distance(out_phrase, gt_phrase)
        ld_val += levenshtein_distance
        word_cnt += len(gt_phrase)
    eval_val = 1 - (ld_val / word_cnt)
    
    return eval_val

@ torch.no_grad()
def test(model : nn.Module, test_loader : DataLoader, args : argparse.Namespace):
    with open("/data/asl-fingerspelling/character_to_prediction_index.json", 'r') as f:
        w2i = json.load(f)

    iw2 = dict()
    iw2[0] = ""
    iw2[1] = "<eos>"
    for k, v in w2i.items():
        iw2[v+2] = k
    args.iw2 = iw2

    for kps, phrase in test_loader:
        kps = kps.to(device)
        phrase = phrase.to(device)
        
        out = model(kps).cpu()
        out = torch.argmax(out, dim=-1)

        gt_phrase = to_phrase(iw2, phrase[0])
        out_phrase = to_phrase(iw2, out[0])
        accuracy = torch.sum(phrase==out)/(phrase.size(0)*phrase.size(1))
        eval = calculate_eval_val(out, phrase, iw2)
        print(out_phrase)
        print(gt_phrase)
        print(accuracy.item())
        print(eval)
        break

    

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()

    argparser.add_argument("--ckpt", required=True, type=str)

    args = argparser.parse_args()

    # model = ASLTransformer(
    #     y_dim=61, y_len=64, d_model=152, n_head=8, n_enc_layers=6, n_dec_layers=6, d_ff=512
    # ).to(device)

    model = ASLTransformer(
        y_dim=61, y_len=64, d_model=152, n_head=8, n_enc_layers=12, d_ff=1024
    ).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    train_dataset = ASLDataset(
        "/data/asl-fingerspelling", 
        split_path='data_val_info.txt',
        n_pad_x_seq=512, 
        n_pad_x_kps=152, 
        n_pad_y=64
    )
    # data loader
    test_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0
    )

    test(model, test_loader, args)