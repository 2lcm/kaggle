import os
import tqdm
import argparse
import wandb
import json

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from model import ASLTransformer
from data import ASLDataset

from Levenshtein import distance

device = 'cuda'

def print_tensor(x, desc=""):
    if desc:
        print(f"{desc}: {x.shape} {x.dtype} {x.min()} {x.max()}")
    else:
        print(f"{x.shape} {x.dtype} {x.min()} {x.max()}")

def to_phrase(dic, index_lst):
    ret = ""
    for val in index_lst:
        val = val.item()
        if val < 3:
            pass
        else:
            ret += dic[val]
    return ret

def calculate_eval_val(x, y, iw2):
    x = x.cpu()
    y = y.cpu()
    ld_val = 0
    word_cnt = 0
    for i in range(x.size(0)):
        gt_phrase = to_phrase(iw2, x[i])
        out_phrase = to_phrase(iw2, y[i])
        levenshtein_distance = distance(out_phrase, gt_phrase)
        ld_val += levenshtein_distance
        word_cnt += len(gt_phrase)
    eval_val = 1 - (ld_val / word_cnt)
    
    return eval_val

def sample_data(loader):
    while True:
        for batch in loader:
            yield batch

def train(
        model : nn.Module, 
        train_loader : DataLoader, 
        val_loader : DataLoader, 
        args : argparse.Namespace
    ):

    train_loader = sample_data(train_loader)
    val_loader = sample_data(val_loader)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    pbar = tqdm.trange(args.steps, dynamic_ncols=True, initial=args.start_step)
    for step in pbar:
        model.train()
        kps, phrase = next(train_loader)
        kps = kps.to(device)
        phrase = phrase.to(device)        

        out = model(kps)

        # one_hot_phrase = nn.functional.one_hot(phrase, 62).to(device)
        # padding_mask = (one_hot_phrase==0)
        # print_tensor(padding_mask)
        # raise NotImplementedError

        out_ = out.permute(0, 2, 1)
        loss = criterion(out_, phrase)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            phrase = phrase.cpu()
            out = torch.argmax(out.cpu(), dim=-1)
            train_accuracy = torch.sum(phrase==out)/(phrase.size(0)*phrase.size(1))
            train_eval = calculate_eval_val(phrase, out, args.iw2)

            model.eval()
            val_kps, gt = next(val_loader)
            val_kps = val_kps.to(device)

            val_out = model(val_kps)
            val_out = torch.argmax(val_out.cpu(), dim=-1)
            val_accuracy = torch.sum(gt==val_out)/(gt.size(0)*gt.size(1))
            val_eval = calculate_eval_val(gt, val_out, args.iw2)

            pbar.set_description(
                f"loss:{loss.item():.4f} " + 
                f"train accuracy:{train_accuracy:.3f} " + 
                f"train eval:{train_eval:.3f} " + 
                f"val accuracy:{val_accuracy:.3f} " + 
                f"val eval:{val_eval:.3f}"
            )

            if args.wandb:
                wandb.log({
                    "loss": loss.item(), 
                    "train accuracy": train_accuracy,
                    "train eval": train_eval,
                    "val accuracy": val_accuracy,
                    "val eval": val_eval
                })

            if step != 0 and step % args.save_step == 0:
                ckpt_path = f"checkpoints/{args.id}/{step:07}.pt"
                torch.save({
                    "model" : model.state_dict(),
                    "optimizer" : optimizer.state_dict(),
                    "step" : step
                    }, ckpt_path)
                if args.wandb:
                    wandb.save(ckpt_path)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()

    argparser.add_argument("--id", required=True)
    argparser.add_argument("--batch_size", default=128, type=int)
    argparser.add_argument("--val_batch_size", default=32, type=int)
    argparser.add_argument("--lr", default=0.001, type=float)
    argparser.add_argument("--resume", action="store_true")
    argparser.add_argument("--load", default="", type=str)
    argparser.add_argument("--start_step", default=0, type=int)
    argparser.add_argument("--steps", default=600000, type=int)
    argparser.add_argument("--save_step", default=1000, type=int)
    argparser.add_argument("--word2index", default="/data/asl-fingerspelling/character_to_prediction_index.json", type=str)
    argparser.add_argument("--wandb", action="store_true")
    

    args = argparser.parse_args()

    # make index2word
    with open(args.word2index, 'r') as f:
        w2i = json.load(f)

    iw2 = dict()
    iw2[0] = ""
    iw2[1] = "<sos>"
    iw2[2] = "<eos>"
    for k, v in w2i.items():
        iw2[v+3] = k
    args.iw2 = iw2

    # id
    os.makedirs(f"checkpoints/{args.id}", exist_ok=True)

    # model
    model = ASLTransformer(
        y_dim=62, y_len=64, d_model=152, n_head=8, n_enc_layers=6, n_dec_layers=1, d_ff=512
    ).to(device)
    if args.load:
        ckpt = torch.load(args.load, map_location=device)
        model.load_state_dict(ckpt['model'])
        if args.resume:
            args.start_step = ckpt['step']

    # dataloader
    train_dataset = ASLDataset(
            "/data/asl-fingerspelling", 
            split_path='data_train_info.txt',
            n_pad_x_seq=512, 
            n_pad_x_kps=152, 
            n_pad_y=64
        )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )

    val_dataset = ASLDataset(
            "/data/asl-fingerspelling", 
            split_path='data_val_info.txt',
            n_pad_x_seq=512, 
            n_pad_x_kps=152, 
            n_pad_y=64
        )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )
    
    # wandb
    if args.wandb:
        wandb.init(project="asl", name=args.id, resume=args.resume)

    # start
    train(model, train_loader, val_loader, args)