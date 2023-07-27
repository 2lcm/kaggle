import os
import tqdm
import argparse
import wandb
import json
import importlib
import shutil
import time

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from data import ASLDataset, get_index2word
from utils import print_tensor, calculate_eval
from focal_loss import FocalLoss

device = 'cuda'

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

    # criterion = FocalLoss()
    criterion = nn.CrossEntropyLoss(ignore_index=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)

    pbar = tqdm.trange(args.steps, dynamic_ncols=True, initial=args.start_step)
    for step in pbar:
        model.train()
        kps, phrase_in, phrase_out = next(train_loader)

        kps = kps.to(device)
        phrase_in = phrase_in.to(device)
        phrase_out = phrase_out.to(device)

        out = model(kps, phrase_in)
        out_ = out.permute(0, 2, 1)

        loss = criterion(out_, phrase_out)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            model.eval()
            out = torch.argmax(out, dim=-1)
            train_accuracy = torch.sum(phrase_out==out)/(phrase_out.size(0)*phrase_out.size(1))
            train_eval = calculate_eval(out, phrase_out, args.i2w, validate=True)

            # validation dataset
            val_kps, val_gt_in, val_gt_out = next(val_loader)
            val_kps = val_kps.to(device)
            val_gt_in = val_gt_in.to(device)
            val_gt_out = val_gt_out.to(device)

            val_out = model(val_kps, val_gt_in)
            val_out = torch.argmax(val_out, dim=-1)
            val_accuracy = torch.sum(val_gt_out==val_out)/(val_gt_out.size(0)*val_gt_out.size(1))
            val_eval = calculate_eval(val_out, val_gt_out, args.i2w, validate=True)

            wandb_val = dict()
            if step % args.val_step == 0:
                train_inf_out = model.inference(kps)
                val_inf_out = model.inference(val_kps)

                train_inf_accuracy = torch.sum(phrase_out==train_inf_out)/(phrase_out.size(0)*phrase_out.size(1))
                train_inf_eval = calculate_eval(train_inf_out, phrase_out, args.i2w, validate=True)
                
                val_inf_accuracy = torch.sum(val_gt_out==val_inf_out)/(val_gt_out.size(0)*val_gt_out.size(1))
                val_inf_eval = calculate_eval(val_inf_out, val_gt_out, args.i2w, validate=True)
                
                if args.wandb:
                    wandb_val.update({
                        "train_inf_accuracy": train_inf_accuracy,
                        "train_inf_eval": train_inf_eval,
                        "val_inf_accuracy": val_inf_accuracy,
                        "val_inf_eval": val_inf_eval
                    })

            if step != 0 and step % args.save_step == 0:
                # save
                ckpt_path = f"checkpoints/{args.id}/{step:07}.pt"
                torch.save({
                    "model" : model.state_dict(),
                    "optimizer" : optimizer.state_dict(),
                    "step" : step
                    }, ckpt_path)


        pbar.set_description(
            f"loss:{loss.item():.4f} " + 
            f"train accuracy:{train_accuracy:.3f} " + 
            f"train eval:{train_eval:.3f} " + 
            f"val accuracy:{val_accuracy:.3f} " + 
            f"val eval:{val_eval:.3f}"
        )

        if args.wandb:
            wandb_val.update({
                "loss": loss.item(), 
                "train accuracy": train_accuracy,
                "train eval": train_eval,
                "val accuracy": val_accuracy,
                "val eval": val_eval
            })
            wandb.log(wandb_val)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()

    argparser.add_argument("--id", required=True, type=str)
    argparser.add_argument("--config", required=True, type=str)
    argparser.add_argument("--batch_size", default=32, type=int)
    argparser.add_argument("--val_batch_size", default=32, type=int)
    argparser.add_argument("--lr", default=0.0001, type=float)
    argparser.add_argument("--resume", action="store_true")
    argparser.add_argument("--load", default="", type=str)
    argparser.add_argument("--start_step", default=0, type=int)
    argparser.add_argument("--steps", default=2000000, type=int)
    argparser.add_argument("--val_step", default=10, type=int)
    argparser.add_argument("--save_step", default=1000, type=int)
    argparser.add_argument("--word2index", default="/data/asl-fingerspelling/character_to_prediction_index.json", type=str)
    argparser.add_argument("--wandb", action="store_true")
    

    args = argparser.parse_args()

    args.i2w = get_index2word(args.word2index)

    # id
    ckpt_path = f"checkpoints/{args.id}"
    os.makedirs(ckpt_path, exist_ok=True)
    config_dst_path = os.path.join(ckpt_path, 'config.py')
    shutil.copy2(args.config, config_dst_path)
    # model config
    config = args.config[:-3].replace("/", ".")
    config_py = importlib.import_module(f"{config}")
    model = config_py.ASLModel.to(device)

    if args.load:
        ckpt = torch.load(args.load, map_location=device)
        model.load_state_dict(ckpt['model'])
        if args.resume:
            args.start_step = ckpt['step']

    # dataloader
    train_dataset = ASLDataset(
            "/data/asl-fingerspelling", 
            split_path='data_train_info.txt',
            x_len=512, 
            y_len=64,
            aug=True
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
            x_len=512, 
            y_len=64,
            aug=False
        )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True
    )
    
    # wandb
    if args.wandb:
        wandb.init(project="asl", name=args.id, resume=args.resume)

    # start
    train(model, train_loader, val_loader, args)