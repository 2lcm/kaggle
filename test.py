import os
import tqdm
import argparse
import json
import importlib

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from data import ASLDataset, get_index2word
from utils import print_tensor, to_phrase, calculate_eval

device = 'cpu'

@ torch.no_grad()
def test(model : nn.Module, test_loader : DataLoader, args : argparse.Namespace):
    i2w = get_index2word("/data/asl-fingerspelling/character_to_prediction_index.json")

    if args.validate:
        N = len(test_loader)
        sum_acc = 0
        sum_eval = 0
        for kps, phrase_in, phrase_out in tqdm.tqdm(test_loader, total=N):
            kps = kps.to(device)
            phrase_in = phrase_in.to(device)
            phrase_out = phrase_out.to(device)
            
            out = model.inference(kps).cpu()
            # out = torch.argmax(out, dim=-1)

            gt_phrase = to_phrase(i2w, phrase_out[0], validate=True)
            out_phrase = to_phrase(i2w, out[0], validate=True)
            accuracy = torch.sum(phrase_out==out)/(phrase_out.size(0)*phrase_out.size(1))
            eval = calculate_eval(out, phrase_out, i2w, validate=True)
            sum_acc += accuracy
            sum_eval += eval
        print(f"Accuracy: {sum_acc/N}")
        print(f"Eval: {sum_eval/N}")
    else:
        for kps, phrase_in, phrase_out in test_loader:
            kps = kps.to(device)
            phrase_in = phrase_in.to(device)
            phrase_out = phrase_out.to(device)
            
            out = model.inference(kps).cpu()
            # out = torch.argmax(out, dim=-1)

            gt_phrase = to_phrase(i2w, phrase_out[0], validate=True)
            out_phrase = to_phrase(i2w, out[0], validate=True)
            accuracy = torch.sum(phrase_out==out)/(phrase_out.size(0)*phrase_out.size(1))
            eval = calculate_eval(out, phrase_out, i2w, validate=True)

            print(out_phrase)
            print(gt_phrase)
            print(accuracy.item())
            print(eval)
            break

    

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()

    argparser.add_argument("--ckpt", required=True, type=str)
    argparser.add_argument("--config", default="", type=str)
    argparser.add_argument("--validate", action="store_true")
    argparser.add_argument("--train", action="store_true")

    args = argparser.parse_args()

    if args.config == "":
        dirname = os.path.dirname(args.ckpt)
        args.config = os.path.join(dirname, "config.py")
    args.config = args.config[:-3].replace("/", ".")
    config_py = importlib.import_module(f"{args.config}")
    model = config_py.ASLModel.to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    if args.train:
        test_dataset = ASLDataset(
            "/data/asl-fingerspelling", 
            split_path='data_train_info.txt',
            x_len=256, 
            y_len=32
        )
    else:
        test_dataset = ASLDataset(
            "/data/asl-fingerspelling", 
            split_path='data_val_info.txt',
            x_len=256, 
            y_len=32
        )

    # data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0
    )

    test(model, test_loader, args)