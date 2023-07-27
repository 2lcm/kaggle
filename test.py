import os
import tqdm
import argparse
import numpy as np
import importlib

import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont

from torch.utils.data import DataLoader

from data import ASLDataset, get_index2word
from utils import print_tensor, to_phrase, calculate_eval

device = 'cpu'

def kps_to_img(kps, gt, pred, sz=256, radius=2):
    out = [np.ones((sz, 2, 3), dtype=np.uint8)*255]
    lines = [(0,1), (1,2), (2,3), (3,4)]
    lines += [(0,5), (5,6), (6,7), (7,8)]
    lines += [(5,9), (9,10), (10,11), (11,12)]
    lines += [(9,13), (13,14), (14,15), (15,16)]
    lines += [(13,17), (0,17), (17,18), (18,19), (19,20)]

    for i in range(8):
        img = Image.new(mode='RGB', size=(sz, sz))
        draw = ImageDraw.Draw(img)
        kps_ = kps[i*8]
        for j in range(21):
            x, y = kps_[j]*sz, kps_[42+j]*sz
            x1, y1 = x-radius, y-radius
            x2, y2 = x+radius, y+radius
            draw.ellipse([(x1, y1), (x2, y2)], fill=(0, 255, 0))
        for j in range(21):
            x, y = kps_[21+j]*sz, kps_[63+j]*sz
            x1, y1 = x-radius, y-radius
            x2, y2 = x+radius, y+radius
            draw.ellipse([(x1, y1), (x2, y2)], fill=(0, 0, 255))
        for a, b in lines:
            p1 = kps_[a]*sz, kps_[42+a]*sz
            p2 = kps_[b]*sz, kps_[42+b]*sz
            p3 = kps_[21+a]*sz, kps_[63+a]*sz
            p4 = kps_[21+b]*sz, kps_[63+b]*sz
            draw.line([p1, p2], fill=(0, 255, 0), width=2)
            draw.line([p3, p4], fill=(0, 0, 255), width=2)
        
        out.append(np.array(img))
        out.append(np.ones((sz, 2, 3), dtype=np.uint8)*255)
    
    out.append(np.zeros((sz, sz*2, 3), dtype=np.uint8))
    out = np.concatenate(out, axis=1)
    img = Image.fromarray(out)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("arial.ttf", 30)
    draw.text(((sz+4)*8, sz/3), f"gt: {gt}", font=font)
    draw.text(((sz+4)*8, sz/3*2), f"pred: {pred}", font=font)
    img.save("test.jpg")
    raise NotImplementedError
    return img

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
        for ii, (kps, phrase_in, phrase_out) in enumerate(test_loader):
            if ii < 9:
                continue

            kps = kps.to(device)
            phrase_in = phrase_in.to(device)
            phrase_out = phrase_out.to(device)
            
            out = model.inference(kps).cpu()
            # out = torch.argmax(out, dim=-1)

            out_img = []
            for i in range(kps.size(0)):
                gt_phrase = to_phrase(i2w, phrase_out[i], validate=True)
                out_phrase = to_phrase(i2w, out[i], validate=True)
                accuracy = torch.sum(phrase_out==out)/(phrase_out.size(0)*phrase_out.size(1))
                eval = calculate_eval(out, phrase_out, i2w, validate=True)

                kps = kps.cpu().numpy()

                img = kps_to_img(kps[i], gt_phrase, out_phrase)
                out_img.append(img)

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
            x_len=512, 
            y_len=64
        )
    else:
        test_dataset = ASLDataset(
            "/data/asl-fingerspelling", 
            split_path='data_val_info.txt',
            x_len=512, 
            y_len=64
        )

    # data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    test(model, test_loader, args)