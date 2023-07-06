import os
import glob
import json
import time
import numpy as np
import pandas as pd
import tqdm
import torch
from multiprocessing import Pool

dirpath = "/data/asl-fingerspelling"

train_fpath = os.path.join(dirpath, "train.csv")
train_csv = pd.read_csv(train_fpath)
train_pd = train_csv[['sequence_id', 'phrase']]

sup_fpath = os.path.join(dirpath, "supplemental_metadata.csv")
sup_csv = pd.read_csv(sup_fpath)
sup_pd = sup_csv[['sequence_id', 'phrase']]
df = pd.concat([train_pd, sup_pd])

df = df.set_index('sequence_id')

with open(os.path.join(dirpath, 'character_to_prediction_index.json'), 'r') as f:
    ch_dict = json.load(f)

def func(fname):
    columns = [f'x_left_hand_{i}' for i in range(21)]
    columns += [f'x_right_hand_{i}' for i in range(21)]
    columns += [f'y_left_hand_{i}' for i in range(21)]
    columns += [f'y_right_hand_{i}' for i in range(21)]

    parquet_data = pd.read_parquet(fname)
    for seq_id in list(set(parquet_data.index)):
        seq = parquet_data.loc[seq_id]
        np_seq = seq.to_numpy()
        np_seq = np.nan_to_num(np_seq, nan=-1)
        torch_seq = torch.from_numpy(np_seq)

        phrase = df.loc[seq_id]['phrase']
        phrase_encoded = []
        for c in phrase:
            phrase_encoded.append(ch_dict[c])

        torch_phrase = torch.tensor(phrase_encoded)

        save_path = os.path.join(dirpath, f'processed/{seq_id}.pt')
        torch.save((torch_seq, torch_phrase), save_path)


if __name__ == "__main__":
    

    fnames = glob.glob(os.path.join(dirpath, "train_landmarks", '*'))
    fnames += glob.glob(os.path.join(dirpath, "supplemental_landmarks", '*'))
    pool = Pool(1)

    for i in tqdm.tqdm(pool.imap_unordered(func, fnames), total=len(fnames)):
        pass
    
    

    