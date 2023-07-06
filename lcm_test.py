def lcm_test_01():
    import pandas as pd
    import os

    csv_file_path = "/data/asl-fingerspelling/train.csv"
    csv_file_path = "/data/asl-fingerspelling/train.csv"
    train_csv = pd.read_csv(csv_file_path)
    print(train_csv)
    print(train_csv.loc[:, 'phrase'].str.len().max())
    print(train_csv.loc[:, 'phrase'].str.len().min())

    # fpath = os.path.join('/data/asl-fingerspelling', train_csv.iloc[0]['path'])
    # print(train_csv.iloc[0]['sequence_id'])
    # print(train_csv.iloc[0]['phrase'])

    # parquet_data = pd.read_parquet(fpath)
    # columns = ['frame']
    # columns += [f'x_left_hand_{i}' for i in range(21)]
    # columns += [f'x_pose_{i}' for i in range(33)]
    # columns += [f'x_right_hand_{i}' for i in range(21)]
    # columns += [f'y_left_hand_{i}' for i in range(21)]
    # columns += [f'y_pose_{i}' for i in range(33)]
    # columns += [f'y_right_hand_{i}' for i in range(21)]

    # print(parquet_data[columns].loc[1817169529])
    

if __name__ == "__main__":
    import random

    with open('data_info.txt', 'r') as f:
        lines = f.read().strip().split("\n")
    
    random.shuffle(lines)

    with open('data_train_info.txt', 'w') as f1, open('data_val_info.txt', 'w') as f2:
        ind = int(len(lines)*0.9)
        for line in lines[:ind]:
            f1.write(line+"\n")
        for line in lines[ind:]:
            f2.write(line+"\n")
            
            
            