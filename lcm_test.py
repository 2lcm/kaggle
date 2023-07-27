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
    
def lcm_test_02():
    import cv2
    import numpy as np

    img1 = cv2.imread("test_04.jpg", cv2.IMREAD_COLOR)
    img2 = cv2.imread("test_06.jpg", cv2.IMREAD_COLOR)
    img3 = cv2.imread("test_07.jpg", cv2.IMREAD_COLOR)
    img4 = cv2.imread("test_09.jpg", cv2.IMREAD_COLOR)

    img = np.concatenate([img1, img2, img3, img4], axis=0)
    img = cv2.resize(img, dsize=None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    cv2.imwrite("test.jpg", img)

if __name__ == "__main__":
    # raise NotImplementedError
    lcm_test_02()
    