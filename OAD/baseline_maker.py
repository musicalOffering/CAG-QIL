import torch
import numpy as np
import json
import os
import os.path as osp
import argparse

from config import *



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_path', type=str, default='rlenv_test_binary')
    parser.add_argument('--save_path', type=str, default='baseline')
    args = parser.parse_args()
    load_path = args.load_path
    if 'binary' in load_path:
        class_agno = True
    else:
        class_agno = False
    print(f'class_agno: {class_agno}')
    save_path = f'{args.save_path}'
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    file_list = os.listdir(load_path)
    for filename in file_list:
        vidname = filename[:-4]
        score = np.load(osp.join(load_path, filename))
        d_list = []
        if class_agno:
            for row in score:
                d = np.argmax(row)
                d_list.append(d)
        else:
            for row in score:
                d = np.argmax(row)
                if d == 0:
                    d = 0
                else:
                    d = 1
                d_list.append(d)
        with open(osp.join(save_path, filename), 'wb') as f:
            np.save(f, np.array(d_list, dtype=np.float32))

    