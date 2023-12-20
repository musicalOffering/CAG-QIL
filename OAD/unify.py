import os
import argparse
import numpy as np
import os.path as osp


parser = argparse.ArgumentParser()
parser.add_argument('--binary_path', type=str, default='rlenv_train_binary')
parser.add_argument('--class_path', type=str, default='rlenv_train_class')
parser.add_argument('--save_path', type=str, default='rlenv_train')
args = parser.parse_args()

path1 = args.binary_path
path2 = args.class_path
savepath = args.save_path

if not osp.exists(savepath):
    os.mkdir(savepath)

names = os.listdir(path1)
for name in names:
    actionness_arr = np.load(osp.join(path1, name))
    score_arr = np.load(osp.join(path2, name))
    assert len(actionness_arr) == len(score_arr)
    full_arr = np.concatenate((actionness_arr, score_arr), axis=1)
    np.save(osp.join(savepath, name), full_arr)
    print(full_arr.shape)
