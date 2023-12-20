import torch
import numpy as np
import json
import os
import os.path as osp
import argparse

from oad_model import OADModel
from config import *

if __name__ == '__main__':
    import sys
    np.set_printoptions(threshold=sys.maxsize)
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_model', type=str, default='class_agno_epoch_5_0.3831.pt')
    parser.add_argument('--subset', type=str, default='test')
    parser.add_argument('--save_path', type=str, default='./rlenv')
    parser.add_argument('--class_agno', type=str, default='True')
    args = parser.parse_args()
    class_agno = args.class_agno
    load_model = args.load_model
    subset = args.subset
    if class_agno == 'true' or class_agno == 'True':
        class_agno = True
    elif class_agno == 'false' or class_agno == 'False':
        class_agno = False
    else:
        raise Exception('invalid class_agno argument')
    if class_agno:
        load_model = osp.join(OAD_MODEL_SAVE_PATH, 'binary', load_model)
        save_path = f'{args.save_path}_{subset}_binary'
    else:
        load_model = osp.join(OAD_MODEL_SAVE_PATH, 'class', load_model)
        save_path = f'{args.save_path}_{subset}_class'
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    model = OADModel(class_agno).to(DEVICE)
    model.load_state_dict(torch.load(load_model))
    model.eval()

    with open(ANNOTATION_PATH, 'r', encoding='utf-8') as fp:
        meta = json.load(fp)
    vid_list = []
    for vidname in meta['database']:
        if(meta['database'][vidname]['subset'] == subset):
            vid_list.append(vidname)
    for vidname in vid_list:
        score_list = []
        feature = np.load(osp.join(FEATURE_PATH, f'{vidname}.npy'))
        duration = len(feature)
        h = torch.zeros(1, LSTM_HIDDEN).to(DEVICE)  # first h
        c = torch.zeros(1, LSTM_HIDDEN).to(DEVICE)  # first c
        for i in range(duration):
            with torch.no_grad():
                snippet = torch.from_numpy(feature[i]).unsqueeze(0).to(DEVICE)
                h, c, score = model.encode(snippet, h, c)
                score = torch.softmax(score, dim=1).squeeze().cpu().numpy()
                score_list.append(score)
        score_stack = np.stack(score_list, axis=0)
        with open(osp.join(save_path, f'{vidname}.npy'), 'wb') as fout:
            np.save(fout, score_stack)
        print(vidname, ' done')