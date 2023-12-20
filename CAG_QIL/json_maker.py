import numpy as np
import os
import os.path as osp
import json
import argparse
from config import *


def clustering(labels):
    #IN [000011100....]
    #OUT [[s1, e1], [s2, e2]...]
    segment_list = []
    prev_action = labels[0]
    if prev_action == 1:
        st = 0
    for i in range(1, len(labels)):
        action = labels[i]
        if prev_action == 0:
            if action == 1:
                #start
                st = i
        else:
            if action == 0:
                #end
                ed = i
                segment_list.append({"segment":[st, ed]})
        prev_action = action
    if action == 1:
        ed = i+1
        segment_list.append({"segment":[st, ed]})
    return segment_list

def make_contents(path):
    output = {}
    cnt = 0
    for filename in os.listdir(path):
        if 'test_0001292' in filename:
            continue
        if filename.endswith('.npy'):
            labels = np.load(osp.join(path, filename))
            clustered = clustering(labels)
            video_name = filename[:-4]
            if len(clustered) > 0:
                cnt += len(clustered)
                output[video_name] = clustered
            else:
                print(f'{filename}')
                output[video_name] = [{"segment": [0.,0.]}]
    print(f'proposal_num: {cnt}')
    return output

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--file_name', type=str, default='qil_test.json')
    args = parser.parse_args()

    mode = args.mode
    if mode == 'eval':
        path = 'eval_result'
    elif mode == 'test':
        path = 'test_result'
    else:
        raise Exception('invalid mode')
    filename = args.file_name

    contents = {}
    contents['results'] = make_contents(path)
    with open(osp.join(PROPOSAL_PATH, filename),'w') as fp:
        json.dump(contents, fp)

