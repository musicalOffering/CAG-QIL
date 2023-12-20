import argparse
import scipy
import json
import numpy as np
import os
import os.path as osp
from config import *
from collections import Counter

def get_instances(labels):
    #IN [000011100....]
    #OUT [[s1, e1], [s2, e2]...]
    ret = []
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
                ret.append([st, ed])
        prev_action = action
    if action == 1:
        ed = i+1
        ret.append([st,ed])
    return ret

def calculate_iou(prediction:list, answer:list):
    intersection = -1
    s1 = prediction[0]
    e1 = prediction[1]
    s2 = answer[0]
    e2 = answer[1]
    if s1 > s2:
        s1, s2 = s2, s1
        e1, e2 = e2, e1
    if e1 <= s2:
        intersection = 0
    else:
        if e2 <= e1:
            intersection = (e2 - s2)
        else:
            intersection = (e1 - s2)
    l1 = e1 - s1
    l2 = e2 - s2
    iou = intersection/(l1 + l2 - intersection)
    return iou



IOU_THRESHOLD = 0.5
#wrongly annotated videos
SUBTRACT_LIST = ['video_test_0000270', 'video_test_0001496',]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--pred_file', type=str, default='qil_test.json')
    parser.add_argument('--gt_file', type=str, default='thumos14_v2.json')
    args = parser.parse_args()
    mode = args.mode
    if mode == 'test':
        subset = 'test'
    elif mode == 'eval':
        subset = 'train'
    pred_file = args.pred_file
    gt_file = args.gt_file

    label_path = LABEL_PATH
    prediction_path = osp.join(PROPOSAL_PATH, pred_file)

    with open(gt_file, 'r', encoding='utf-8') as fp:
        answer_meta = json.load(fp)

    answer_list = []
    filenames = []
    for filename in answer_meta['database']:
        if answer_meta['database'][filename]['subset'] == subset:
            filenames.append(filename)
    for filename in filenames:
        name_seg = [filename]
        tmp = np.load(f'{label_path}{filename}.npy')
        labels = []
        for i in tmp:
            d = np.argmax(i)
            if d == 0:
                d = 0
            else:
                d = 1
            labels.append(d)
        seg_list = get_instances(labels)
        name_seg.append(seg_list)
        answer_list.append(name_seg)
    answer_list = sorted(answer_list, key=lambda a: a[0])

    
    with open(prediction_path, 'r', encoding='utf-8') as fp:
        prediction_meta = json.load(fp)
    counter = Counter()
    prediction_list = []
    for filename in prediction_meta['results']:
        name_seg = [filename]
        seg_list = []
        for chunk in prediction_meta['results'][filename]:
            st = chunk['segment'][0]
            ed = chunk['segment'][1]
            seg_list.append([st, ed])
            counter[(ed-st)] += 1
        name_seg.append(seg_list)
        prediction_list.append(name_seg)
    prediction_list = sorted(prediction_list, key=lambda a: a[0])
    cum_tp = 0
    cum_p = 0
    cum_a = 0
    name_tp = []
    for name_answer, name_prediction in zip(answer_list, prediction_list):
        if name_answer[0] != name_prediction[0]:
            print(name_answer[0], name_prediction[0])
        assert name_answer[0] == name_prediction[0]
        if name_answer[0] in SUBTRACT_LIST:
            continue
        prediction = np.array(name_prediction[1])
        answer = np.array(name_answer[1])

        profit = np.zeros((len(answer), len(prediction)))
        for i in range(len(answer)):
            for j in range(len(prediction)):
                profit[i][j] = calculate_iou(answer[i], prediction[j])
        r, c = scipy.optimize.linear_sum_assignment(profit, maximize=True)

        tp = np.sum(np.where(profit[r, c] >= IOU_THRESHOLD, 1, 0))
        name_tp.append((name_answer[0], tp))
        cum_tp += tp
        cum_a += answer.shape[0]
        cum_p += prediction.shape[0]
    print(prediction_path)
    print('IOU threshold: ', IOU_THRESHOLD)
    precision = cum_tp/cum_p
    recall = cum_tp/cum_a
    f1 = (2*precision*recall)/(precision + recall)
    print('precision, recall, f1 : ', precision, recall, f1)
    print(counter.most_common(20))
    name_tp = sorted(name_tp, key=lambda a: -a[1])
