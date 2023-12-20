import pickle
import os
import json
import numpy as np
from random import uniform, sample
from random import randint
from copy import deepcopy
from config import *

def get_range(length):
    '''
    get appropriate range for env
    '''
    if length < MAX_ENV_LEN:
        return 0, length
    start = uniform(0, 1-(MAX_ENV_LEN/length))
    end = start + (MAX_ENV_LEN/length)
    return int(start*length), int(end*length)

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

class Env():
    """Custom Environment that follows gym interface"""
    def __init__(self, action_q_len=24, score_q_len=24, mode='train'):
        self.mode = mode
        self.action_q_len = action_q_len
        self.score_q_len = score_q_len
        self.owari = False
        with open(ANNOTATION_PATH, 'r', encoding='utf-8') as f:
            self.meta = json.load(f)
        if mode == 'test':
            self.env_path = TEST_ENV_PATH
        elif mode == 'train':
            self.env_path = TRAIN_ENV_PATH
        elif mode == 'eval':
            self.env_path = TRAIN_ENV_PATH
        else:
            raise NotImplementedError('Invalid mode in environment')
        print('ENV_PATH: ', self.env_path)
        self.env_filelist = []
        for f in os.listdir(self.env_path):
            self.env_filelist.append(f[:-4])
        self.label_dict = dict()
        for filename in self.env_filelist:
            with open(f'{self.env_path}{filename}.npy', 'rb') as f:
                score_list = np.load(f)
            duration = len(score_list)
            with open(f'{LABEL_PATH}{filename}.npy', 'rb') as f:
                labels = np.load(f)
            assert duration == len(labels)
            label_seq = []
            for label in labels:
                d = np.argmax(label)
                if d == 0:
                    d = 0
                else:
                    d = 1
                label_seq.append(d)
            label_seq = np.array(label_seq, dtype=np.int64)
            self.label_dict[filename] = label_seq
        print('env setup complete')
        


    def step(self, action):
        '''
        action: 0 or 1 (int)
        '''
        action_answer = self.label_list[self.cur_idx]
        if action == 1:   
            if action_answer == 1:
                reward = 0.1
            else:
                reward = -0.1
        elif action == 0:
            if action_answer == 1:
                reward = -0.1
            else:
                reward = 0.1
        else:
            raise NotImplementedError()
        self.cur_idx += 1
        self.action_q.append(action)
        self.action_q.pop(0)
        self.score_q.append(self.score_list[self.cur_idx])
        self.score_q.pop(0)
        obs = (np.stack(self.score_q, axis=0).astype(np.float32), np.array(self.action_q, dtype=np.int64))
        done = self.cur_idx + 1 == len(self.score_list)
        return obs, reward, done, dict(gt_instances=self.gt_instances)

    def reset(self):
        if self.mode == 'train':
            filename = sample(self.env_filelist, 1)[0]
        elif self.mode == 'test' or self.mode == 'eval':
            if len(self.env_filelist) != 0:
                print('env_filelist length: ', len(self.env_filelist))
                filename = self.env_filelist.pop()
            else:
                self.owari = True
                #no more file available
                return None
        else:
            raise NotImplementedError('Invalid mode in environment')
        self.current_filename = filename
        with open(f'{self.env_path}{filename}.npy', 'rb') as f:
            score_list = np.load(f)

        if self.mode == 'test' or self.mode == 'eval':
            start, end = 0, len(score_list)
        else:
            start, end = get_range(len(score_list))
        self.score_list = score_list[start:end]
        self.label_list = self.label_dict[filename][start:end-1]
        self.gt_instances = get_instances(self.label_list)
        assert len(self.score_list) == len(self.label_list) + 1
        first_score = score_list[0]
        if first_score[0] > first_score[1]:
            first_action = 0
        else:
            first_action = 1
        self.score_q = [first_score for _ in range(self.score_q_len)]
        self.action_q = [first_action for _ in range(self.action_q_len)]
        self.cur_idx = 1
        self.score_q.pop(0)
        self.score_q.append(score_list[self.cur_idx])
        obs = (np.stack(self.score_q, axis=0).astype(np.float32), np.array(self.action_q, dtype=np.int64))
        return obs

    def render(self, mode='human'):
        print('render not supported')

    def close (self):
        pass

