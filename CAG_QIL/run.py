import torch
import argparse
import os
import os.path as osp
import numpy as np

from env import Env
from dqn_utils import QNetwork
from config import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--action_q_len', type=int, default=24)
    parser.add_argument('--score_q_len', type=int, default=24)
    parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--load_model', type=str, default='1300000step_actor.pt')
    args = parser.parse_args()
    action_q_len = args.action_q_len
    score_q_len = args.score_q_len
    identifier = f'{score_q_len}_{action_q_len}'
    mode = args.mode
    if mode == 'eval':
        save_path = 'eval_result'
    elif mode == 'test':
        save_path = 'test_result'
    else:
        raise Exception('invalid mode')
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    load_model = args.load_model
    print('score_q_len: ', score_q_len)
    print('info_q_len: ', action_q_len)

    env = Env(action_q_len=action_q_len, score_q_len=score_q_len, mode=mode)
    model = QNetwork(action_q_len=24, score_q_len=24)
    model.eval()
    model.load_state_dict(torch.load(osp.join(f'models{identifier}', load_model)))
    obs = env.reset()
    current_filename = env.current_filename
    action = int(env.action_q[-1])
    action_list = [action]
    while True:
        action = model.get_action(obs, evaluation=True)
        action_list.append(int(action))
        obs, reward, done, info = env.step(action)
        if done:
            action = model.get_action(obs, evaluation=True)
            action_list.append(int(action))
            action_list = np.array(action_list, dtype=np.int64)
            np.save(osp.join(save_path, f'{current_filename}.npy'), action_list)
            print(current_filename, 'done, len: ', len(action_list))
            #next episode
            obs = env.reset()
            if env.owari:
                break
            current_filename = env.current_filename
            action = int(env.action_q[-1])
            action_list = [action]
    print('all done!')