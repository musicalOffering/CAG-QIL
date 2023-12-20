import numpy as np
import pickle
import pickle
import os
import os.path as osp
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--action_q_len', type=int, default=24)
    parser.add_argument('--score_q_len', type=int, default=24)
    parser.add_argument('--save_path', type=str, default='train_traj')
    args = parser.parse_args()
    action_q_len = str(args.action_q_len)
    score_q_len = str(args.score_q_len)
    savepath = args.save_path
    zeros = []
    ones = []
    for filename in os.listdir(savepath):
        path = osp.join(savepath, filename)
        with open(path, 'rb') as f:
            e = pickle.load(f)
        for i in range(len(e)):
            action = e[i][1]
            if action == 1:
                ones.append(e[i])
            else:
                zeros.append(e[i])

    with open(f'expert_ones_{score_q_len}_{action_q_len}.pkl', 'wb') as f:
        pickle.dump(ones, f)

    with open(f'expert_zeros_{score_q_len}_{action_q_len}.pkl', 'wb') as f:
        pickle.dump(zeros, f)
