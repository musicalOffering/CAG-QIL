from config import *
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse
import os
import os.path as osp
from env import Env
from dqn_utils import QNetwork
from dqn_utils import ExpertMemory, AgentMemory
from dqn_utils import calculate_iou, recent_mean, get_hungarian_score, get_instances
from collections import Counter



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--action_q_len', type=int, default=24)
    parser.add_argument('--score_q_len', type=int, default=24)
    args = parser.parse_args()
    action_q_len = args.action_q_len
    score_q_len = args.score_q_len
    identifier = f'{score_q_len}_{action_q_len}'
    save_path = f'models{identifier}/'
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    #Networks
    qnet = QNetwork(action_q_len=action_q_len, score_q_len=score_q_len)
    #Memories
    expert_memory = ExpertMemory()
    agent_memory = AgentMemory()
    #Making Env
    env = Env(action_q_len=action_q_len, score_q_len=score_q_len)
    #simple stats
    score_arr = []
    score_mean_arr = []
    max_score = -1
    step_cnt = 0
    prev_save_cnt = 0
    episode_cnt = 0
    running_tp = 0
    running_p = 0
    running_a = 0
    updated = False
    while True:
        if step_cnt > MAX_STEPS:
            print('train ended!')
            break
        running_reward = 0
        d = False
        s = env.reset()
        action_q = [env.action_q[-1]]
        while not d:
            step_cnt += 1
            save_cnt = step_cnt % SAVE_CNT
            if prev_save_cnt > save_cnt:
                name = f'{step_cnt}step_actor.pt'
                torch.save(qnet.state_dict(), osp.join(save_path, name))
            prev_save_cnt = save_cnt
            if updated:
                a = qnet.get_action(s)
            else:
                a = qnet.get_action(s, decrease_eps=False)
            action_q.append(a)
            next_s, r, d, info = env.step(a)
            running_reward += r
            agent_memory.append(s, a, next_s)
            if step_cnt % Q_UPDATE_CNT == 0:
                updated = qnet.update(agent_memory, expert_memory)
            if step_cnt % TARGET_UPDATE_CNT == 0:
                qnet.synchronize_target()
            if d :
                assert len(action_q) == len(env.label_list)
                episode_cnt += 1
                gt_instances = info['gt_instances']
                pred_instances = get_instances(action_q)
                print('*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-')
                print(f'eps: {qnet.eps}')
                print(f'step_cnt:{step_cnt}')
                print(f'episode_len:{len(action_q)}')
                tmp = get_hungarian_score(gt_instances, pred_instances)
                running_tp += tmp['tp']
                running_p += tmp['p']
                running_a += tmp['a']
                c = Counter(action_q)
                print('0, 1:', c[0], c[1])
                print('reward: ', running_reward)

                if episode_cnt % PLOT_CNT == 0:
                    precision = running_tp/(running_p+1e-8)
                    recall = running_tp/(running_a+1e-8)
                    f1 = (2*precision*recall)/(precision+recall+1e-8)
                    print(f'f1 score: {f1}')
                    score_arr.append(f1)
                    running_a = 0
                    running_p = 0
                    running_tp = 0
                    
                    if len(score_arr) > STAT_LEN:
                        score_mean_arr.append(recent_mean(score_arr))
                        if max_score < recent_mean(score_arr):
                            max_score = recent_mean(score_arr)
                            name = f'{step_cnt}step_{max_score:.5f}_actor.pt'
                            torch.save(qnet.state_dict(), osp.join(save_path, name))

                    if len(score_mean_arr) > STAT_LEN:
                        plt.plot(np.arange(len(score_mean_arr)), score_mean_arr)
                        plt.savefig(f'stat{identifier}.png')
                    
            s = next_s