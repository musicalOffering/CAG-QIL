import os
import os.path as osp
import argparse
import numpy as np
import pickle

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_path', type=str, default='rlenv_train')
    parser.add_argument('--label_path', type=str, default='label')
    parser.add_argument('--save_path', type=str, default='train_traj')
    parser.add_argument('--action_q_len', type=int, default=24)
    parser.add_argument('--score_q_len', type=int, default=24)
    args = parser.parse_args()
    env_path = args.env_path
    label_path = args.label_path
    savepath= args.save_path
    action_q_len = args.action_q_len
    score_q_len = args.score_q_len
    if not osp.exists(savepath):
        os.mkdir(savepath)
    for filename in os.listdir(env_path):
        vidname = filename[:-4]
        scores = np.load(osp.join(env_path, filename))
        num_classes = scores.shape[1]
        #[L, S]
        labels = np.load(osp.join(label_path, filename))
        #[L,]
        assert len(scores) == len(labels)
        oracle_action = np.zeros((len(labels)), dtype=np.int64)
        index = np.logical_not(labels[:,0]==1)
        oracle_action[index] = 1
        oracle_action = oracle_action.tolist()
        print('trajectory len: ' ,len(scores))
        score_q = []
        dummy_score = np.zeros(num_classes, dtype=np.float32)
        dummy_score[0] = 1.
        dummy_score[2] = 1.
        for _ in range(score_q_len):
            score_q.append(dummy_score)
        action_q = [0 for _ in range(action_q_len)]
        score_q.append(scores[0])
        score_q.pop(0)
        next_state = (np.stack(score_q, axis=0), np.array(action_q, dtype=np.int64))
        sas_list = []
        for i in range(len(scores)-1):
            state = next_state
            action = oracle_action[i]
            action_q.append(oracle_action[i])
            action_q.pop(0)
            score_q.append(scores[i+1])
            score_q.pop(0)
            next_state = (np.stack(score_q, axis=0), np.array(action_q, dtype=np.int64))
            sas_list.append([state, action, next_state])
        with open(osp.join(savepath, f'{vidname}.pkl'), 'wb') as f:
            pickle.dump(sas_list, f)