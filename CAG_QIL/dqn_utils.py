import numpy as np
import random
import pickle
import scipy
import torch
import torch.nn as nn
import os.path as osp
from config import *
from collections import deque


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)

def one_hot_embedding(labels, num_classes):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    y = torch.eye(num_classes).to(DEVICE) 
    return y[labels]

class WeightedEmbedding(nn.Module):
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(vocab_size, dim))
        self.to(DEVICE)

    def forward(self, x):
        #IN: x [B, L, C]
        #C == vocab_size
        #OUT: [B, L, E]
        x = x.unsqueeze(3)
        #[B, L, C, 1]
        x = x*self.weight
        #[B, L, C, E]
        x = torch.sum(x, dim=2)
        #[B, L, E]
        return x

class Model(nn.Module):
    def __init__(self, q_len=24):
        super().__init__()
        self.actionness_embedding = WeightedEmbedding(2, ACTIONNESS_EMBEDDING_DIM)
        self.score_embedding = WeightedEmbedding(NUM_CLASSES, SCORE_EMBEDDING_DIM)
        self.action_embedding = nn.Embedding(2, ACTION_EMBEDDING_DIM)
        self.linears = nn.Sequential(
            nn.Linear(q_len*(ACTIONNESS_EMBEDDING_DIM + ACTION_EMBEDDING_DIM) + SCORE_EMBEDDING_DIM, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 2)
        )
        self.apply(weight_init)
        self.to(DEVICE)

    def forward(self, state):
        info_q, action_q = state
        batch_size = info_q.size(0)
        #info_process
        actionness_q = info_q[:, :, :2]
        score_q = info_q[:, :, 2:]
        embedded_actionness =self.actionness_embedding(actionness_q).view(batch_size, -1)
        embedded_score = self.score_embedding(score_q)[:, -1, :].view(batch_size, -1)
        embedded_action = self.action_embedding(action_q.long()).view(batch_size, -1)
        unified = torch.cat([embedded_actionness, embedded_score, embedded_action], dim=1)
        q_value = self.linears(unified)
        return q_value


class QNetwork(nn.Module):
    #Q-Network
    def __init__(self, action_q_len=24, score_q_len=24):
        super().__init__()
        self.action_q_len = action_q_len
        self.score_q_len = score_q_len
        self.eps = 1.0
        self.cnt = 0
        self.pred_network = Model(q_len=action_q_len)
        self.target_network = Model(q_len=action_q_len)
        self.opt = torch.optim.AdamW(self.pred_network.parameters(), lr=1e-4)
        self.apply(weight_init)
        self.synchronize_target()
        self.to(DEVICE)

    def synchronize_target(self):
        print('synchonizing target')
        self.target_network.load_state_dict(self.pred_network.state_dict())

    '''
    def get_action(self, s, deterministic=False):
        #s: Tuple(feature_q, action_q)
        self.eps -= 1e-6
        if self.eps < self.min_eps:
            self.eps = self.min_eps
        tmp = random.randrange(10000)
        if tmp < 10000*self.eps and not deterministic:
            return random.randrange(2)
        feature_q, action_q = s
        feature_q = torch.from_numpy(feature_q).to(DEVICE)
        action_q = torch.from_numpy(action_q).to(DEVICE)
        if len(feature_q.size()) < 3:
            feature_q = feature_q.unsqueeze(0)
            action_q = action_q.unsqueeze(0)
            s = (feature_q, action_q)
        else:
            print(feature_q.size())
            raise Exception()
        with torch.no_grad():
            q = self.pred_network(s)
            a = torch.argmax(q, dim=1).cpu().numpy().item()
            tmp = random.randrange(500)
            if tmp == 1:# and not deterministic:
                print('q')
                print(q)
            return a
    '''
    def get_action(self, s, evaluation=False, decrease_eps=True):
        #s: Tuple(score_q, action_q)
        if decrease_eps:
            self.eps -= 3e-6
        if self.eps < MIN_EPS:
            self.eps = MIN_EPS
        tmp = random.randrange(10000)
        if tmp < 10000*self.eps and not evaluation:
            return random.randrange(2)
        score_q, action_q = s
        score_q = torch.from_numpy(score_q).to(DEVICE)
        action_q = torch.from_numpy(action_q).to(DEVICE)
        if len(score_q.size()) < 3:
            score_q = score_q.unsqueeze(0)
            action_q = action_q.unsqueeze(0)
            s = (score_q, action_q)
        else:
            print(score_q.size())
            raise Exception()
        with torch.no_grad():
            q = self.pred_network(s)
            a = torch.argmax(q, dim=1).cpu().numpy().item()
            tmp = random.randrange(500)
            if tmp == 1 and not evaluation:
                print('q')
                print(q)
            return a

    def get_q(self, s, a):
        q = self.pred_network(s)
        a = one_hot_embedding(a, 2)
        q = torch.sum(q*a, dim=1)
        #print('q:', q)
        return q

    '''
    def get_target_q(self, s):
        target_q = torch.max(self.target_network(s), dim=1)[0]
        return target_q.detach()
    '''
    
    def get_target_q(self, s):
        target_q = torch.max(self.target_network(s), dim=1)[0]
        #print('targetq: ', target_q)
        return target_q.detach()
    

    def update(self, agent_memory, expert_memory):
        if len(agent_memory.sas) < LEARNING_START_LEN:
            return False
        agent_sas = agent_memory.get_batch(int(BATCH_SIZE*(1-SAMPLE_RATIO)))
        expert_sas = expert_memory.get_batch(int(BATCH_SIZE*SAMPLE_RATIO))
        states, actions, next_states = [list(i) for i in zip(*agent_sas)]
        expert_states, expert_actions, expert_next_states = [list(i) for i in zip(*expert_sas)]
        rewards = [-0.1 for _ in range(len(states))]
        expert_rewards = [0.1 for _ in range(len(expert_states))]
        states.extend(expert_states)
        score_q, action_q = [list(i) for i in zip(*states)]
        score_q = torch.from_numpy(np.stack(score_q, axis=0).astype(np.float32)).to(DEVICE)
        action_q = torch.from_numpy(np.array(action_q)).long().to(DEVICE)
        states = (score_q, action_q)
        actions.extend(expert_actions)
        actions = torch.LongTensor([int(i) for i in actions]).to(DEVICE)
        rewards.extend(expert_rewards)
        rewards = torch.FloatTensor(rewards).to(DEVICE)
        next_states.extend(expert_next_states)
        score_q, action_q = [list(i) for i in zip(*next_states)]
        score_q = torch.from_numpy(np.stack(score_q, axis=0).astype(np.float32)).to(DEVICE)
        action_q = torch.from_numpy(np.array(action_q)).long().to(DEVICE)
        next_states = (score_q, action_q)

        q = self.get_q(states, actions)
        target_q = self.get_target_q(next_states)
        td_error = torch.mean((q - (rewards + GAMMA*target_q))**2)
        self.opt.zero_grad()
        td_error.backward()
        self.opt.step()
        return True

class ExpertMemory:
    def __init__(self, transition_num:int=80000):
        with open(osp.join(EXPERT_TRAJ_PATH, 'expert_zeros_24_24.pkl'), 'rb') as f:
            zeros = pickle.load(f)
        len_zeros = len(zeros)
        assert len_zeros > transition_num*ZERO_RATIO
        random.shuffle(zeros)
        zeros = zeros[:int(ZERO_RATIO*transition_num)]

        with open(osp.join(EXPERT_TRAJ_PATH, 'expert_ones_24_24.pkl'), 'rb') as f:
            ones = pickle.load(f)
        len_ones = len(ones)
        assert len_ones > transition_num*ONE_RATIO
        random.shuffle(ones)
        ones = ones[:int(ONE_RATIO*len_ones)]
        zeros.extend(ones)
        self.expert_memory = zeros
        np.random.shuffle(self.expert_memory)
        self.cur_idx = 0
        print('expert memory successfully loaded')

    def get_batch(self, batch_size):
        if self.cur_idx + batch_size > len(self.expert_memory):
            np.random.shuffle(self.expert_memory)
            self.cur_idx = 0
        ret = self.expert_memory[self.cur_idx:self.cur_idx+batch_size]
        self.cur_idx += batch_size
        return ret


class AgentMemory:
    def __init__(self):
        self.sas = deque(maxlen=MAX_LEN)

    def append(self, state, action, next_state):
        self.sas.append([state, action, next_state])

    def get_batch(self, batch_size):
        assert len(self.sas) > batch_size
        tmp = list(range(len(self.sas)))
        random.shuffle(tmp)
        indice_arr = tmp[:batch_size]
        ret = []
        for i in indice_arr:
            ret.append(self.sas[i])
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
    iou = intersection/((l1 + l2 - intersection) + 1e-8)
    return iou

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

def get_hungarian_score(answer:list, prediction:list):
    #IN: answer[[st,ed], [st,ed]...], prediction[[st,ed],[st,ed]...]
    #OUT: tp(True positive), p(Positive), a(Answer)
    if len(answer) == 0:
        answer.append([0,0])
    if len(prediction) == 0:
        prediction.append([0,0])
    answer = np.array(answer)
    prediction = np.array(prediction)
    profit = np.zeros((len(answer), len(prediction)))
    for i in range(len(answer)):
        for j in range(len(prediction)):
            profit[i][j] = calculate_iou(answer[i], prediction[j])
    r, c = scipy.optimize.linear_sum_assignment(profit, maximize=True)
    tp = np.sum(np.where(profit[r, c] >= IOU_THRESHOLD, 1, 0))
    a = answer.shape[0]
    p = prediction.shape[0]
    return {'tp':tp, 'p':p, 'a':a}

def recent_mean(arr):
    if len(arr) < STAT_LEN:
        return np.mean(arr)
    else:
        return np.mean(arr[-STAT_LEN:])


if __name__ == '__main__':
    '''
    m = Model(q_len=6)
    dummy_score = torch.ones(4, 6, NUM_CLASSES).to(DEVICE)
    dummy_action = torch.zeros(4, 6).to(DEVICE)
    state = (dummy_score, dummy_action)
    print(m(state))
    '''
    exp = ExpertMemory()
    print(type(exp.get_batch(4)[0][0][1]))