import torch
import torch.nn as nn
import argparse
import numpy as np
import os
import os.path as osp
from tqdm import tqdm

from oad_model import OADModel
from oad_model import MultiCrossEntropyLoss
from dataset import THUMOS14_Train, THUMOS14_Test
from config import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--eval_epoch', type=int, default=10)
    parser.add_argument('--model_name', type=str, default='class')
    args = parser.parse_args()
    epoch = args.epoch
    batch_size = args.batch_size
    eval_epoch = args.eval_epoch
    model_name = args.model_name
    if model_name == 'class':
        class_agno = False
    elif model_name == 'binary':
        class_agno = True
    else:
        raise Exception('invalid model_name arguement: choose it from `class` or `binary`')
    if not osp.isdir(OAD_MODEL_SAVE_PATH):
        os.mkdir(OAD_MODEL_SAVE_PATH)
    if not osp.isdir(osp.join(OAD_MODEL_SAVE_PATH, model_name)):
        os.mkdir(osp.join(OAD_MODEL_SAVE_PATH, model_name))

    trainset = THUMOS14_Train(class_agno)
    testset = THUMOS14_Test(class_agno)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)
    model = OADModel(class_agno).to(DEVICE)
    criterion = MultiCrossEntropyLoss()
    eval_criterion = MultiCrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4)
    epoch_cnt = 0
    for e in range(epoch):
        print('epoch', e+1)
        model.train()
        cumulated_loss = 0
        cnt = 0
        for sample in tqdm(train_loader):
            data = sample['data'].to(DEVICE)
            label = sample['label'].to(DEVICE)
            if class_agno:
                label = label.view(-1, 2)
            else:
                label = label.view(-1, NUM_CLASSES)
            score = model(data)
            loss = criterion(score, label)
            loss_val = loss.detach().cpu().numpy()
            if loss_val > 100:
                raise Exception('numerical unstability')
            cnt += 1
            cumulated_loss += loss_val
            opt.zero_grad()
            loss.backward()
            opt.step()
        print('in epoch', e+1, ' loss_avg: ', cumulated_loss/cnt)
        epoch_cnt += 1
        if epoch_cnt == eval_epoch:
            print('start evaluating after epoch', e+1)
            epoch_cnt = 0
            eval_cumulated_loss = 0 
            cnt = 0
            model.eval()
            for sample in tqdm(test_loader):
                data = sample['data'].to(DEVICE)
                label = sample['label'].to(DEVICE)
                if class_agno:
                    label = label.view(-1, 2)
                else:
                    label = label.view(-1, NUM_CLASSES)
                with torch.no_grad():
                    score = model(data)
                    loss = eval_criterion(score, label)
                loss_val = loss.detach().cpu().numpy()
                eval_cumulated_loss += loss_val
                cnt += 1
            print('eval_loss: ', eval_cumulated_loss/cnt)
            epoch_info = f'epoch_{e+1}_'
            tmp = str(eval_cumulated_loss/cnt)[:6]
            torch.save(model.state_dict(), f'{OAD_MODEL_SAVE_PATH}{model_name}/{epoch_info}{tmp}.pt')

