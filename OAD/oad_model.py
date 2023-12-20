import torch
import torch.nn as nn
from config import *

class MultiCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(MultiCrossEntropyLoss, self).__init__()

    def forward(self, pred, target):
        #IN: pred: unregularized logits [B, C] target: multi-hot representaiton [B, C]
        target_sum = torch.sum(target, dim=1)
        target_div = torch.where(target_sum != 0, target_sum, torch.ones_like(target_sum)).unsqueeze(1)
        target = target/target_div
        logsoftmax = nn.LogSoftmax(dim=1).to(pred.device)
        output = torch.sum(-target * logsoftmax(pred), 1)
        return torch.mean(output)

if __name__ == '__main__':
    criterion = MultiCrossEntropyLoss()
    criterion(torch.ones(2,3), torch.ones(2,3))


class OADModel(nn.Module):
    def __init__(self, class_agno):
        super(OADModel, self).__init__()
        self.class_agno = class_agno
        self.preprocess = nn.Sequential(
            nn.Linear(FEATURE_SIZE, LSTM_IN),
            nn.ReLU(),
        )
        self.dropout = nn.Dropout(DROPOUT_RATE)
        self.lstm = nn.LSTMCell(LSTM_IN, LSTM_HIDDEN)
        if class_agno:
            self.classifier = nn.Linear(LSTM_HIDDEN, 2)
        else:
            self.classifier = nn.Linear(LSTM_HIDDEN, NUM_CLASSES)

    def encode(self, feature_in:torch.FloatTensor, h:torch.FloatTensor, c:torch.FloatTensor):
        x = self.preprocess(feature_in)
        h, c = self.lstm(self.dropout(x), (h, c))
        score = self.classifier(self.dropout(h))
        return h, c, score

    def forward(self, feature_in:torch.FloatTensor):
        batch_size = feature_in.size(0)
        feature_len = feature_in.size(1)
        h = torch.zeros(batch_size, LSTM_HIDDEN).to(DEVICE)  # first h
        c = torch.zeros(batch_size, LSTM_HIDDEN).to(DEVICE)  # first c
        score_stack = []

        # Encoder -> time t
        for step in range(feature_len):
            h, c, score = self.encode(
                feature_in[:, step], h, c,
            )
            score_stack.append(score)
        if self.class_agno:
            enc_scores = torch.stack(score_stack, dim=1).view(-1, 2)
        else:
            enc_scores = torch.stack(score_stack, dim=1).view(-1, NUM_CLASSES)
        return enc_scores


