import numpy as np
import torch
import torch.nn as nn


class CNNCTC(nn.Module):
    def __init__(self, class_num, mode='train'):
        super(CNNCTC, self).__init__()
        feature = [
            nn.Conv2d(3, 50, stride=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(50),
            nn.ReLU(inplace=True),
            nn.Conv2d(50, 100, stride=1, kernel_size=3, padding=1),
            nn.Dropout(p=0.1),
            nn.Conv2d(100, 100, stride=1, kernel_size=3, padding=1),
            nn.Dropout(p=0.1),
            nn.BatchNorm2d(100),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(100, 200, stride=1, kernel_size=3, padding=1),
            nn.Dropout(p=0.2),
            nn.Conv2d(200, 200, stride=1, kernel_size=3, padding=1),
            nn.Dropout(p=0.2),
            nn.BatchNorm2d(200),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(200, 250, stride=1, kernel_size=3, padding=1),
            nn.Dropout(p=0.3),
            nn.BatchNorm2d(250),
            nn.ReLU(inplace=True),
            nn.Conv2d(250, 300, stride=1, kernel_size=3, padding=1),
            nn.Dropout(p=0.3),
            nn.Conv2d(300, 300, stride=1, kernel_size=3, padding=1),
            nn.Dropout(p=0.3),
            nn.BatchNorm2d(300),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(300, 350, stride=1, kernel_size=3, padding=1),
            nn.Dropout(p=0.4),
            nn.BatchNorm2d(350),
            nn.ReLU(inplace=True),
            nn.Conv2d(350, 400, stride=1, kernel_size=3, padding=1),
            nn.Dropout(p=0.4),
            nn.Conv2d(400, 400, stride=1, kernel_size=3, padding=1),
            nn.Dropout(p=0.4),
            nn.BatchNorm2d(400),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2)
        ]

        classifier = [
            nn.Linear(1600, 900),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            # nn.Linear(900, 200),
            # nn.ReLU(inplace=True),
            nn.Linear(900, class_num)
        ]
        self.mode = mode
        self.feature = nn.Sequential(*feature)
        self.classifier = nn.Sequential(*classifier)

    def forward(self, x):  # x: batch, window, slice channel, h, w
        result = []
        for s in range(x.shape[1]):
            result.append(self.single_forward(x[:, s, :, :, :]))
        out = torch.stack(result)
        if self.mode != 'train':
            return self.decode(out)
        return out

    def single_forward(self, x):
        feat = self.feature(x)
        feat = feat.view(feat.shape[0], -1)  # flatten
        out = self.classifier(feat)
        return out

    def decode(self, pred):
        pred = pred.permute(1, 0, 2).cpu().data.numpy()  # batch, step, class
        seq = []
        for i in range(pred.shape[0]):
            seq.append(self.pred_to_string(pred[i]))
        return seq

    @staticmethod
    def pred_to_string(pred):  # step, class
        seq = []
        for i in range(pred.shape[0]):
            label = np.argmax(pred[i])
            seq.append(label)
        out = []
        for i in range(len(seq)):
            if len(out) == 0:
                if seq[i] != 0:
                    out.append(seq[i])
            else:
                if seq[i] != 0 and seq[i] != seq[i - 1]:
                    out.append(seq[i])
        return out
