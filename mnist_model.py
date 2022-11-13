import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np


class RobustMnist(Dataset):

    def __init__(self, param):
        super(RobustMnist, self).__init__()
        self.data = np.zeros((60000, 1, 28, 28), dtype=np.float)
        self.target = np.zeros((60000,), dtype=np.int)
        for t in range(1000, 60001, 1000):
            data_array = np.load(param['robust_file']+f'_{t}.npy')
            target_array = np.load(param['target_file']+f'_{t}.npy')
            self.data[t-1000:t, 0, :] = data_array.copy()
            self.target[t-1000:t] = target_array.copy()

    def __len__(self):
        return self.target.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]


class SoftLeNet(nn.Module):

    def __init__(self, param, eps=1e-6):
        super(SoftLeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, param['channels1'], 3, 1)
        self.conv2 = nn.Conv2d(param['channels1'], param['channels2'], 3, 1)
        self.fc1 = nn.Linear(9216, param['hidden'])
        self.fc2 = nn.Linear(param['hidden'], 10)
        self.eps = eps

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = torch.maximum(F.softmax(x, dim=1), torch.tensor(self.eps))
        return torch.minimum(output, torch.tensor(1 - 9*self.eps))

class Lenet(nn.Module):

    def __init__(self, param, perform_softmax = True):
        super(Lenet, self).__init__()
        self.conv1 = nn.Conv2d(1, param['channels1'], 3, 1)
        self.conv2 = nn.Conv2d(param['channels1'], param['channels2'], 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, param['hidden'])
        self.fc2 = nn.Linear(param['hidden'], 10)

        self.perform_softmax = perform_softmax

    def forward(self, x):
        x = x.float()
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        logits = self.fc2(x)

        if self.perform_softmax:
            softmax_output = F.softmax(logits, dim = 1)
            return softmax_output

        else:
            return logits

class LogitLenet(nn.Module):

    def __init__(self, param):
        super(LogitLenet, self).__init__()
        self.conv1 = nn.Conv2d(1, param['channels1'], 3, 1)
        self.conv2 = nn.Conv2d(param['channels1'], param['channels2'], 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, param['hidden'])
        self.fc2 = nn.Linear(param['hidden'], 10)

    def forward(self, x):
        x = x.float()
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

class FeatureNet(Lenet):

    def __init__(self, param):
        super(FeatureNet, self).__init__(param)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        feature = F.relu(x)
        return feature
