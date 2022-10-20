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


class CoordChange(nn.Module):
    """
    Spherical transformation followed by stereographic projection
    """

    def __init__(self, m=9):
        super(CoordChange, self).__init__()
        self.m = m  # number of classes minus 1

    def forward(self, x):
        mu = torch.sqrt(x)
        t = 2 * mu[:, :self.m] / (1 - mu[:, self.m].unsqueeze(1).repeat(1, self.m))
        return t


class JacobianReg(nn.Module):

    def __init__(self, epsilon):
        super(JacobianReg, self).__init__()
        self.epsilon = epsilon

    def forward(self, data, output):
        c = output.shape[1]
        m = c - 1

        # Coordinate change
        new_output = torch.sqrt(output)
        new_output = 2 * new_output[:, :m] / (1 - new_output[:, m].unsqueeze(1).repeat(1, m))

        # Compute Jacobian matrix
        jac = torch.zeros(m, *data.shape)
        grad_output = torch.zeros(*new_output.shape)
        for i in range(m):
            grad_output.zero_()
            grad_output[:, i] = 1
            jac[i] = torch.autograd.grad(new_output, data, grad_outputs=grad_output, retain_graph=True)[0]
        '''
        for i in range(m):
            if data.grad is not None:
                data.grad.zero_()
            grad_output.zero_()
            grad_output[:, i] = 1
            new_output.backward(grad_output, retain_graph=True)
            jac[i] = data.grad.data
        '''
        jac = torch.transpose(jac, dim0=0, dim1=1)
        jac = torch.reshape(jac, (jac.shape[0], -1))

        # Compute delta and rho
        delta = torch.sqrt(output/c).sum(dim=1)
        delta = 2*torch.acos(delta)
        rho = (2*(1-torch.sqrt(output[:, m])) - output[:, :m].sum(dim=1))/(1-torch.sqrt(output[:, m]))

        # Compute regularization
        reg = F.elu(torch.sqrt(torch.square(jac).sum(dim=1)) - delta/(rho*self.epsilon))
        return reg.mean()


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

    def __init__(self, param):
        super(Lenet, self).__init__()
        self.conv1 = nn.Conv2d(1, param['channels1'], 3, 1)
        self.conv2 = nn.Conv2d(param['channels1'], param['channels2'], 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, param['hidden'])
        self.fc2 = nn.Linear(param['hidden'], 10)

    def forward(self, x):
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
        output = F.log_softmax(x, dim=1)
        return output


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
