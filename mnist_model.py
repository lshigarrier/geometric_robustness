import torch
import torch.nn as nn
import torch.nn.functional as F


class IsometryReg(nn.Module):

    def __init__(self, epsilon):
        super(IsometryReg, self).__init__()
        self.epsilon = epsilon

    def forward(self, data, output):
        # Number of classes
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
        jac = torch.transpose(jac, dim0=0, dim1=1)
        jac = jac.view(jac.shape[0], jac.shape[1], -1)

        # Gram matrix of Jacobian
        jac = torch.bmm(jac, torch.transpose(jac, 1, 2))

        # Compute the change of basis matrix
        change = output[:, m] / torch.square(
            2 * torch.sqrt(output[:, m]) - torch.norm(output[:, :c-1], p=1, dim=1))

        # Distance from center of simplex
        delta = torch.sqrt(output / c).sum(dim=1)
        delta = 2 * torch.acos(delta)

        # Diagonal embedding
        change = torch.diag_embed(change.unsqueeze(1).repeat(1, m))
        change = change * (delta ** 2)[:, None, None]
        change = change / self.epsilon ** 2

        # Compute regularization term (alpha in docs)
        reg = self.epsilon**2*torch.linalg.norm((jac - change).view(len(data), -1), dim=1)

        # Return
        return reg.mean()


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
        jac = jac.view(jac.shape[0], -1)

        # Compute delta and rho
        delta = torch.sqrt(output/c).sum(dim=1)
        delta = 2*torch.acos(delta)
        rho = (2*(1-torch.sqrt(output[:, m])) - output[:, :m].sum(dim=1))/(1-torch.sqrt(output[:, m]))

        # Compute regularization
        reg = F.elu(torch.sqrt(torch.square(jac).sum(dim=1)) - delta/(rho*self.epsilon))
        return reg.mean()


class SoftLenet(nn.Module):

    def __init__(self, param, eps=1e-6):
        super(SoftLenet, self).__init__()
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
