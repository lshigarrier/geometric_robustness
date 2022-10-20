import torch
import torch.nn.functional as F
import torch.optim as optim
import torchattacks
import matplotlib.pyplot as plt
import time
import psutil
import os

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from mnist_model import SoftLeNet, JacobianReg
from mnist_utils import load_yaml
from attacks_vis import plot_curves


# -------------------------------------------- Training & Testing ------------------------------------------------------


def train(param, model, device, train_loader, optimizer, epoch, eta, jacreg):
    epoch_loss = 0
    epoch_entropy = 0
    epoch_reg = 0
    model.train()
    tic = time.time()
    if param['reg'] and epoch >= param['epoch_reg']:
        print(f'Eta:{eta}')
    for par in model.parameters():
        par.requires_grad = True
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data.requires_grad = True
        output = model(data)
        if param['reg'] and epoch >= param['epoch_reg']:
            reg = jacreg(data, output)
            entropy = F.cross_entropy(output, target)
            loss = (1 - eta) * entropy + eta * reg
        else:
            entropy, reg = torch.tensor(0), torch.tensor(0)
            loss = F.cross_entropy(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()*len(data)
        epoch_entropy += entropy.item()*len(data)
        epoch_reg += reg.item()*len(data)
        if param['verbose'] and (batch_idx % param['log_interval'] == 0):
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Cross Entropy: {:.6f}, Reg: {:.6f}'.format(
                epoch, batch_idx*len(data), len(train_loader.dataset), 100.*batch_idx/len(train_loader),
                loss.item(), entropy.item(), reg.item()))
            print(f'Elapsed time (s): {time.time() - tic}')
            print(f'Memory usage (GB): {psutil.Process(os.getpid()).memory_info()[0] / (2. ** 30)}')
            tic = time.time()
    epoch_loss /= len(train_loader.dataset)
    epoch_entropy /= len(train_loader.dataset)
    epoch_reg /= len(train_loader.dataset)
    if param['verbose']:
        print('Train set: Average loss: {:.4f}, Average cross entropy: {:.4f}, Average reg: {:.4f}'.format(
            epoch_loss, epoch_entropy, epoch_reg))
    return epoch_loss, epoch_entropy, epoch_reg


def test(param, model, device, test_loader, epoch, eta, jacreg, attack=None):
    test_loss = 0
    test_entropy = 0
    test_reg = 0
    correct = 0
    adv_correct = 0
    adv_total = 0
    tic = time.time()
    for par in model.parameters():
        par.requires_grad = False
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        original_image = data.clone()
        data.requires_grad = True
        output = model(data)
        if param['reg'] and epoch >= param['epoch_reg']:
            reg = jacreg(data, output)
            entropy = F.cross_entropy(output, target)
            loss = (1 - eta) * entropy + eta * reg
        else:
            entropy, reg = torch.tensor(0), torch.tensor(0)
            loss = F.cross_entropy(output, target)
        model.zero_grad()
        test_loss += loss.item()*len(data)
        test_entropy += entropy.item()*len(data)
        test_reg += reg.item()*len(data)
        if len(data) == 1:
            pred = output.argmax(dim=1, keepdim=True)[0]
            correct_pred = pred.eq(target.view_as(pred)).item()
            correct += correct_pred
            if param['adv_test'] and correct_pred:
                # use predicted label as target label (or not)
                adv_data = attack(original_image, target)
                adv_output = model(adv_data)
                adv_pred = adv_output.argmax(dim=1, keepdim=True)
                adv_correct += adv_pred.eq(target.view_as(adv_pred)).sum().item()
                adv_total += 1
        else:
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            if param['adv_test']:
                # use predicted label as target label (or not)
                adv_data = attack(original_image, pred.view_as(target))  # pred or target
                adv_output = model(adv_data)
                adv_pred = adv_output.argmax(dim=1, keepdim=True)
                adv_correct += adv_pred.eq(pred.view_as(adv_pred)).sum().item()  # pred or target
                adv_total += 1
        if not(param['train']) and param['verbose'] and (batch_idx % param['log_interval'] == 0):
            print('Test: {}/{} ({:.0f}%)\tLoss: {:.6f}, Cross Entropy: {:.6f}, Reg: {:.6f}'.format(
                batch_idx * len(data), len(test_loader.dataset), 100. * batch_idx / len(test_loader),
                loss.item(), entropy.item(), reg.item()))
            print(f'Elapsed time (s): {time.time() - tic}')
            print(f'Memory usage (GB): {psutil.Process(os.getpid()).memory_info()[0] / (2. ** 30)}')
            tic = time.time()
    test_loss /= len(test_loader.dataset)
    test_entropy /= len(test_loader.dataset)
    test_reg /= len(test_loader.dataset)
    if param['adv_test']:
        print('Test set: Average loss: {:.4f}, Average cross entropy: {:.4f}, Average reg: {:.4f}, '
              'Accuracy: {}/{} ({:.0f}%), Robust accuracy: {}/{} ({:.0f}%)\n'.format(
               test_loss, test_entropy, test_reg,
               correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset),
               adv_correct, adv_total, 100. * adv_correct / adv_total))
    else:
        print('Test set: Average loss: {:.4f}, Average cross entropy: {:.4f}, Average reg: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, test_entropy, test_reg,
            correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
    return test_loss, test_entropy, test_reg


def initialize(param, device):
    if param['load']:
        test_kwargs = {'batch_size': param['testing_batch_size']}
        print(f'Using testing batch size for test loader')
    else:
        test_kwargs = {'batch_size': param['test_batch_size']}
        print(f'Using training batch size for test loader')
    if param['load']:
        train_kwargs = {'batch_size': param['testing_batch_size']}
        print(f'Using testing batch size for train loader')
    else:
        train_kwargs = {'batch_size': param['batch_size']}
        print(f'Using training batch size for train loader')
    if torch.cuda.is_available():
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    dataset1 = datasets.MNIST('./data/mnist', train=True, download=True, transform=transforms.ToTensor())
    subset = torch.utils.data.Subset(dataset1, range(1000))
    dataset2 = datasets.MNIST('./data/mnist', train=False, transform=transforms.ToTensor())
    train_loader = DataLoader(dataset1, **train_kwargs)
    light_train_loader = DataLoader(subset, **train_kwargs)
    test_loader = DataLoader(dataset2, **test_kwargs)

    model = SoftLeNet(param).to(device)
    if param['load']:
        print(f'Loading weights')
        model.load_state_dict(torch.load(f'models/{param["name"]}/{param["model"]}', map_location='cpu'))
    else:
        print(f'Randomly initialized weights')

    jacreg = JacobianReg(param['epsilon'])

    print('Initialization done')

    return train_loader, light_train_loader, test_loader, model, jacreg


def training(param, device, train_loader, test_loader, model, jacreg, attack=None):
    torch.autograd.set_detect_anomaly(True)
    optimizer = optim.Adam(model.parameters(), lr=param['learning_rate'])
    loss_list, entropy_list, reg_list = [], [], []
    test_loss_list, test_entropy_list, test_reg_list = [], [], []

    for epoch in range(1, param['epochs'] + 1):
        eta = param['eta_min'] * (param['eta_max']/param['eta_min'])**((epoch - 1)/(param['epochs'] - 1))
        epoch_loss, epoch_entropy, epoch_reg = train(param, model, device, train_loader, optimizer, epoch, eta, jacreg)
        test_loss, test_entropy, test_reg = test(param, model, device, test_loader, epoch, eta, jacreg, attack=attack)
        if epoch % param['save_step'] == 0:
            torch.save(model.state_dict(), f'models/{param["name"]}/{epoch:05d}.pt')
        loss_list.append(epoch_loss)
        entropy_list.append(epoch_entropy)
        reg_list.append(epoch_reg)
        test_loss_list.append(test_loss)
        test_entropy_list.append(test_entropy)
        test_reg_list.append(test_reg)
    fig1 = plot_curves(loss_list, test_loss_list, "Loss function", "Epoch", "Loss")
    fig2 = plot_curves(entropy_list, test_entropy_list, "Cross Entropy", "Epoch", "Cross entropy")
    fig3 = plot_curves(reg_list, test_reg_list, "Regularization", "Epoch", "Regularization")
    return fig1, fig2, fig3


# ---------------------------------------------------- Main ------------------------------------------------------------


def main():
    param = load_yaml('param_jac')
    torch.manual_seed(param['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using {device}')
    train_loader, light_train_loader, test_loader, model, jacreg = initialize(param, device)
    attack = None
    if param['adv_test']:
        attack = torchattacks.PGD(model, eps=param['budget'], alpha=param['alpha'], steps=param['max_iter'], random_start=False)
    if param['train']:
        print(f'Start training')
        _ = training(param, device, train_loader, test_loader, model, jacreg, attack=attack)
    else:
        print(f'Start testing')
        if param['loader'] == 'test':
            loader = test_loader
            print('Using test loader')
        else:
            loader = light_train_loader
            print('Using light train loader')
        test(param, model, device, loader, 1, param['eta_max'], jacreg, attack=attack)
    plt.show()


if __name__ == '__main__':
    main()
