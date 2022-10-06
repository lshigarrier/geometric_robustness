import torch
import torch.nn.functional as F
import torch.optim as optim
import functorch
import matplotlib.pyplot as plt
import time
import psutil
import os

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from mnist_model import SoftLeNet
from mnist_utils import load_yaml
from attacks_utils import FastGradientSignUntargeted
from attacks_vis import plot_curves, plot_hist


# ------------------------------------------ Isometric Regularization --------------------------------------------------


def create_transform(model=None):
    """
    Spherical transformation followed by stereographic projection
    """
    def transform(data):
        if model:
            data = model(data)
        m = data.shape[1] - 1
        mu = torch.sqrt(data)
        phi = 2*mu[:, :m]/(1 - mu[:, m].unsqueeze(1).repeat(1, m))
        return phi
    return transform


def jacobian_transform(data, model, device):
    data = data.unsqueeze(1)
    transform = create_transform(model)
    jac = functorch.vmap(functorch.jacrev(transform), randomness='different')(data).to(device)
    jac = jac.squeeze()
    if len(data) == 1:
        jac = jac.unsqueeze(0)
    jac = torch.reshape(jac, (jac.shape[0], jac.shape[1], -1))
    return jac


def change_matrix(output, epsilon, nb_class):
    change = output[:, nb_class-1]/torch.square(2*torch.sqrt(output[:, nb_class-1]) - torch.norm(output[:, :nb_class-1], p=1, dim=1))
    delta = 2*torch.acos(1/torch.sqrt(torch.tensor(nb_class)))
    return functorch.vmap(torch.diag)(change.unsqueeze(1).repeat(1, nb_class-1))*delta**2/epsilon**2


def iso_loss_transform(output, target, data, epsilon, model, device, test_mode=False):
    nb_class = output.shape[1]
    jac = jacobian_transform(data, model, device)
    assert not torch.isnan(jac).any()
    change = change_matrix(output, epsilon, nb_class)
    jac = torch.bmm(jac, torch.transpose(jac, 1, 2))
    cross_entropy = F.cross_entropy(output, target)
    reg = epsilon**2*torch.linalg.norm((jac - change).view(len(data), -1), dim=1).sum()/len(data)
    if test_mode:
        # print(f'cross entropy: {cross_entropy}\nreg: {reg}\njac*tjac: {jac}\nchange: {change}')
        return cross_entropy, reg, change
    return cross_entropy, reg


# -------------------------------------------- Training & Testing ------------------------------------------------------


def train(param, model, device, train_loader, optimizer, epoch, lmbda):
    epoch_loss = 0
    epoch_entropy = 0
    epoch_reg = 0
    model.train()
    if param['verbose'] and param['reg']:
        print(f'Lambda: {lmbda}')
    tic = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data.requires_grad = True
        optimizer.zero_grad()
        output = model(data)
        if param['reg']:
            cross_entropy, reg = iso_loss_transform(output, target, data, param['epsilon_l2'], model, device)
            loss = (1 - lmbda) * cross_entropy + lmbda * reg
        else:
            cross_entropy, reg = torch.tensor(0), torch.tensor(0)
            loss = F.cross_entropy(output, target)
        assert not torch.isnan(loss).any()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()*len(data)
        epoch_entropy += cross_entropy.item()*len(data)
        epoch_reg += reg.item()*len(data)
        if param['verbose'] and (batch_idx % param['log_interval'] == 0):
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Cross Entropy: {:.6f}, Reg: {:.6f}'.format(
                epoch, batch_idx*len(data), len(train_loader.dataset), 100.*batch_idx/len(train_loader),
                loss.item(), cross_entropy.item(), reg.item()))
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


def test(param, model, device, test_loader, lmbda, attack=None):
    model.eval()
    test_loss = 0
    test_entropy = 0
    test_reg = 0
    correct = 0
    adv_correct = 0
    adv_total = 0
    hist_correct = []
    hist_incorrect = []
    tic = time.time()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            if param['reg']:
                cross_entropy, reg = iso_loss_transform(output, target, data, param['epsilon_l2'], model, device)
                loss = (1 - lmbda) * cross_entropy + lmbda * reg
            else:
                cross_entropy, reg = torch.tensor(0), torch.tensor(0)
                loss = F.cross_entropy(output, target)
            test_loss += loss.item()*len(data)
            test_entropy += cross_entropy.item()*len(data)
            test_reg += reg.item()*len(data)
            if len(data) == 1:
                pred = output.argmax(dim=1, keepdim=True)[0]
                correct_pred = pred.eq(target.view_as(pred)).item()
                correct += correct_pred
                if param['adv_test'] and correct_pred:
                    # use predicted label as target label (or not)
                    # with torch.enable_grad():
                    adv_data = attack.perturb(data, target, False)
                    adv_output = model(adv_data)
                    adv_pred = adv_output.argmax(dim=1, keepdim=True)
                    adv_correct += adv_pred.eq(target.view_as(adv_pred)).sum().item()
                    adv_total += 1
                    if adv_pred:
                        hist_correct.append(reg.item())
                    else:
                        hist_incorrect.append(reg.item())
            else:
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
                if param['adv_test']:
                    # use predicted label as target label (or not)
                    # with torch.enable_grad():
                    adv_data = attack.perturb(data, pred.view_as(target), False)  # pred or target
                    adv_output = model(adv_data)
                    adv_pred = adv_output.argmax(dim=1, keepdim=True)
                    adv_correct += adv_pred.eq(pred.view_as(adv_pred)).sum().item()  # pred or target
                    adv_total += 1
            if not(param['train']) and param['verbose'] and (batch_idx % param['log_interval'] == 0):
                print('Test: {}/{} ({:.0f}%)\tLoss: {:.6f}, Cross Entropy: {:.6f}, Reg: {:.6f}'.format(
                    batch_idx * len(data), len(test_loader.dataset), 100. * batch_idx / len(test_loader),
                    loss.item(), cross_entropy.item(), reg.item()))
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
    return test_loss, test_entropy, test_reg, hist_correct, hist_incorrect


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
        model.load_state_dict(torch.load(f'models/isometry/{param["name"]}/{param["model"]}', map_location='cpu'))
        model.eval()
    else:
        print(f'Randomly initialized weights')

    print('Initialization done')

    return train_loader, light_train_loader, test_loader, model


def training(param, device, train_loader, test_loader, model, attack=None):
    optimizer = optim.SGD(model.parameters(), lr=param['learning_rate'])
    loss_list, entropy_list, reg_list = [], [], []
    test_loss_list, test_entropy_list, test_reg_list = [], [], []

    for epoch in range(1, param['epochs'] + 1):
        # lmbda = param['lambda_min'] + (epoch - 1)/(param['epochs'] - 1)*(param['lambda_max'] - param['lambda_min'])
        lmbda = param['lambda_min'] * (param['lambda_max']/param['lambda_min'])**((epoch - 1)/(param['epochs'] - 1))
        epoch_loss, epoch_entropy, epoch_reg = train(param, model, device, train_loader, optimizer, epoch, lmbda)
        test_loss, test_entropy, test_reg, _, _ = test(param, model, device, test_loader, lmbda, attack=attack)
        if epoch % param['save_step'] == 0:
            torch.save(model.state_dict(), f'models/isometry/{param["name"]}/{epoch:05d}.pt')
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


def testing_loss(param, device, loader, model):
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            if param['reg']:
                cross_entropy, reg = iso_loss_transform(output, target, data, param['epsilon_l2'], model, device, test_mode=True)
                loss = (1 - param['lambda_max']) * cross_entropy + param['lambda_max'] * reg
            else:
                loss = F.cross_entropy(output, target)
            if len(data) == 1:
                pred = output.argmax(dim=1, keepdim=True)[0]
            else:
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            print(f'loss: {loss}\npred: {pred}\ntarget: {target}')
            if batch_idx == 0:
                break


def main():
    param = load_yaml('param_iso')
    torch.manual_seed(param['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using {device}')
    train_loader, light_train_loader, test_loader, model = initialize(param, device)
    attack = None
    if param['adv_test']:
        attack = FastGradientSignUntargeted(model, device, param['budget'], param['alpha'],
                                            min_val=0, max_val=1, max_iters=param['max_iter'],
                                            _type=param['perturbation_type'], _loss='cross_entropy')
    if param['train']:
        print(f'Start training')
        _ = training(param, device, train_loader, test_loader, model, attack=attack)
    else:
        print(f'Start testing')
        # testing_loss(param, device, train_loader, model)
        if param['loader'] == 'test':
            loader = test_loader
            print('Using test loader')
        else:
            loader = light_train_loader
            print('Using light train loader')
        if param['jacobian']:
            for batch_idx, (data, target) in enumerate(loader):
                output = model(data)
                if param['reg']:
                    cross_entropy, reg, change = iso_loss_transform(output, target, data, param['epsilon_l2'], model, device, test_mode=True)
                jac = jacobian_transform(data, model, device)[0]
                svd = torch.linalg.svdvals(jac)
                std, mean = torch.std_mean(svd)
                print(torch.sqrt(change[0, 0, 0]))
                print(svd)
                print(len(svd))
                print(mean)
                print(std)
                break
        else:
            lmbda = param['lambda_min'] * (param['lambda_max'] / param['lambda_min']) ** ((param['test_epoch'] - 1) / (param['epochs'] - 1))
            _, _, _, hist_correct, hist_incorrect = test(param, model, device, loader, lmbda, attack=attack)
            max_reg = min(max(hist_incorrect), max(hist_correct))
            hist1 = plot_hist(hist_correct, f'Robust points\n{len(hist_correct)}', 'Regularization', 'Number of points', xmax=max_reg)
            hist2 = plot_hist(hist_incorrect, f'Non-robust points\n{len(hist_incorrect)}', 'Regularization', 'Number of points', xmax=max_reg)
    plt.show()


if __name__ == '__main__':
    main()
