import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
import psutil
import os

from mnist_utils import load_yaml, initialize
from test_geo_reg import test
from attacks_utils import TorchAttackGaussianNoise, TorchAttackFGSM, TorchAttackPGD, TorchAttackPGDL2, TorchAttackDeepFool, TorchAttackCWL2
from defense_utils import parseval_orthonormal_constraint
from attacks_vis import plot_curves


def train(param, model, reg_model, teacher_model, device, train_loader, optimizer, epoch, eta, attack=None):
    # Initialize variables
    epoch_loss    = 0
    epoch_entropy = 0
    epoch_reg     = 0

    # Make model stochastic and compute gradient graph
    model.train()

    # Display lambda value
    if param['reg']:
        print(f'Eta:{eta}')

    # Cycle through data
    tic = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        # Push to GPU/CPU
        data, target = data.to(device), target.to(device)

        # Ensure grad is on
        data.requires_grad = True

        # Adversarial train
        if param['adv_train']:
            # Update attacker
            attack.model = model
            attack.set_attacker()

            # Generate attacks
            data = attack.perturb(data, target)

        # Forward pass
        output = model(data)

        # Calculate soft-labels

        # Compute loss
        if param['distill']:
            ## Sanity check that this method is equivalent to original criterion
            # batch_size = labels.size(0)
            # label_onehot = torch.FloatTensor(batch_size, data.num_classes)
            # label_onehot.zero_()
            # label_onehot.scatter_(1, labels.view(-1, 1), 1)
            # print("One Hot", label_onehot[0])
            # print(torch.sum(-label_onehot * F.log_softmax(outputs, -1), -1).mean())

            soft_labels = F.softmax(teacher_model(data) / param["distill_temp"], -1)
            # torch.log(output) or F.log_softmax(output, -1) ?
            entropy = torch.sum(-soft_labels * torch.log(output), -1).mean()

            # Do not compute regularization
            reg = torch.tensor(0)

            # Loss is only cross entropy
            loss = entropy

        elif param['reg']:
            # Compute regularization term and cross entropy loss
            reg     = reg_model(data, output, device)
            entropy = F.cross_entropy(output, target)

            # Loss with regularization
            loss = (1 - eta) * entropy + eta * reg

        else:
            # Compute cross entropy loss
            entropy = F.cross_entropy(output, target)

            # Do not compute regularization
            reg = torch.tensor(0)

            # Loss is only cross entropy
            loss = entropy

        # Gradients set to zero
        optimizer.zero_grad()

        # Backpropagation
        loss.backward()

        # Update model parameters
        optimizer.step()

        # Parseval Tight Constraint
        if param['parseval_train']:
            model = parseval_orthonormal_constraint(model)

        # Update running totals
        epoch_loss    += loss.item()*len(data)
        epoch_entropy += entropy.item()*len(data)
        epoch_reg     += reg.item()*len(data)

        # Display
        if param['verbose'] and (batch_idx % param['log_interval'] == 0):
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Cross Entropy: {:.6f}, Reg: {:.6f}'.format(
                epoch, batch_idx*len(data), len(train_loader.dataset), 100.*batch_idx/len(train_loader),
                loss.item(), entropy.item(), reg.item()))
            print(f'Elapsed time (s): {time.time() - tic}')
            print(f'Memory usage (GB): {psutil.Process(os.getpid()).memory_info()[0] / (2. ** 30)}')
            tic = time.time()

    # Calculate results
    epoch_loss    /= len(train_loader.dataset)
    epoch_entropy /= len(train_loader.dataset)
    epoch_reg     /= len(train_loader.dataset)

    # Display
    if param['verbose']:
        print('Train set: Average loss: {:.4f}, Average cross entropy: {:.4f}, Average reg: {:.4f}'.format(
            epoch_loss, epoch_entropy, epoch_reg))

    # Return results
    return epoch_loss, epoch_entropy, epoch_reg


def training(param, device, train_loader, test_loader, model, reg_model, teacher_model, optimizer, attack=None):
    ## Initialize
    # ---------------------------------------------------------------------- #
    # Initiate variables
    loss_list, entropy_list, reg_list = [], [], []
    test_loss_list, test_entropy_list, test_reg_list = [], [], []
    # ---------------------------------------------------------------------- #

    ## Cycle through epochs
    # ---------------------------------------------------------------------- #
    for epoch in range(1, param['epochs'] + 1):
        # Set eta term
        # eta = param['eta_min'] * (param['eta_max']/param['eta_min'])**((epoch - 1)/(param['epochs'] - 1))
        if epoch < param['epoch_reg']:
            eta = param['eta_min']
        else:
            eta = param['eta_max']

        # Train
        epoch_loss, epoch_entropy, epoch_reg = train(param, model, reg_model, teacher_model, device, train_loader, optimizer, epoch, eta, attack)

        # Validate
        test_loss, test_entropy, test_reg = test(param, model, reg_model, device, test_loader, eta, attack=attack, train=True)

        # Checkpoint model weights
        if epoch % param['save_step'] == 0:
            torch.save(model.state_dict(), f'models/{param["name"]}/{epoch:05d}.pt')

        # Collect statistics
        loss_list.append(epoch_loss)
        entropy_list.append(epoch_entropy)
        reg_list.append(epoch_reg)
        test_loss_list.append(test_loss)
        test_entropy_list.append(test_entropy)
        test_reg_list.append(test_reg)

    if param['plot']:
        # Display plot
        fig1 = plot_curves(loss_list, test_loss_list, "Loss function", "Epoch", "Loss")
        fig2 = plot_curves(entropy_list, test_entropy_list, "Cross Entropy", "Epoch", "Cross entropy")
        fig3 = plot_curves(reg_list, test_reg_list, "Regularization", "Epoch", "Regularization")

        # Return
        return fig1, fig2, fig3
    return None


def main():
    # Detect anomaly in autograd
    torch.autograd.set_detect_anomaly(True)

    # Load configurations
    param = load_yaml('train_geo_reg_conf')

    # Set random seed
    torch.manual_seed(param['seed'])

    # Declare CPU/GPU usage
    if param['gpu_number'] is not None:
        os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = param['gpu_number']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using {device}')

    # Load data and model
    train_loader, test_loader, model, reg_model, teacher_model, optimizer = initialize(param, device, train=True)

    # Load attacker
    attack = None
    if param['adv_train']:

        if param["attack_type"] == "fgsm":
            attack = TorchAttackFGSM(model = model,
                                     eps   = param['budget'])

        elif param['attack_type'] == 'gn':
            attack = TorchAttackGaussianNoise(model = model,
                                              std   = param['budget'])

        elif param['attack_type'] == "pgd":
            if param['perturbation_type'] == 'linf':
                attack = TorchAttackPGD(model        = model,
                                        eps          = param['budget'],
                                        alpha        = param['alpha'],
                                        steps        = param['max_iter'],
                                        random_start = param['random_start'])
            elif param['perturbation_type'] == 'l2':
                attack = TorchAttackPGDL2(model        = model,
                                          eps          = param['budget'],
                                          alpha        = param['alpha'],
                                          steps        = param['max_iter'],
                                          random_start = param['random_start'])
            else:
                print("Invalid perturbation_type in config file, please use 'linf' or 'l2'")
                exit()

        elif param['attack_type'] == "deep_fool":
            attack = TorchAttackDeepFool(model     = model,
                                         max_iters = param['max_iter'])

        elif param['attack_type'] == "cw":
            attack = TorchAttackCWL2(model     = model,
                                     max_iters = param['max_iter'])

        else:
            print("Invalid attack_type in config file, please use 'fgsm' or add a new class in attacks_utils....")
            exit()

    # Train model
    print(f'Start training')
    _ = training(param, device, train_loader, test_loader, model, reg_model, teacher_model, optimizer, attack=attack)

    if param['plot']:
        plt.show()


if __name__ == '__main__':
    main()
