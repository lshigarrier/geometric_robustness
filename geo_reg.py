import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import time
import psutil
import os

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from mnist_model import Lenet, IsometryReg, JacobianReg, compute_jacobian, get_jacobian_bound
from mnist_utils import load_yaml
from attacks_utils import TorchAttackGaussianNoise, TorchAttackFGSM, TorchAttackPGD, TorchAttackPGDL2, TorchAttackDeepFool, TorchAttackCWL2
from defense_utils import parseval_orthonormal_constraint
from attacks_vis import plot_curves, plot_hist


# -------------------------------------------- Training & Testing ------------------------------------------------------


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

            soft_labels = F.softmax(teacher_model(data, perform_softmax=False) / param["distill_temp"], -1)
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


def test(param, model, reg_model, device, test_loader, eta, attack=None):
    # Make model deterministic and turn off gradients
    model.eval()

    # Initialize variables
    test_loss    = 0
    test_entropy = 0
    test_reg     = 0
    correct      = 0
    adv_correct  = 0
    test_bound           = []
    test_bound_robust    = []
    test_bound_nonrobust = []
    # data_robust_list        = []
    # data_nonrobust_list     = []
    # adv_data_robust_list    = []
    # adv_data_nonrobust_list = []
    # data_robust_flag     = False
    # data_nonrobust_flag  = False
    tic          = time.time()

    ## Cycle through data
    # ---------------------------------------------------------------- #
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            # Push to device
            data, target = data.to(device), target.to(device)

            # Compute loss

            if param['reg']:
                with torch.enable_grad():
                    # Ensure grad is on
                    data.requires_grad = True

                    # Forward pass
                    output = model(data)

                    # Compute regularization term and
                    reg = reg_model(data, output, device)

                    # Compute cross entropy
                    entropy = F.cross_entropy(output, target)

                    # Loss with regularization
                    loss = (1 - eta) * entropy + eta * reg

            else:
                # Forward pass
                output = model(data)

                # Compute cross entropy loss
                entropy = F.cross_entropy(output, target)

                # Do not compute regularization
                reg = torch.tensor(0)

                # Loss is only cross entropy
                loss = entropy

            # Running statistics
            test_loss    += loss.item()*len(data)
            test_entropy += entropy.item()*len(data)
            test_reg     += reg.item()*len(data)

            ## Check standard and adversarial accuracy
            # ---------------------------------------------------------------- #
            # Get prediction
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            # If batch size is 1
            if len(data) == 1:
                pred = pred[0]

            # Running total of correct
            correct_mask = pred.eq(target.view_as(pred)).view(-1)
            correct += correct_mask.sum().item()

            # Test adversary
            if param['adv_test'] and correct_mask.any():
                # Compute the max singular value and the bound
                if param['test_bound']:
                    new_data = data[correct_mask].clone()
                    with torch.enable_grad():
                        # Ensure grad is on
                        new_data.requires_grad = True

                        # Forward pass
                        new_output = model(new_data)

                        # Compute jacobian and bound
                        jac = compute_jacobian(new_data, new_output, device)
                        bound = get_jacobian_bound(new_output, param['epsilon'])
                        sv_max = torch.max(torch.linalg.svdvals(jac), dim=1)[0]

                        # Compute mask for points respecting the bound
                        diff = (bound - sv_max).squeeze()
                        test_bound.append(diff)
                        # bound_mask = diff.gt(0.575).view(-1)

                # Generate attacks
                adv_data = attack.perturb(data[correct_mask], target[correct_mask])
                ## For testing purposes
                # assert not torch.isnan(adv_data).any()
                diff_tensor = adv_data.contiguous().view(adv_data.shape[0], -1) - data[correct_mask].contiguous().view(adv_data.shape[0], -1)
                min_norm = torch.max(torch.abs(diff_tensor), dim=1)[0].min()
                # print(f'Min Linf norm: {min_norm}')
                if min_norm < 0.9*param['budget']:
                    print('PERTURBATION IS TOO SMALL!!!')

                # Feed forward
                adv_output = model(adv_data)

                # Get prediction
                adv_pred = adv_output.argmax(dim=1, keepdim=True)

                # Collect statistics
                adv_correct_mask = adv_pred.eq(target[correct_mask].view_as(adv_pred)).view(-1)
                adv_correct += adv_correct_mask.sum().item()
                if param['test_bound']:
                    '''
                    data_robust_mask    = bound_mask.logical_and(adv_correct_mask)
                    data_nonrobust_mask = bound_mask.logical_and(torch.logical_not(adv_correct_mask))
                    data_robust         = new_data[data_robust_mask]
                    data_nonrobust      = new_data[data_nonrobust_mask]
                    if data_robust.numel() > 0:
                        # data_robust_flag = True
                        adv_data_robust  = adv_data[data_robust_mask]
                        data_robust_list.append(data_robust)
                        adv_data_robust_list.append(adv_data_robust)
                    if data_nonrobust.numel() > 0:
                        # data_nonrobust_flag = True
                        adv_data_nonrobust  = adv_data[data_nonrobust_mask]
                        data_nonrobust_list.append(data_nonrobust)
                        adv_data_nonrobust_list.append(adv_data_nonrobust)
                    '''
                    test_bound_robust.append(diff[adv_correct_mask])
                    test_bound_nonrobust.append(diff[torch.logical_not(adv_correct_mask)])

                    # if data_robust_flag and data_nonrobust_flag:
                    #    break

            ## Display results
            # ---------------------------------------------------------------- #
            if not(param['train']) and param['verbose'] and (batch_idx % param['log_interval'] == 0):
                print('Test: {}/{} ({:.0f}%)\tLoss: {:.6f}, Cross Entropy: {:.6f}, Reg: {:.6f}'.format(
                    batch_idx * len(data), len(test_loader.dataset), 100. * batch_idx / len(test_loader),
                    loss.item(), entropy.item(), reg.item()))
                print(f'Elapsed time (s): {time.time() - tic}')
                print(f'Memory usage (GB): {psutil.Process(os.getpid()).memory_info()[0] / (2. ** 30)}')
                tic = time.time()

    ## Calculate results, display and return
    # ---------------------------------------------------------------- #
    test_loss    /= len(test_loader.dataset)
    test_entropy /= len(test_loader.dataset)
    test_reg     /= len(test_loader.dataset)
    if param['adv_test']:
        print('Test set: Average loss: {:.4f}, Average cross entropy: {:.4f}, Average reg: {:.4f}, '
              'Accuracy: {}/{} ({:.0f}%), Robust accuracy: {}/{} ({:.0f}%)\n'.format(
               test_loss, test_entropy, test_reg,
               correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset),
               adv_correct, correct, 100. * adv_correct / correct))
    else:
        print('Test set: Average loss: {:.4f}, Average cross entropy: {:.4f}, Average reg: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, test_entropy, test_reg,
            correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
    if param['test_bound']:
        '''
        data_robust_list = torch.cat(data_robust_list, dim=0)
        data_nonrobust_list = torch.cat(data_nonrobust_list, dim=0)
        adv_data_robust_list = torch.cat(adv_data_robust_list, dim=0)
        adv_data_nonrobust_list = torch.cat(adv_data_nonrobust_list, dim=0)
        torch.save(data_robust_list, 'data/img/jac_3_test_robust_image.pt')
        torch.save(adv_data_robust_list, 'data/img/jac_3_test_adv_robust_image.pt')
        torch.save(data_nonrobust_list, 'data/img/jac_3_test_nonrobust_image.pt')
        torch.save(adv_data_nonrobust_list, 'data/img/jac_3_test_adv_nonrobust_image.pt')
        '''
        test_bound = torch.cat(test_bound, dim=0).flatten().tolist()
        test_bound_robust = torch.cat(test_bound_robust, dim=0).flatten().tolist()
        test_bound_nonrobust = torch.cat(test_bound_nonrobust, dim=0).flatten().tolist()
        return test_bound, test_bound_robust, test_bound_nonrobust
    return test_loss, test_entropy, test_reg


# ---------------------------------------- Initialization & Main Loop --------------------------------------------------


def initialize(param, device):
    ## Load TEST batch size
    # -------------------------------------------------------------- #
    # Use evaluation batch size
    if param['load']:
        test_kwargs = {'batch_size': param['testing_batch_size']}
        print(f'Using testing batch size for test loader')

    # Use validation batch size
    else:
        test_kwargs = {'batch_size': param['test_batch_size']}
        print(f'Using training batch size for test loader')

    ## Load TRAIN batch size
    # -------------------------------------------------------------- #
    # use evaluation batch size
    if param['load']:
        train_kwargs = {'batch_size': param['testing_batch_size']}
        print(f'Using testing batch size for train loader')

    # Use train batch size
    else:
        train_kwargs = {'batch_size': param['batch_size']}
        print(f'Using training batch size for train loader')

    ## Machine settings
    # -------------------------------------------------------------- #
    if torch.cuda.is_available():
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory' : True,
                       'shuffle'    : True}

        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    ## Load dataset from torchvision
    # -------------------------------------------------------------- #
    # Train set
    dataset1 = datasets.MNIST('./data/mnist', train=True, download=True, transform=transforms.ToTensor())

    # Small train set
    subset = torch.utils.data.Subset(dataset1, range(1000))

    # Test set
    dataset2 = datasets.MNIST('./data/mnist', train=False, transform=transforms.ToTensor())

    # Create data loaders
    train_loader       = DataLoader(dataset1, **train_kwargs)
    light_train_loader = DataLoader(subset, **train_kwargs)
    test_loader        = DataLoader(dataset2, **test_kwargs)

    ## Load model
    # -------------------------------------------------------------- #
    # Initalize network class
    model = Lenet(param).to(device)

    # Load parameters from file
    if param['load']:
        print(f'Loading weights')
        model.load_state_dict(torch.load(f'models/{param["name"]}/{param["model"]}', map_location='cpu'))

    else:
        print(f'Randomly initialized weights')

    ## Initialize regularization class
    # -------------------------------------------------------------- #
    if param['reg_type'] == 'jacobian':
        reg_model = JacobianReg(param['epsilon'], barrier=param['barrier'])
    elif param['reg_type'] == 'isometry':
        reg_model = IsometryReg(param['epsilon'])
    else:
        reg_model = None

    # Load teacher model
    if param['distill']:
        # Initalize network class
        teacher_model = Lenet(param).to(device)

        print(f'Loading weights onto teacher model')
        teacher_model.load_state_dict(torch.load(f'models/{param["name"]}/{param["model"]}', map_location='cpu'))

        # Make model deterministic and turn off gradient computations
        teacher_model.eval()
    else:
        teacher_model = None

    # Set optimizer
    # optimizer = optim.SGD(model.parameters(), lr=param['learning_rate'])
    optimizer = optim.Adam(model.parameters(), lr=param['learning_rate'])

    print('Initialization done')
    return train_loader, light_train_loader, test_loader, model, reg_model, teacher_model, optimizer


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
        test_loss, test_entropy, test_reg = test(param, model, reg_model, device, test_loader, eta, attack)

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


# ---------------------------------------------------- Main ------------------------------------------------------------


def one_train_or_test(param):
    # Set random seed
    torch.manual_seed(param['seed'])

    # Declare CPU/GPU usage
    if param['gpu_number'] is not None:
        os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = param['gpu_number']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using {device}')

    # Load data and model
    train_loader, light_train_loader, test_loader, model, reg_model, teacher_model, optimizer = initialize(param, device)

    # Load attacker
    attack = None
    if param['adv_test'] or param['adv_train']:

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
    if param['train']:
        print(f'Start training')
        _ = training(param, device, train_loader, test_loader, model, reg_model, teacher_model, optimizer, attack=attack)

    # Test model
    else:
        print(f'Start testing')
        # Set data loader
        if param['loader'] == 'test':
            loader = test_loader
            print('Using test loader')
        else:
            loader = light_train_loader
            print('Using light train loader')

        # Compute eta value
        if param['test_epoch'] < param['epoch_reg']:
            eta = param['eta_min']
        else:
            eta = param['eta_max']
        # eta = param['eta_min'] * (param['eta_max'] / param['eta_min']) ** ((param['test_epoch'] - 1) / (param['epochs'] - 1))

        # Launch testing
        if param['test_bound']:
            test_bound, test_bound_robust, test_bound_nonrobust = test(param, model, reg_model, device, loader, eta, attack)
            _ = plot_hist(test_bound, "All points", "Bound minus max singular value", "Number")
            _ = plot_hist(test_bound_robust, "Robust points", "Bound minus max singular value", "Number")
            _ = plot_hist(test_bound_nonrobust, "Non-robust points", "Bound minus max singular value", "Number")
        else:
            test(param, model, reg_model, device, loader, eta, attack)

    if param['plot']:
        plt.show()


def main():
    # Detect anomaly in autograd
    torch.autograd.set_detect_anomaly(True)

    # Load configurations
    param = load_yaml('config_geo_reg')

    if param['loop']:
        # Create config lists
        eta_list = []
        # eta_list = [1e-4, 3e-4, 5e-4, 7e-4, 9e-4, 1e-3, 3e-3, 5e-3, 7e-3, 9e-3, 1e-2]
        # model_list = ['iso-4_1', 'iso-4_3', 'iso-4_5', 'iso-4_7', 'iso-4_9',
        #              'iso-3_1', 'iso-3_3', 'iso-3_5', 'iso-3_7', 'iso-3_9', 'iso-2_1']

        # Loop over configurations
        for i in range(len(eta_list)):
            # param['name'] = 'isometry/' + model_list[i]
            # param['eta_max'] = eta_list[i]
            # print(param['name'])
            one_train_or_test(param)

    else:
        one_train_or_test(param)


if __name__ == '__main__':
    main()
