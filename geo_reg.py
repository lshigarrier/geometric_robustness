import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import time
import psutil
import os

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from mnist_model import SoftLeNet, LogitLenet, IsometryReg, JacobianReg
from mnist_utils import load_yaml
from attacks_utils import FastGradientSignUntargeted, TorchAttackDeepFool, TorchAttackCWL2
from attacks_vis import plot_curves


# -------------------------------------------- Training & Testing ------------------------------------------------------


def train(param, model, reg_model, teacher_model, device, train_loader, optimizer, epoch, eta, attack=None):
    # Initialize variables
    epoch_loss    = 0
    epoch_entropy = 0
    epoch_reg     = 0

    # Make model stochastic and compute gradient graph
    model.train()

    # Display lambda value
    if param['reg'] and epoch >= param['epoch_reg']:
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
            entropy = torch.sum(-soft_labels * F.log_softmax(output, -1), -1).mean()

            # Do not compute regularization
            reg = torch.tensor(0)

            # Loss is only cross entropy
            loss = entropy

        elif param['reg'] and epoch >= param['epoch_reg']:
            # Compute regularization term and cross entropy loss
            reg     = reg_model(data, output)
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


def test(param, model, reg_model, device, test_loader, epoch, eta, attack=None):
    # Make model deterministic and turn off gradients
    model.eval()

    # Initialize variables
    test_loss    = 0
    test_entropy = 0
    test_reg     = 0
    correct      = 0
    adv_correct  = 0
    tic          = time.time()

    ## Cycle through data
    # ---------------------------------------------------------------- #
    with torch.enable_grad() if param['adv_test'] else torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            # Push to device
            data, target = data.to(device), target.to(device)

            # Ensure grad is on
            data.requires_grad = True

            # Forward pass
            output = model(data)

            # Compute loss
            if param['reg'] and epoch >= param['epoch_reg']:
                # Compute regularization term and cross entropy
                reg     = reg_model(data, output)
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
            model.zero_grad()

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
            if param['adv_test']:
                # Generate attacks
                adv_data = attack.perturb(data[correct_mask], target[correct_mask])

                # Feed forward
                adv_output = model(adv_data)

                # Get prediction
                adv_pred = adv_output.argmax(dim=1, keepdim=True)

                # Collect statistics
                adv_correct += adv_pred.eq(target[correct_mask].view_as(adv_pred)).sum().item()

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
    return test_loss, test_entropy, test_reg


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
    model = SoftLeNet(param).to(device)

    # Load parameters from file
    if param['load']:
        print(f'Loading weights')
        model.load_state_dict(torch.load(f'models/{param["name"]}/{param["model"]}', map_location='cpu'))

    else:
        print(f'Randomly initialized weights')

    ## Initialize regularization class
    # -------------------------------------------------------------- #
    if param['reg_type'] == 'jacobian':
        reg_model = JacobianReg(param['epsilon'])
    elif param['reg_type'] == 'isometry':
        reg_model = IsometryReg(param['epsilon'])
    else:
        reg_model = None

    # Load teacher model
    if param['distill']:
        # Initalize network class
        teacher_model = LogitLenet(param).to(device)

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
    # Detect anomaly in autograd
    torch.autograd.set_detect_anomaly(True)
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
        eta = param['eta_min'] * (param['eta_max']/param['eta_min'])**((epoch - 1)/(param['epochs'] - 1))

        # Train
        epoch_loss, epoch_entropy, epoch_reg = train(param, model, reg_model, teacher_model, device, train_loader, optimizer, epoch, eta, attack)

        # Validate
        test_loss, test_entropy, test_reg = test(param, model, reg_model, device, test_loader, epoch, eta, attack)

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

    # Display plot
    # fig1 = plot_curves(loss_list, test_loss_list, "Loss function", "Epoch", "Loss")
    # fig2 = plot_curves(entropy_list, test_entropy_list, "Cross Entropy", "Epoch", "Cross entropy")
    # fig3 = plot_curves(reg_list, test_reg_list, "Regularization", "Epoch", "Regularization")

    # Return
    # return fig1, fig2, fig3
    return 0


# ---------------------------------------------------- Main ------------------------------------------------------------


def main():
    # Load configurations
    param = load_yaml('param_geo_reg')

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
            attack = FastGradientSignUntargeted(model,
                                                device,
                                                epsilon   = param['budget'],
                                                alpha     = param['alpha'],
                                                min_val   = 0,
                                                max_val   = 1,
                                                max_iters = param['max_iter'],
                                                _type     = param['perturbation_type'],
                                                _loss     = 'cross_entropy')

        elif param['attack_type'] == "deep_fool":
            attack = TorchAttackDeepFool(model=model)

        elif param['attack_type'] == "cw":
            attack = TorchAttackCWL2( model=model)

        else:
            print("Invalid attack_type in config file, please use 'fgsm' or add a new class in attacks_utils....")
            exit()

    # Train model
    if param['train']:
        print(f'Start training')
        _ = training(param, device, train_loader, test_loader, model, reg_model, teacher_model, optimizer, attack=None)

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
        eta = param['eta_min'] * (param['eta_max'] / param['eta_min']) ** ((param['test_epoch'] - 1) / (param['epochs'] - 1))

        # Launch testing
        test(param, model, reg_model, device, loader, param['test_epoch'], eta, attack)

    plt.show()


if __name__ == '__main__':
    main()
