import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import time
import psutil
import os

from mnist_utils import load_yaml, initialize, moving_average
from test_geo_reg import test
from attacks_utils import TorchAttackGaussianNoise, TorchAttackFGSM, TorchAttackPGD, TorchAttackPGDL2, TorchAttackDeepFool, TorchAttackCWL2
from defense_utils import parseval_orthonormal_constraint
from attacks_vis import plot_curves


def train(param, model, reg_model, teacher_model, device, train_loader, optimizer, epoch, eta, attack=None):
    # Initialize variables
    epoch_loss    = []
    epoch_entropy = []
    epoch_reg     = []
    epoch_norm    = []
    epoch_hold    = []
    epoch_frob    = []
    epoch_bound   = []

    # Make model stochastic and compute gradient graph
    model.train()

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

        # Compute regularization and norms
        if param['compute_reg']:
            reg, norm, norm_hold, norm_frob, bound = reg_model(data, output, device)
        else:
            reg       = torch.tensor(0)
            norm      = torch.tensor(0)
            norm_hold = torch.tensor(0)
            norm_frob = torch.tensor(0)
            bound     = torch.tensor(0)

        ## Compute loss
        # Distillation
        if param['distill']:
            ## Sanity check that this method is equivalent to original criterion
            # batch_size = labels.size(0)
            # label_onehot = torch.FloatTensor(batch_size, data.num_classes)
            # label_onehot.zero_()
            # label_onehot.scatter_(1, labels.view(-1, 1), 1)
            # print("One Hot", label_onehot[0])
            # print(torch.sum(-label_onehot * F.log_softmax(outputs, -1), -1).mean())

            soft_labels = F.softmax(teacher_model(data) / param["distill_temp"], -1)
            # numerical stability
            c = soft_labels.shape[1]
            soft_labels = soft_labels * (1 - c * 1e-6) + 1e-6
            # torch.log(output) or F.log_softmax(output, -1) ?
            entropy = torch.sum(-soft_labels * torch.log_softmax(output, -1), -1).mean()

            # Loss is only cross entropy
            loss = entropy

        # Suppress maximum eigenvalue
        elif param['max_eig']:
            # Compute regularization term and cross entropy loss
            c           = output.shape[1]
            new_output  = F.softmax(output, dim=1) * (1 - c * 1e-6) + 1e-6  # for numerical stability
            max_eig_reg = torch.sum(1/new_output, dim=1).mean()
            entropy     = F.cross_entropy(output, target)

            # Loss is only cross entropy
            loss = entropy + eta * max_eig_reg

        # Jacobian regularization only
        elif param['only_reg']:
            # Compute cross entropy loss
            entropy = F.cross_entropy(output, target)

            # Loss with regularization
            loss = (1 - eta) * entropy + eta * norm

        # Geometric regularization
        elif param['reg']:
            # Compute cross entropy loss
            entropy = F.cross_entropy(output, target)

            # Loss with regularization
            loss = (1 - eta) * entropy + eta * reg

        # Baseline
        else:
            # Compute cross entropy loss
            entropy = F.cross_entropy(output, target)

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
        # epoch_loss    += loss.item()*len(data)
        # epoch_entropy += entropy.item()*len(data)
        # epoch_reg     += reg.item()*len(data)
        epoch_loss.append(loss.item())
        epoch_entropy.append(entropy.item())
        epoch_reg.append(reg.item())
        epoch_norm.append(norm.item())
        epoch_hold.append(norm_hold.item())
        epoch_frob.append(norm_frob.item())
        epoch_bound.append(bound.item())


        # Display
        if param['verbose'] and (batch_idx % param['log_interval'] == 0):
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Cross Entropy: {:.6f}, Reg: {:.6f}'.format(
                epoch, batch_idx*len(data), len(train_loader.dataset), 100.*batch_idx/len(train_loader),
                loss.item(), entropy.item(), reg.item()))
            print(f'Elapsed time (s): {time.time() - tic}')
            print(f'Memory usage (GB): {psutil.Process(os.getpid()).memory_info()[0] / (2. ** 30)}')
            tic = time.time()

    # Calculate results
    # epoch_loss    /= len(train_loader.dataset)
    # epoch_entropy /= len(train_loader.dataset)
    # epoch_reg     /= len(train_loader.dataset)

    # Display
    if param['verbose']:
        print('Train set: Average loss: {:.4f}, Average cross entropy: {:.4f}, Average reg: {:.4f}'.format(
            np.mean(epoch_loss), np.mean(epoch_entropy), np.mean(epoch_reg)))

    # Return results
    return epoch_loss, epoch_entropy, epoch_reg, epoch_norm, epoch_hold, epoch_frob, epoch_bound


def training(param, device, train_loader, test_loader, model, reg_model, teacher_model, optimizer, attack=None):
    ## Initialize
    # ---------------------------------------------------------------------- #
    # Initiate variables
    loss_list, entropy_list, reg_list = [], [], []
    norm_list, hold_list, frob_list, bound_list = [], [], [], []
    test_loss_list, test_entropy_list, test_reg_list = [], [], []
    # ---------------------------------------------------------------------- #

    ## Cycle through epochs
    # ---------------------------------------------------------------------- #
    for epoch in range(1, param['epochs'] + 1):

        # Train
        epoch_loss, epoch_entropy, epoch_reg, epoch_norm, epoch_hold, epoch_frob, epoch_bound = train(param, model, reg_model, teacher_model, device, train_loader, optimizer, epoch, param['eta'], attack)

        # Validate
        test_loss, test_entropy, test_reg = test(param, model, reg_model, device, test_loader, param['eta'], attack=attack, train=True)

        # Checkpoint model weights
        if epoch % param['save_step'] == 0:
            torch.save(model.state_dict(), f'models/{param["name"]}/{epoch:05d}.pt')

        # Collect statistics
        loss_list    = [*loss_list, *epoch_loss]
        entropy_list = [*entropy_list, *epoch_entropy]
        reg_list     = [*reg_list, *epoch_reg]
        norm_list    = [*norm_list, *epoch_norm]
        hold_list    = [*hold_list, *epoch_hold]
        frob_list    = [*frob_list, *epoch_frob]
        bound_list   = [*bound_list, *epoch_bound]

        test_loss_list.append(test_loss)
        test_entropy_list.append(test_entropy)
        test_reg_list.append(test_reg)

    # Moving average
    loss_list      = moving_average(loss_list, 50)
    entropy_list   = moving_average(entropy_list, 50)
    reg_list       = moving_average(reg_list, 50)
    norm_list_avg  = moving_average(norm_list, 50)
    hold_list      = moving_average(hold_list, 50)
    frob_list      = moving_average(frob_list, 50)
    bound_list_avg = moving_average(bound_list, 50)
    if param['plot']:
        # Display plot
        figs = [
            plot_curves([loss_list], [None], "Training Loss", "Batch", "Loss"),
            plot_curves([entropy_list], [None], "Training Cross Entropy", "Batch", "Cross entropy"),
            plot_curves([reg_list], [None], "Training Regularization", "Batch", "Regularization"),
            plot_curves([norm_list_avg, hold_list],
                        ["Spectral", "Holder"],
                        "Jacobian matrix norms", "Batch", "Norm"),
            plot_curves([frob_list], [None], "Frobenius Norm", "Batch", "Norm"),
            plot_curves([bound_list_avg], [None], "Bound", "Batch", "Norm"),
            plot_curves([[bound_list[i] - norm_list[i] for i in range(len(bound_list))]], [None], "Bound minus True Norm", "Batch", "bound - norm"),
            plot_curves([test_loss_list], [None], "Test Loss", "Epoch", "Loss"),
            plot_curves([test_entropy_list], [None], "Test Cross Entropy", "Epoch", "Cross Entropy"),
            plot_curves([test_reg_list], [None], "Test Regularization", "Epoch", "Regularization")
        ]

        return figs
    return None


def one_run(param):
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
    figs = training(param, device, train_loader, test_loader, model, reg_model, teacher_model, optimizer, attack=attack)

    if param['plot']:
        prefix = './outputs/'
        name = param['name'].split('/')[1]
        paths = ["_train_loss.png",
                 "_train_entropy.png",
                 "_train_reg.png",
                 "_norms.png",
                 "_frob_norm.png",
                 "_bound.png",
                 "_bound-norm.png",
                 "_test_loss.png",
                 "_test_entropy.png",
                 "_test_reg.png"
                 ]
        for i in range(len(figs)):
            figs[i].savefig(prefix+name+paths[i])
    #    plt.show()


def main():
    # Detect anomaly in autograd
    torch.autograd.set_detect_anomaly(True)

    # Load configurations
    param = load_yaml('train_geo_reg_conf')

    # Loop
    if param['loop']:
        prefix = 'jacobian/'
        # NO ADVERSARIAL TRAINING WHILE I CAN'T INSTALL TORCHATTACKS ON DORMAMMU
        for i in range(2,7):
            print('-' * 102)
            if i == 0:
                print('baseline_4')
            if i == 1:
                # baseline with another seed
                print('baseline_5')
                param['name'] = prefix + 'baseline_5'
                param['seed'] = 42
            elif i == 2:
                # jacobian reg with medium eps
                print('jac_13')
                param['name'] = prefix + 'jac_13'
                param['reg'] = True
            elif i == 3:
                # jacobian reg with small eps
                print('jac_12')
                param['name'] = prefix + 'jac_12'
                param['epsilon'] = 0.1
            elif i == 4:
                # jacobian reg with large eps
                print('jac_14')
                param['name'] = prefix + 'jac_14'
                param['epsilon'] = 8.4
            elif i == 5:
                # suppress max eigenvalue
                print('max_eig_2')
                param['name'] = prefix + 'max_eig_2'
                param['max_eig'] = True
                param['reg'] = False
                param['eta'] = 0.1
            elif i == 6:
                # regularization only
                print('only_reg_1')
                param['name'] = prefix + 'only_reg_1'
                param['only_reg'] = True
                param['max_eig'] = False
                param['eta'] = 0.03
            elif i == 7:
                # distillation
                print('distill_2_baseline_4')
                os.system('cp ./models/jacobian/baseline_4/00010.pt ./models/jacobian/distill_baseline_4/teacher.pt')
                param['name'] = prefix + 'distill_2_baseline_4'
                param['distill'] = True
                param['max_eig'] = False
            elif i == 8:
                # Parseval network
                print('parseval_2')
                param['name'] = prefix + 'parseval_2'
                param['parseval_train'] = True
                param['distill'] = False
            elif i == 8:
                # adversarial training
                print('adv_train_1')
                param['name'] = prefix + 'adv_train_1'
                param['adv_train'] = True
                param['reg'] = False
            one_run(param)

    else:
        one_run(param)


if __name__ == '__main__':
    main()
