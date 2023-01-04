import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import time
import psutil
import os

from mnist_utils import load_yaml, initialize, moving_average
# from mnist_model import JacobianReg
from test_geo_reg import test
from attacks_utils import TorchAttackGaussianNoise, TorchAttackFGSM, TorchAttackPGD, TorchAttackPGDL2, TorchAttackDeepFool, TorchAttackCWL2
from defense_utils import parseval_orthonormal_constraint
from attacks_vis import plot_curves


def train(param, model, reg_model, teacher_model, device, train_loader, optimizer, epoch, eta, attack=None):
    # Initialize variables
    epoch_loss    = []
    epoch_entropy = []
    epoch_reg     = []
    # epoch_jac_reg = []
    epoch_norm    = []
    # epoch_hold    = []
    # epoch_frob    = []
    # epoch_bound   = []

    # Make model stochastic and compute gradient graph
    model.train()

    # Cycle through data
    tic = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        # Push to GPU/CPU
        data, target = data.to(device), target.to(device)

        # Adversarial train
        if param['adv_train']:
            # Update attacker
            attack.model = model
            attack.set_attacker()

            # Generate attacks
            data = attack.perturb(data, target)

        # Ensure grad is on
        data.requires_grad = True

        # Forward pass
        output = model(data)

        # Compute regularization and norms
        if param['compute_reg']:
            reg, norm = reg_model(data, output, device)
            # reg, norm, norm_hold, norm_frob, bound = reg_model(data, output, device)
            # if param['reg_type'] == 'isometry':
            #    jac_reg, norm, norm_hold, norm_frob, bound = JacobianReg(param['epsilon'], barrier=param['barrier'])(data, output, device)
        else:
            reg       = torch.tensor(0)
            norm      = torch.tensor(0)
            # norm_hold = torch.tensor(0)
            # norm_frob = torch.tensor(0)
            # bound     = torch.tensor(0)

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
        # if param['reg_type'] == 'isometry':
        #    epoch_jac_reg.append(jac_reg.item())
        epoch_norm.append(norm.item())
        # epoch_hold.append(norm_hold.item())
        # epoch_frob.append(norm_frob.item())
        # epoch_bound.append(bound.item())


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
    return epoch_loss, epoch_entropy, epoch_reg, epoch_norm
    # return epoch_loss, epoch_entropy, epoch_reg, epoch_jac_reg, epoch_norm, epoch_hold, epoch_frob, epoch_bound


def training(param, device, train_loader, test_loader, model, reg_model, teacher_model, optimizer, attack=None):
    ## Initialize
    # ---------------------------------------------------------------------- #
    # Initiate variables
    loss_list, entropy_list, reg_list, norm_list = [], [], [], []
    # jac_reg_list, hold_list, frob_list, bound_list = [], [], [], []
    test_loss_list, test_entropy_list, test_reg_list = [], [], []
    # ---------------------------------------------------------------------- #

    ## Cycle through epochs
    # ---------------------------------------------------------------------- #
    for epoch in range(1, param['epochs'] + 1):

        # Train
        epoch_loss, epoch_entropy, epoch_reg, epoch_norm = train(param, model, reg_model, teacher_model, device, train_loader, optimizer, epoch, param['eta'], attack)

        # Validate
        test_loss, test_entropy, test_reg, _ = test(param, model, reg_model, device, test_loader, param['eta'], attack=attack, train=True)

        # Checkpoint model weights
        if epoch % param['save_step'] == 0:
            torch.save(model.state_dict(), f'models/{param["name"]}/epoch_{epoch:02d}.pt')

        # Collect statistics
        loss_list    = [*loss_list, *epoch_loss]
        entropy_list = [*entropy_list, *epoch_entropy]
        reg_list     = [*reg_list, *epoch_reg]
        norm_list    = [*norm_list, *epoch_norm]
        # hold_list    = [*hold_list, *epoch_hold]
        # frob_list    = [*frob_list, *epoch_frob]
        # bound_list   = [*bound_list, *epoch_bound]
        # jac_reg_list = [*jac_reg_list, *epoch_jac_reg]

        test_loss_list.append(test_loss)
        test_entropy_list.append(test_entropy)
        test_reg_list.append(test_reg)

    # Moving average
    loss_list      = moving_average(loss_list, 50)
    entropy_list   = moving_average(entropy_list, 50)
    reg_list       = moving_average(reg_list, 50)
    norm_list_avg  = moving_average(norm_list, 50)
    # hold_list      = moving_average(hold_list, 50)
    # frob_list      = moving_average(frob_list, 50)
    # bound_list_avg = moving_average(bound_list, 50)
    # jac_reg_list   = moving_average(jac_reg_list, 50)
    if param['plot']:
        # Display plot
        figs = [
            plot_curves([loss_list], [None], "Training Loss", "Batch", "Loss"),
            plot_curves([entropy_list], [None], "Training Cross Entropy", "Batch", "Cross entropy"),
            plot_curves([reg_list], [None], "Training Regularization", "Batch", "Regularization"),
            plot_curves([norm_list_avg], ["Holder"], "Jacobian matrix norm", "Batch", "Norm"),
            plot_curves([test_loss_list], [None], "Test Loss", "Epoch", "Loss"),
            plot_curves([test_entropy_list], [None], "Test Cross Entropy", "Epoch", "Cross Entropy"),
            plot_curves([test_reg_list], [None], "Test Regularization", "Epoch", "Regularization")
        ]
        '''
            plot_curves([norm_list_avg, hold_list],
                        ["Spectral", "Holder"],
                        "Jacobian matrix norms", "Batch", "Norm"),
            plot_curves([frob_list], [None], "Frobenius Norm", "Batch", "Norm"),
            plot_curves([bound_list_avg], [None], "Bound", "Batch", "Norm"),
            plot_curves([[bound_list[i] - norm_list[i] for i in range(len(bound_list))]], [None], "Bound minus True Norm", "Batch", "bound - norm"),
        '''
        # if param['reg_type'] == 'isometry':
        #    figs.append(plot_curves([jac_reg_list], [None], "Jacobian regularization", "Batch", "Regularization"))

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
        '''
                 "_frob_norm.png",
                 "_bound.png",
                 "_bound-norm.png",
        '''
        paths = ["_train_loss.png",
                 "_train_entropy.png",
                 "_train_reg.png",
                 "_norms.png",
                 "_test_loss.png",
                 "_test_entropy.png",
                 "_test_reg.png"
                 ]
        if param['reg_type'] == 'isometry':
            paths.append("_jac_reg.png")
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
        prefix = 'train_conf/'
        conf_files = [
            'baseline',
            'iso01',
            'iso42',
            'iso84',
            'jac01',
            'jac42',
            'jac84',
            'adv_train',
            'noise_train',
            'parseval',
            'only_reg',
            'max_eig',
            'distillation'
        ]
        for conf_file in conf_files:
            print('=' * 101)
            for i in range(1, 11):
                print('-' * 101)
                param = load_yaml(prefix+conf_file+'_conf')
                param['seed'] = i
                param['name'] += f'r{i}'
                if conf_file == 'distillation':
                    os.system(f'cp ./models/train_sweep/baseline/r{i}/epoch_10.pt ./models/train_sweep/distillation/r{i}/teacher.pt')
                one_run(param)

    else:
        one_run(param)


if __name__ == '__main__':
    main()
