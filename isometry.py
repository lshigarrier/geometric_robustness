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
from mnist_model import SoftLeNet, LogitLenet, Lenet
from mnist_utils import load_yaml
from attacks_utils import FastGradientSignUntargeted, TorchAttackDeepFool, TorchAttackCWL2, TorchAttackFGSM, TorchAttackPGD
from attacks_vis import plot_side_by_side

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
    # Add dimension
    data = data.unsqueeze(1)

    # Get stereographic projection
    transform = create_transform(model)

    # Calculate jacobian 
    jac = functorch.vmap(functorch.jacrev(transform), randomness='different')(data).to(device)

    # Fix dimensioning
    jac = jac.squeeze()
    if len(data) == 1:
        jac = jac.unsqueeze(0)
    jac = torch.reshape(jac, (jac.shape[0], jac.shape[1], -1))
    return jac


def change_matrix(output, epsilon, nb_class):
    # Coordinate change 
    change = output[:, nb_class-1]/torch.square(2*torch.sqrt(output[:, nb_class-1]) - torch.norm(output[:, :nb_class-1], p=1, dim=1))
    
    # Distance from center of simplex
    delta = 2*torch.acos(1/torch.sqrt(torch.tensor(nb_class)))

    # Diagonalize
    return functorch.vmap(torch.diag)(change.unsqueeze(1).repeat(1, nb_class-1))*delta**2/epsilon**2


def iso_loss_transform(output, target, data, epsilon, model, device, test_mode=False):
    # Number of classes
    nb_class = output.shape[1]

    # Calculate jacobian of stereographic projection
    jac = jacobian_transform(data, model, device)
    assert not torch.isnan(jac).any()

    # Gram matrix of jacobian
    jac = torch.bmm(jac, torch.transpose(jac, 1, 2))

    # Distance to a decision boundary in the probability simplex (delta/rho in docs)
    change = change_matrix(output, epsilon, nb_class)
    
    # Calculate standard loss
    cross_entropy = F.cross_entropy(output, target)

    # Regularization term (alpha in docs)
    reg = epsilon**2*torch.linalg.norm((jac - change).view(len(data), -1), dim=1).sum()/len(data)

    # Return 
    if test_mode:
        # print(f'cross entropy: {cross_entropy}\nreg: {reg}\njac*tjac: {jac}\nchange: {change}')
        return cross_entropy, reg, change

    else:
        return cross_entropy, reg


# -------------------------------------------- Training & Testing ------------------------------------------------------


def train(param, model, device, train_loader, optimizer, epoch, lmbda, teacher_model, attack=None):
    # Initiate variables
    epoch_loss      = 0
    epoch_entropy   = 0
    epoch_reg       = 0

    # Make model stochastic and compute gradient gragh
    model.train()

    # Display lambda value
    if param['verbose'] and param['reg']:
        print(f'Lambda: {lmbda}')

    # Cycle through data
    tic = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        # Push to GPU/CPU
        data, target = data.to(device), target.to(device)

        # Ensure grad is on and gradients set to zero
        data.requires_grad = True
        optimizer.zero_grad()

        # Adversarial train
        if param['adv_train']:
            # Update attacker
            attack.model = model
            attack.set_attacker()

            # Generate attacks
            data = attack.perturb(data, target)  

        # Forward pass
        output = model(data)

        # Compute loss
        if param['distill']:
            # Get soft-labels
            soft_labels = F.softmax(teacher_model(data, perform_softmax = False) / param["distill_temp"], -1)

            # Calculate loss with soft_labels
            cross_entropy = torch.sum(-soft_labels * torch.log(output), -1).mean()

            # Do not compute regularization
            reg =  torch.tensor(0)

            # Loss is only cross entropy
            loss = cross_entropy

        elif param['reg']:
            # Compute cross entropy loss and regularization term
            cross_entropy, reg = iso_loss_transform(output, target, data, param['epsilon_l2'], model, device)

            # Loss with regularization
            loss = (1 - lmbda) * cross_entropy + lmbda * reg

        else:
            # Compute cross entropy loss
            cross_entropy = F.cross_entropy(output, target)

            # Do not compute regularization
            reg =  torch.tensor(0)

            # Loss is only cross entropy
            loss = cross_entropy

        # Backpropogation
        assert not torch.isnan(loss).any()
        loss.backward()

        # Update model parameters
        optimizer.step()

        # Update running totals
        epoch_loss      += loss.item()*len(data)
        epoch_entropy   += cross_entropy.item()*len(data)
        epoch_reg       += reg.item()*len(data)

        # Display
        if param['verbose'] and (batch_idx % param['log_interval'] == 0):
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Cross Entropy: {:.6f}, Reg: {:.6f}'.format(
                epoch, batch_idx*len(data), len(train_loader.dataset), 100.*batch_idx/len(train_loader),
                loss.item(), cross_entropy.item(), reg.item()))
            print(f'Elapsed time (s): {time.time() - tic}')
            print(f'Memory usage (GB): {psutil.Process(os.getpid()).memory_info()[0] / (2. ** 30)}')
            tic = time.time()

    # Calculate results        
    epoch_loss      /= len(train_loader.dataset)
    epoch_entropy   /= len(train_loader.dataset)
    epoch_reg       /= len(train_loader.dataset)

    # Display
    if param['verbose']:
        print('Train set: Average loss: {:.4f}, Average cross entropy: {:.4f}, Average reg: {:.4f}'.format(
            epoch_loss, epoch_entropy, epoch_reg))
    
    # Return results
    return epoch_loss, epoch_entropy, epoch_reg


def test(param, model, device, test_loader, lmbda, attack=None):
    # Make model deterministic and turn off gradients
    model.eval()

    # Initialize variables
    test_loss       = 0
    test_entropy    = 0
    test_reg        = 0
    correct         = 0
    adv_correct     = 0
    adv_total       = 0
    hist_correct    = []
    hist_incorrect  = []
    tic             = time.time()

    ## Cycle through data
    #----------------------------------------------------------------#
    with torch.enable_grad() if param['adv_test'] else torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            # Push to device
            data, target = data.to(device), target.to(device)

            # Forward pass
            output = model(data)

            # Compute loss
            if param['reg']:
                # Compute cross entropy loss and regularization term
                cross_entropy, reg = iso_loss_transform(output, target, data, param['epsilon_l2'], model, device)

                # Loss with regularization
                loss = (1 - lmbda) * cross_entropy + lmbda * reg
            else:
                # Compute cross entropy loss
                cross_entropy = F.cross_entropy(output, target)

                # Do not compute regularization
                reg =  torch.tensor(0)

                # Loss is only cross entropy
                loss = cross_entropy

            # Running statistics
            test_loss       += loss.item()*len(data)
            test_entropy    += cross_entropy.item()*len(data)
            test_reg        += reg.item()*len(data)

            ## Check standard and adversarial accuracy
            #----------------------------------------------------------------#
            # If batch size is 1
            if len(data) == 1 and False:
                # Get prediction
                pred = output.argmax(dim=1, keepdim=True)[0]

                # Bool if prediction is correct
                correct_pred = pred.eq(target.view_as(pred)).item()
                correct += correct_pred

                # If correct test adversarial attack
                if param['adv_test'] and correct_pred:
                    # use predicted label as target label (or not)
                    # with torch.enable_grad():

                    # Generate attack
                    adv_data = attack.perturb(data, target)

                    # Feed forward
                    adv_output = model(adv_data)

                    # Get adversairal prediction
                    adv_pred = adv_output.argmax(dim=1, keepdim=True)

                    # Update running totals
                    adv_correct += adv_pred.eq(target.view_as(adv_pred)).sum().item()
                    adv_total += len(data)
                    if adv_pred:
                        # If unfooled
                        hist_correct.append(reg.item())
                    else:
                        # If fooled
                        hist_incorrect.append(reg.item())
            
            # If multiple images in batch
            else:
                # Get prediction
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

                # Running total of correct
                correct_mask = pred.eq(target.view_as(pred)).view(-1)
                correct += correct_mask.sum().item()

                # Test adversary
                if param['adv_test']:
                    # use predicted label as target label (or not)
                    # with torch.enable_grad():
                    data.requires_grad = True

                    # Generate attacks
                    adv_data = attack.perturb(data[correct_mask], target[correct_mask])  

                    # Feed forward
                    adv_output = model(adv_data)

                    # Get prediction
                    adv_pred = adv_output.argmax(dim=1, keepdim=True)

                    # Collect statistics
                    adv_correct += adv_pred.eq(target[correct_mask].view_as(adv_pred)).sum().item()  # pred or target
                    adv_total   = correct

                    
                    fooled_mask = adv_pred.ne(target[correct_mask].view_as(adv_pred)).view(-1)
                    if fooled_mask.sum().item() != 0:

                        # Save a single attack
                        if param['save_an_image']:
                            param['save_an_image'] = False
                            # Random sample
                            idx = torch.randint(fooled_mask.sum(), (1,)).item()
                            
                            plot_side_by_side(  img         = data[correct_mask][fooled_mask][idx], 
                                                adv_img     = adv_data[fooled_mask][idx], 
                                                pred        = pred[correct_mask][fooled_mask][idx], 
                                                adv_pred    = adv_pred[fooled_mask][idx],
                                                title       = param['perturbation_type'].upper() + ": " + str(param['budget']) + " " + param['attack_type'].upper(),
                                                save_path   = "img/attacks/" + param['attack_type'] + "_" + param['perturbation_type'] + "_" + str(param['budget']) + ".png")
            
            ## Display results
            #----------------------------------------------------------------#
            if not(param['train']) and param['verbose'] and (batch_idx % param['log_interval'] == 0):
                print('Test: {}/{} ({:.0f}%)\tLoss: {:.6f}, Cross Entropy: {:.6f}, Reg: {:.6f}'.format(
                    batch_idx * len(data), len(test_loader.dataset), 100. * batch_idx / len(test_loader),
                    loss.item(), cross_entropy.item(), reg.item()))
                print(f'Elapsed time (s): {time.time() - tic}')
                print(f'Memory usage (GB): {psutil.Process(os.getpid()).memory_info()[0] / (2. ** 30)}')
                tic = time.time()
    
    ## Calculate results, display and return
    #----------------------------------------------------------------#
    test_loss       /= len(test_loader.dataset)
    test_entropy    /= len(test_loader.dataset)
    test_reg        /= len(test_loader.dataset)
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
    ## Load TEST batch size
    #--------------------------------------------------------------#
    # use evaluation batch size
    if param['load']:
        test_kwargs = {'batch_size': param['testing_batch_size']}
        print(f'Using testing batch size for test loader')

    # use validation batch size
    else:
        test_kwargs = {'batch_size': param['test_batch_size']}
        print(f'Using training batch size for test loader')

    ## Load TRAIN batch size
    #--------------------------------------------------------------#
    # use evaluation batch size
    if param['load']:
        train_kwargs = {'batch_size': param['testing_batch_size']}
        print(f'Using testing batch size for train loader')

    # use train batch size
    else:
        train_kwargs = {'batch_size': param['batch_size']}
        print(f'Using training batch size for train loader')

    ## Machine settings
    #--------------------------------------------------------------#
    if torch.cuda.is_available():
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory' : True,
                       'shuffle'    : True}

        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    ## Load dataset from torchvision
    #--------------------------------------------------------------#
    # train set
    dataset1 = datasets.MNIST('./data/mnist', train=True, download=True, transform=transforms.ToTensor())

    # small train set
    subset = torch.utils.data.Subset(dataset1, range(1000))

    # test set
    dataset2 = datasets.MNIST('./data/mnist', train=False, transform=transforms.ToTensor())

    # create data loaders
    train_loader        = DataLoader(dataset1, **train_kwargs)
    light_train_loader  = DataLoader(subset, **train_kwargs)
    test_loader         = DataLoader(dataset2, **test_kwargs)

    ## Load model
    #--------------------------------------------------------------#
    # initalize network class

    model = Lenet(param).to(device)

    # load parameters from file
    if param['load']:
        print(f'Loading weights')
        model.load_state_dict(torch.load(f'models/isometry/{param["name"]}/{param["model"]}', map_location='cpu'))

        # make model deterministic and turn of gradient computations
        model.eval()

    else:
        print(f'Randomly initialized weights')

    # Load teacher model
    if param['distill']:
        # initalize network class
        teacher_model = Lenet(param).to(device)

        print(f'Loading weights onto teacher model')
        teacher_model.load_state_dict(torch.load(f'models/isometry/{param["name"]}/{param["model"]}', map_location='cpu'))

        # make model deterministic and turn of gradient computations
        teacher_model.eval()
    else:
        teacher_model = None

    # Set optimizer
    optimizer = optim.SGD(model.parameters(), lr=param['learning_rate'])

    print('Initialization done')
    return train_loader, light_train_loader, test_loader, model, optimizer, teacher_model


def training(param, device, train_loader, test_loader, model, optimizer, teacher_model, attack=None):
    ## Initialize
    #----------------------------------------------------------------------#
    # Initiate variables
    loss_list, entropy_list, reg_list = [], [], []
    test_loss_list, test_entropy_list, test_reg_list = [], [], []
    #----------------------------------------------------------------------#

    ## Cycle through epochs
    #----------------------------------------------------------------------#
    for epoch in range(1, param['epochs'] + 1):
        # Set lambda term
        # lmbda = param['lambda_min'] + (epoch - 1)/(param['epochs'] - 1)*(param['lambda_max'] - param['lambda_min'])
        lmbda = param['lambda_min'] * (param['lambda_max']/param['lambda_min'])**((epoch - 1)/(param['epochs'] - 1))

        # Train
        epoch_loss, epoch_entropy, epoch_reg = train(param, model, device, train_loader, optimizer, epoch, lmbda, teacher_model, attack=attack)

        # Validate
        test_loss, test_entropy, test_reg, _, _ = test(param, model, device, test_loader, lmbda, attack=attack)

        # Checkpoint model weights
        if epoch % param['save_step'] == 0:
            torch.save(model.state_dict(), f'models/isometry/{param["name"]}/{epoch:05d}.pt')

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

    # # Return
    # return fig1, fig2, fig3

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
    # Load configurations
    param = load_yaml('param_iso')

    # Set random seed
    torch.manual_seed(param['seed'])

    # Declare CPU/GPU useage
    if param['gpu_number'] is not None:
        os.environ["CUDA_DEVICE_ORDER"]     = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"]  = param['gpu_number']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using {device}')

    # Load data and model
    train_loader, light_train_loader, test_loader, model, optimizer, teacher_model = initialize(param, device)

    # Load attacker
    attack = None
    if param['adv_test'] or param['adv_train']:

        if param["attack_type"] == "fgsm":
            attack = TorchAttackFGSM(   model   = model,
                                        eps     = param['budget'])

        elif param['attack_type'] == "pgd":
            attack = TorchAttackPGD(model   = model,
                                    eps     = param['budget'],
                                    alpha   = param['alpha'],
                                    steps   = param['max_iter'],
                                    )

        elif param['attack_type'] == "deep_fool":
            attack = TorchAttackDeepFool(model = model)

        elif param['attack_type'] == "cw":
            attack = TorchAttackCWL2( model = model)

        else:
            print("Invalid attack_type in config file, please use 'fgsm' or add a new class in attacks_utils....")
            exit()

    # Train model
    if param['train']:
        print(f'Start training')
        _ = training(param, device, train_loader, test_loader, model, optimizer, teacher_model, attack=attack)

    # Test model
    else:
        print(f'Start testing')
        # testing_loss(param, device, train_loader, model)

        # Set data loader
        if param['loader'] == 'test':
            loader = test_loader
            print('Using test loader')
        else:
            loader = light_train_loader
            print('Using light train loader')

        # Compute jacobian
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
        
        # 
        else:
            # Compute lambda value
            lmbda = param['lambda_min'] * (param['lambda_max'] / param['lambda_min']) ** ((param['test_epoch'] - 1) / (param['epochs'] - 1))

            # Get regularization terms for correct and incorrect adversraial predictions 
            _, _, _, hist_correct, hist_incorrect = test(param, model, device, loader, lmbda, attack=attack)

            if param['reg']:
                # Get max regularization term used
                max_reg = min(max(hist_incorrect), max(hist_correct))

                # Plot 
                hist1 = plot_hist(hist_correct, f'Robust points\n{len(hist_correct)}', 'Regularization', 'Number of points', xmax=max_reg)
                hist2 = plot_hist(hist_incorrect, f'Non-robust points\n{len(hist_incorrect)}', 'Regularization', 'Number of points', xmax=max_reg)
    
    plt.show()


if __name__ == '__main__':
    main()
