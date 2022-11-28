import torch
import torch.optim as optim
import yaml
import argparse

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from mnist_model import Lenet, IsometryReg, JacobianReg


def moving_average(array, window_size):
    i = 0
    moving_averages = []
    while i < len(array) - window_size + 1:
        this_window = array[i: i + window_size]
        window_average = sum(this_window) / window_size
        moving_averages.append(window_average)
        i += 1
    return moving_averages


def load_yaml(file_name=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='test_geo_reg_conf')
    args = parser.parse_args()
    if file_name:
        yaml_file = f'config/{file_name}.yml'
    else:
        yaml_file = f'config/{args.yaml}.yml'
    with open(yaml_file) as file:
        param = yaml.load(file, Loader=yaml.FullLoader)
    return param


def initialize(param, device, train):
    ## Load TEST batch size
    # -------------------------------------------------------------- #
    # Use evaluation batch size
    if not train:
        test_kwargs = {'batch_size': param['testing_batch_size']}
        print(f'Using testing batch size for test loader')

    # Use validation batch size
    else:
        test_kwargs = {'batch_size': param['test_batch_size']}
        print(f'Using training batch size for test loader')

    ## Load TRAIN batch size
    # -------------------------------------------------------------- #
    # use evaluation batch size
    if not train:
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
    if not train:
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

    if train:
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
        return train_loader, test_loader, model, reg_model, teacher_model, optimizer

    else:
        print('Initialization done')
        return light_train_loader, test_loader, model, reg_model
