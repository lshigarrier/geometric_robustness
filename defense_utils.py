import os
import torch
from mnist_utils import load_yaml
import torch.nn as nn
import numpy as np
from mnist_model import Lenet
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
plt.style.use('seaborn-deep')
from torchvision import datasets, transforms

def parseval_orthonormal_constraint(model, beta = 0.0003, percent_of_rows = 0.3):
    # From paper: https://arxiv.org/pdf/1704.08847.pdf
    with torch.no_grad():
        state_dict = model.state_dict()
        for name, param in model.named_parameters():
            module = getattr(model, name.split('.')[0])

            # Scaling factor for 2D convs in https://www.duo.uio.no/bitstream/handle/10852/69487/master_mathialo.pdf?sequence=1
            if isinstance(module, nn.Conv2d):
                k = float(module.kernel_size[0])
            else:
                k = 1.0

            # Constraint
            if 'weight' in name:
                # Flatten
                w = param.view(param.size(0), -1) if 'conv' in name else param

                # Sample 30% of Rows
                S = torch.from_numpy(np.random.binomial(1, percent_of_rows, (w.size(0)))).bool()

                # Update from orginal paper
                w[S,:] = ((1 + beta) * w[S,:]) - ((beta / k) * torch.mm(w[S,:], torch.mm(w[S,:].T, w[S,:])))

                # Set parameters
                state_dict[name] = w.view_as(param)

        model.load_state_dict(state_dict)

    return model

def check_parseval_tightness(model, save_path, comparison_model = None):
    # From paper: https://arxiv.org/pdf/1704.08847.pdf

    num_layers = len(list(model.named_parameters()))
    named_parameters = iter(model.named_parameters())

    if comparison_model:
        assert num_layers == len(list(comparison_model.named_parameters())), "Comparison model must have same number of parameters..."
        comparison_named_parameters = iter(comparison_model.named_parameters())

    
    with torch.no_grad():
        for i in range(num_layers):
            
            name, param = next(named_parameters)

            if comparison_model:
                comparison_name, comparison_param = next(comparison_named_parameters)

            if 'weight' in name:
                # Flatten
                w = param.view(param.size(0), -1) if 'conv' in name else param
                singular_values = torch.linalg.svdvals(w).cpu().numpy()

                fig = plt.figure()
                
                if comparison_model:
                    comparison_w = comparison_param.view(comparison_param.size(0), -1) if 'conv' in comparison_name else comparison_param
                    comparison_singular_values = torch.linalg.svdvals(comparison_w).cpu().numpy()

                    print("Regular", len(singular_values), "Parseval", len(comparison_singular_values))

                    # Plot
                    bins = np.linspace(0, max(max(singular_values), max(comparison_singular_values)), 30)
                    plt.hist([singular_values, comparison_singular_values], bins=bins, range = (0, max(max(singular_values), max(comparison_singular_values))), alpha=0.6, label=["Regular", "Parseval"])
                    plt.legend(loc='upper right')
                else:
                    bins = np.linspace(0, max(singular_values), 30)
                    plt.hist(singular_values, bins=bins, range = (0, max(singular_values)), alpha=0.6, label=name, color='blue')

                fig.savefig(save_path + "_layer_" + name + ".png")
        

if __name__ == '__main__':

    # Load configurations
    param = load_yaml('param_iso')

    # Declare CPU/GPU useage
    if param['gpu_number'] is not None:
        os.environ["CUDA_DEVICE_ORDER"]     = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"]  = param['gpu_number']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load architecture
    model = Lenet(param).to(device)
    parseval_model = Lenet(param).to(device)        

    # load parameters from file
    model.load_state_dict(torch.load(f'models/isometry/{param["name"]}/{"baseline.pt"}', map_location='cpu'))
    model.eval()

    parseval_model.load_state_dict(torch.load(f'models/isometry/{param["name"]}/{"00010.pt"}', map_location='cpu'))
    parseval_model.eval()

    # Check tightness
    check_parseval_tightness(   model = model, 
                                comparison_model = parseval_model,
                                save_path = "img/parseval_tightness/comparison")

    