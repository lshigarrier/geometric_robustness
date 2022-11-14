import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os

from geo_reg import initialize
from mnist_utils import load_yaml
from mnist_model import Lenet, compute_jacobian, get_jacobian_bound


def vis_init(param):
    # Load images
    data_robust = torch.load(param['data_robust_file'])
    adv_data_robust = torch.load(param['adv_data_robust_file'])
    data_nonrobust = torch.load(param['data_nonrobust_file'])
    adv_data_nonrobust = torch.load(param['adv_data_nonrobust_file'])

    # Image to plot
    if param['use_robust']:
        data = data_robust[param['data_index']].unsqueeze(0)
        adv_data = adv_data_robust[param['data_index']].unsqueeze(0)
    else:
        data = data_nonrobust[param['data_index']].unsqueeze(0)
        adv_data = adv_data_nonrobust[param['data_index']].unsqueeze(0)

    # Directions spanning the visualization plan
    if param['random_dir']:
        dir1 = F.normalize(torch.randn(param['dim']), dim=0).view_as(data)
    else:
        dir1 = F.normalize((adv_data - data).view(-1), dim=0).view_as(data)
    dir2 = torch.randn(param['dim'])
    # Orthonormalization
    dir2 = F.normalize(dir2 - torch.dot(dir1.view(-1), dir2.view(-1)) * dir1.view(-1), dim=0).view_as(data)

    # List of perturbations
    range_points = (param['nb_points'] - 1) // 2
    perturb = [param['scale'] * u / range_points * param['epsilon'] for u in range(-range_points, range_points + 1)]

    return data, dir1, dir2, perturb


def visualize(param, device, model, data, dir1, dir2, perturb):
    # Initialize heat maps
    sv_heat_map = np.zeros((param['nb_points'], param['nb_points']))
    pred_heat_map = np.zeros((param['nb_points'], param['nb_points']))
    soft_heat_map = np.zeros((param['nb_points'], param['nb_points']))

    for i in range(len(perturb)):
        for j in range(len(perturb)):
            iteration = param['nb_points'] * i + j
            if param['verbose'] and iteration % 100 == 0:
                print(iteration)
            current_data = data.clone() + dir1 * perturb[i] + dir2 * perturb[j]
            with torch.enable_grad():
                # Ensure grad is on
                current_data.requires_grad = True
                current_data.grad = None

                # Forward pass
                current_output = model(current_data)

                # Compute jacobian and bound
                jac = compute_jacobian(current_data, current_output, device)
                bound = get_jacobian_bound(current_output, param['epsilon'])
                sv_max = torch.max(torch.linalg.svdvals(jac), dim=1)[0]

                sv_heat_map[i, j] = sv_max.item() - bound.item()
                pred_heat_map[i, j] = (current_output.argmax(dim=1, keepdim=True)[0]).item()
                soft_heat_map[i, j] = (current_output.max(dim=1, keepdim=True)[0]).item()

    # Colors
    n_colors = len(np.unique(pred_heat_map))
    cmap = plt.get_cmap('tab10', n_colors)
    # Plot
    fig1, ax1 = plt.subplots()
    hm1 = ax1.imshow(sv_heat_map, cmap='viridis', interpolation='nearest', origin='lower')
    fig1.colorbar(hm1)
    fig2, ax2 = plt.subplots()
    hm2 = ax2.imshow(soft_heat_map, cmap='viridis', interpolation='nearest', origin='lower')
    fig2.colorbar(hm2)
    fig3, ax3 = plt.subplots()
    ax3.imshow(pred_heat_map, cmap=cmap, interpolation='nearest', origin='lower')
    return fig1, fig2, fig3


def main():
    # Detect anomaly in autograd
    torch.autograd.set_detect_anomaly(True)

    # Load configurations
    param = load_yaml('config_vis')

    # Set random seed
    torch.manual_seed(param['seed'])

    # Initialize the visualization
    data, dir1, dir2, perturb = vis_init(param)

    # Declare CPU/GPU usage
    if param['gpu_number'] is not None:
        os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = param['gpu_number']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using {device}')

    # Load model
    model = Lenet(param).to(device)
    model.load_state_dict(torch.load(f'models/{param["name"]}/{param["model"]}', map_location='cpu'))

    # Create the figures
    _ = visualize(param, device, model, data, dir1, dir2, perturb)
    plt.show()


if __name__ == '__main__':
    main()