import torch
import os
from autoattack import AutoAttack
from mnist_utils import load_yaml, initialize


def one_auto_attack(param, device):
    # Load data and model
    light_train_loader, test_loader, model, reg_model = initialize(param, device, train=False)

    # Create adversary
    adversary = AutoAttack(model,
                           norm    =param['norm'],
                           eps     =param['epsilon'],
                           log_path=param['save_dir']+'log_'+param['model'][:-2]+'txt',
                           version =param['version'])

    # Create images and labels
    lx = []
    ly = []
    for (x, y) in test_loader:
        lx.append(x)
        ly.append(y)
    x_test = torch.cat(lx, 0)
    y_test = torch.cat(ly, 0)

    # Run attack and save images
    with torch.no_grad():
        adv_complete = adversary.run_standard_evaluation(x_test, y_test, bs=param['testing_batch_size'])
        torch.save({'adv_complete': adv_complete},
                   f"{param['save_dir']}/{param['model'][:-3]}_{param['norm']}_eps_{param['epsilon']:.2f}_v_{param['version']}.pth")


def main():
    files = ['baseline_4_5.pt',
             'baseline_5_3.pt',
             'jac_12_2.pt',
             'jac_13_3.pt',
             'jac_14_4.pt',
             'iso_9_2.pt',
             'iso_10_2.pt',
             'iso_11_3.pt',
             'iso_12_2.pt',
             'iso_13_2.pt',
             'iso_14_3.pt',
             'distill_1_baseline_4_8.pt',
             'max_eig_2_8.pt',
             'only_reg_1_7.pt',
             'parseval_1_8.pt',
             'adv_train_1_10.pt']

    # Load configurations
    param = load_yaml('auto_attack_conf')

    # Declare CPU/GPU usage
    if param['gpu_number'] is not None:
        os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = param['gpu_number']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using {device}')

    # Loop
    for file in files:
        param['model'] = file
        print('-'*101)
        print(param['model'][:-3])
        one_auto_attack(param, device)


if __name__ == '__main__':
    main()