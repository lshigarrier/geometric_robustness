import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
import psutil
import os
import wandb

from mnist_model import compute_jacobian, get_jacobian_bound
from mnist_utils import load_yaml, initialize
from attacks_utils import TorchAttackGaussianNoise, TorchAttackFGSM, TorchAttackPGD, TorchAttackPGDL2, TorchAttackDeepFool, TorchAttackCWL2
from attacks_vis import plot_hist, plot_side_by_side


def test(param, model, reg_model, device, test_loader, eta, attack=None, train=False):
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

            if param['compute_reg']:
                with torch.enable_grad():
                    # Ensure grad is on
                    data.requires_grad = True

                    # Forward pass
                    output = model(data)

                    # Compute regularization term
                    reg, _, _, _, _ = reg_model(data, output, device)

            else:
                # Forward pass
                output = model(data)

                # Do not compute regularization
                reg = torch.tensor(0)

            # Compute cross entropy
            entropy = F.cross_entropy(output, target)

            if param['reg']:
                # Loss with regularization
                loss = (1 - eta) * entropy + eta * reg

            else:
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
            if not train and param['adv_test'] and correct_mask.any():
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

                fooled_mask = adv_pred.ne(target[correct_mask].view_as(adv_pred)).view(-1)
                if fooled_mask.sum().item() != 0:

                    # Save a single attack
                    if param['save_an_image']:
                        param['save_an_image'] = False
                        # Random sample
                        idx = torch.randint(fooled_mask.sum(), (1,)).item()

                        plot_side_by_side(img       = data[correct_mask][fooled_mask][idx],
                                          adv_img   = adv_data[fooled_mask][idx],
                                          pred      = pred[correct_mask][fooled_mask][idx],
                                          adv_pred  = adv_pred[fooled_mask][idx],
                                          title     = param['perturbation_type'].upper() + ": " + str(param['budget']) + " " + param['attack_type'].upper(),
                                          save_path = "img/attacks/" + param['attack_type'] + "_" + param['perturbation_type'] + "_" + str(param['budget']) + ".png")

            ## Display results
            # ---------------------------------------------------------------- #
            if not train and param['verbose'] and (batch_idx % param['log_interval'] == 0):
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
    if not train and param['adv_test']:
        print('Test set: Average loss: {:.4f}, Average cross entropy: {:.4f}, Average reg: {:.4f}, '
              'Accuracy: {}/{} ({:.0f}%), Robust accuracy: {}/{} ({:.0f}%)\n'.format(
               test_loss, test_entropy, test_reg,
               correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset),
               adv_correct, correct, 100. * adv_correct / correct))

        wandb.log({ 'Test Loss'         : test_loss,
                    'Test Avg CE'       : test_entropy,
                    'Avg Reg'           : test_reg,
                    'Correct'           : correct,
                    'Total Tested'      : len(test_loader.dataset),
                    'Accuracy'          : 100. * correct / len(test_loader.dataset),
                    'Adv Correct'       : adv_correct,
                    'Adv Total Tested'  : correct,
                    'Adv Accuracy'      : 100. * adv_correct / correct})

    else:
        print('Test set: Average loss: {:.4f}, Average cross entropy: {:.4f}, Average reg: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, test_entropy, test_reg,
            correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
    if not train and param['test_bound']:
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
    return test_loss, test_entropy, test_reg, adv_correct


def one_run():
    # Load configurations
    param = load_yaml('test_geo_reg_conf')

    # Set random seed
    torch.manual_seed(param['seed'])

    # Initialize wandb
    if param["run_sweep"]:
        wandb.init()

        # Convert WandB config
        for key, value in dict(wandb.config).items():
            param[key] = value

    else:
        wandb.init( project = param["wandb_project_name"],
                    entity  = "geometric_robustness",
                    config  = param)

    # Declare CPU/GPU usage
    if param['gpu_number'] is not None:
        os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = param['gpu_number']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using {device}')

    # Load data and model
    light_train_loader, test_loader, model, reg_model = initialize(param, device, train=False)

    # Load attacker
    attack = None
    if param['adv_test']:

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

    # Test model
    print(f'Start testing')
    # Set data loader
    if param['loader'] == 'test':
        loader = test_loader
        print('Using test loader')
    else:
        loader = light_train_loader
        print('Using light train loader')

    # Launch testing
    if param['test_bound']:
        test_bound, test_bound_robust, test_bound_nonrobust = test(param, model, reg_model, device, loader, param['eta'], attack)
        _ = plot_hist(test_bound, "All points", "Bound minus max singular value", "Number")
        _ = plot_hist(test_bound_robust, "Robust points", "Bound minus max singular value", "Number")
        _ = plot_hist(test_bound_nonrobust, "Non-robust points", "Bound minus max singular value", "Number")
    else:
        _, _, _, adv_correct = test(param, model, reg_model, device, loader, param['eta'], attack)
        with open('outputs/jacobian/robust_acc/'+param['model'][:-2]+'txt', 'a') as file:
            file.write(str(adv_correct)+'\n')

    if param['plot']:
        plt.show()


def main():
    # Detect anomaly in autograd
    torch.autograd.set_detect_anomaly(True)

    # Load configurations
    param = load_yaml('test_geo_reg_conf')

    # Run WandB sweep
    if param["run_sweep"]:
        # Load sweep config
        sweep_config = load_yaml('attack_sweep')

        # Initialize sweep
        sweep_id = wandb.sweep(sweep  =sweep_config,
                               project=param["wandb_project_name"],
                               entity ="geometric_robustness")

        # Start sweep agent
        wandb.agent(sweep_id, function=one_run)

    # Run as normal
    else:
        one_run()


if __name__ == '__main__':
    main()
