import os
import torch
from mnist_utils import load_yaml
import torch.nn as nn
import numpy as np
from mnist_model import Lenet
import matplotlib.pyplot as plt


def parseval_orthonormal_constraint(model, beta = 0.0003):
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
                w = param.view(-1,1)

                # Sample 30% of Rows
                S = torch.from_numpy(np.random.binomial(1, 0.3, (w.size(0)))).bool()

                # Update from orginal paper
                w[S] = ((1 - beta) * w[S]) - ((beta / k) * torch.mm(w[S], torch.mm(w[S].T, w[S])))

                # Set parameters
                state_dict[name] = w.view_as(param)

        model.load_state_dict(state_dict)

    return model


def check_parseval_tightness(model, save_path):
    # From paper: https://arxiv.org/pdf/1704.08847.pdf

    fig = plt.figure()
    count = 0
    max_singular_value = []
    with torch.no_grad():
        state_dict = model.state_dict()
        for name, param in model.named_parameters():
            # Constraint
            if 'weight' in name:
                # Flatten
                w = param.view(-1,1)

                # Sample 30% of Rows
                p = min(800 / w.size(0), 1)
                S = torch.from_numpy(np.random.binomial(1, p, (w.size(0)))).bool()

                # Update from orginal paper
                wTw_eigvalues, _ = torch.linalg.eig(torch.mm(w[S], w[S].T))

                a = wTw_eigvalues.real.cpu().tolist()
                max_singular_value.append(max(a))

                # if count == 0:
                    
                #     # print(, min(a), np.mean(a), np.std(a))
                #     # print(torch.min(wTw_eigvalues.real), torch.max(wTw_eigvalues.real), wTw_eigvalues.size(), wTw_eigvalues.real.mean().item(), wTw_eigvalues.real.std().item())
                #     # exit()
                #     plt.hist(max(a), bins=1, range = (0,max(a)), alpha=0.6, label=name)
                #     fig.savefig(save_path)
                #     break

                count += 1
        # plt.legend(loc='upper right')
        plt.hist(max_singular_value, bins=40, range = (0,max(max_singular_value) + 2), alpha=0.6, label=name)
        fig.savefig(save_path)
        print(max_singular_value)


def main():
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
    model.load_state_dict(torch.load(f'models/{param["name"]}/{"baseline.pt"}', map_location='cpu'))
    model.eval()

    parseval_model.load_state_dict(torch.load(f'models/{param["name"]}/{"parseval.pt"}', map_location='cpu'))
    parseval_model.eval()

    print("Regular")
    check_parseval_tightness(model, "img/parseval_tightness/baseline.png")

    print("Parseval")
    check_parseval_tightness(parseval_model, "img/parseval_tightness/parseval.png")


if __name__ == '__main__':
    main()
    