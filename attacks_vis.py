import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_side_by_side(img, adv_img, pred, adv_pred, title, save_path):
    
    petrubation = adv_img - img

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.suptitle(title, fontsize=14, y=0.95)

    for ax in [ax1, ax2, ax3]:
        ax.set_xticks([])
        ax.set_yticks([])
    
    ax1.imshow(img.permute(1, 2, 0).cpu().detach(), cmap="gray")
    ax1.set_title("Orginal")
    ax1.set_xlabel("Prediction: " + str(pred.item()))


    ax2.imshow(adv_img.permute(1, 2, 0).cpu().detach(), cmap="gray")
    ax2.set_title("Adversarial")
    ax2.set_xlabel("Prediction: " + str(adv_pred.item()))


    ax3.imshow(petrubation.permute(1, 2, 0).cpu().detach(), cmap="gray")
    ax3.set_title("Perturbation")

    linf = round(torch.norm(petrubation, p=float('inf')).item(),2)
    l2   = round(torch.norm(petrubation, p=2).item(),2)
    ax3.set_xlabel("Linf: " + str(linf) + "  L2: " + str(l2))

    fig.savefig(save_path)

def plot_accuracy(epsilons, accuracies):
    fig, ax = plt.subplots()
    ax.plot(epsilons, accuracies, "*-")
    ax.set_yticks(np.arange(0, 1.1, step=0.1))
    ax.set_xticks(np.arange(0, epsilons[-1]+0.05, step=0.05))
    ax.set_title("Accuracy vs Epsilon")
    ax.set_xlabel("Epsilon")
    ax.set_ylabel("Accuracy")
    return fig


def plot_examples(epsilons, examples):
    cnt = 0
    fig = plt.figure(figsize=(10, 10))
    for i in range(len(epsilons)):
        for j in range(len(examples[i])):
            cnt += 1
            ax = plt.subplot(len(epsilons), len(examples[0]), cnt)
            ax.set_xticks([], [])
            ax.set_yticks([], [])
            if j == 0:
                ax.set_ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
            orig_prob, orig, orig_ex, adv_prob, adv, ex = examples[i][j]
            ax.set_title(f'{orig}({orig_prob:.2f}) -> {adv}({adv_prob:.2f})')
            ax.imshow(ex, cmap="gray")
    fig.tight_layout()
    return fig


def plot_one_img(img, title):
    fig, ax = plt.subplots()
    ax.set_xticks([], [])
    ax.set_yticks([], [])
    ax.set_title(title)
    ax.imshow(img, cmap="gray")
    return fig


def plot_img(paths, ent_lists, pred_lists, prob_lists):
    n = len(pred_lists)
    fig, axs = plt.subplots(3, n)
    for i in range(n):
        for j in range(3):
            axs[j][i].set_xticks([], [])
            axs[j][i].set_yticks([], [])
            if j == 0:
                idx = 0
            elif j == 1:
                idx = np.argmax(ent_lists[i])
            else:
                idx = -1
            axs[j][i].set_title(f'{pred_lists[i][idx].item()}\n{np.max(prob_lists[i][idx])*100:.2f}%')
            axs[j][i].imshow(paths[i][idx], cmap="gray")
    return fig


def plot_one_graph(xdata, ydata, title, xlabel, ylabel, multiple=False):
    fig, ax = plt.subplots()
    if multiple:
        for i in range(ydata.shape[1]):
            ax.plot(xdata, ydata[:, i])
    else:
        ax.plot(xdata, ydata)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig


def plot_graph(xdata, ydatas, title, xlabel, ylabel, multiple=False):
    fig, axs = plt.subplots(3, 3)
    i1, i2 = 0, 0
    for ydata in ydatas:
        if multiple:
            for j in range(ydata.shape[1]):
                axs[i1][i2].plot(xdata, ydata[:, j])
        else:
            axs[i1][i2].plot(xdata, ydata)
        axs[i1][i2].set_xlabel(xlabel)
        axs[i1][i2].set_ylabel(ylabel)
        i2 = (i2 + 1) % 3
        if i2 == 0:
            i1 += 1
    fig.suptitle(title)
    return fig


def plot_img_file(path, nb_img, target_path=None):
    """
    Plot images from a .npy file
    :param path: path to the .npy file
    :param nb_img: the first nb_img will be plotted
    :param target_path: path to a .npy file containing the target of the images (optional)
    :return: a pyplot figure
    """
    images = np.load(path+'.npy')
    if target_path:
        target = np.load(target_path+'.npy')
    fig = plt.figure()
    for i in range(nb_img):
        ax = plt.subplot(1, nb_img, i+1)
        ax.set_xticks([], [])
        ax.set_yticks([], [])
        if target_path:
            ax.set_title(f'{target[i]}')
        ax.imshow(images[i], cmap="gray")
    fig.tight_layout()
    return fig


def plot_path(path, nb_img, target_path=None):
    """
    Plot images from a list
    :param path: list of images
    :param nb_img: number of images to plot
    :param target_path: targets of the images (optional)
    :return: a pyplot figure
    """
    index = [round(x*(len(path)-1)/(nb_img-1)) for x in range(nb_img)]
    fig = plt.figure()
    count = 1
    for i in index:
        ax = plt.subplot(1, nb_img, count)
        ax.set_xticks([], [])
        ax.set_yticks([], [])
        if target_path:
            ax.set_title(f'{i}\n{target_path[i]}')
        ax.imshow(path[i], cmap="gray")
        count += 1
    fig.tight_layout()
    return fig


def plot_hist(data, title, xlabel, ylabel, bins=None, xmax=None, xline=None):
    fig, ax = plt.subplots()
    if bins:
        ax.hist(data, bins=bins, color='c', edgecolor='k', alpha=0.65)
    else:
        ax.hist(data, bins='auto', color='c', edgecolor='k', alpha=0.65)
    if xline:
        plt.axvline(xline, color='r', linestyle='dashed', linewidth=1)
    if xmax:
        ax.set_xlim(0, xmax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig


def plot_curves(list1, list2, title, xlabel, ylabel):
    fig, ax = plt.subplots()
    ax.plot(list1, label='Training')
    ax.plot(list2, label='Testing')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    return fig


def plot_robust_curves(budgets, base_data, robust_data, adv_data, xlabel, ylabel):
    fig, ax = plt.subplots()
    ax.plot(budgets, base_data, label='Baseline', marker='.', color='blue', linewidth=4, markersize=12, linestyle='dashed')
    ax.plot(budgets, robust_data, label='Regularized', marker='.', color='green', linewidth=4, markersize=12, linestyle='solid')
    ax.plot(budgets, adv_data, label='Adversarial training', marker='.', color='red', linewidth=4, markersize=12, linestyle='dotted')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    return fig


def main():
    plt.rcParams.update({'font.size': 16})

    file0 = 'outputs/icassp/budgets.txt'
    file1 = 'outputs/icassp/vanilla_2_robust_acc.txt'
    file2 = 'outputs/icassp/iso_8_robust_acc.txt'
    file3 = 'outputs/icassp/adv_train_robust_acc.txt'

    budgets = np.loadtxt(file0)
    base_data = np.loadtxt(file1)
    base_data /= 100
    # base_data = base_data/base_data[0]*10000

    robust_data = np.loadtxt(file2)
    robust_data /= 100
    # robust_data = robust_data/robust_data[0]*10000

    adv_data = np.loadtxt(file3)
    adv_data /= 100
    # robust_data = robust_data/robust_data[0]*10000

    _ = plot_robust_curves(budgets, base_data, robust_data, adv_data, 'Attack perturbation', 'Accuracy (%)')
    plt.show()


if __name__ == '__main__':
    main()
