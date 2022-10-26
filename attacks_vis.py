import matplotlib.pyplot as plt
import numpy as np


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


def plot_robust_curves(budgets, base_data, robust_data, xlabel, ylabel):
    fig, ax = plt.subplots()
    ax.plot(budgets, base_data, label='Baseline', marker='.', color='blue')
    ax.plot(budgets, robust_data, label='Regularized', marker='.', color='green')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    return fig


def main():
    file0 = 'models/isometry/iso_8/budgets.txt'
    file1 = 'models/isometry/vanilla_2/robust_acc.txt'
    file2 = 'models/isometry/iso_8/robust_acc.txt'
    budgets = np.loadtxt(file0)
    base_data = np.loadtxt(file1)
    base_data /= 100
    # base_data = base_data/base_data[0]*10000
    robust_data = np.loadtxt(file2)
    robust_data /= 100
    # robust_data = robust_data/robust_data[0]*10000
    _ = plot_robust_curves(budgets, base_data, robust_data, 'Attack perturbation', 'Accuracy (%)')
    file0 = 'models/isometry/iso_8/noises.txt'
    file1 = 'models/isometry/vanilla_2/noise_acc.txt'
    file2 = 'models/isometry/iso_8/noise_acc.txt'
    budgets = np.loadtxt(file0)
    base_data = np.loadtxt(file1)
    base_data /= 100
    # base_data = base_data/base_data[0]*10000
    robust_data = np.loadtxt(file2)
    robust_data /= 100
    # robust_data = robust_data/robust_data[0]*10000
    __ = plot_robust_curves(budgets, base_data, robust_data, 'Standard deviation', 'Accuracy (%)')
    plt.show()


if __name__ == '__main__':
    main()
