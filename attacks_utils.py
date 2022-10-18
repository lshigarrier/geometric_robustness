from pickletools import optimize
import torch
import torch.nn.functional as F
import numpy as np
from scipy.linalg import orth, eigh
from art.attacks.evasion import DeepFool, CarliniLInfMethod
from art.estimators.classification import PyTorchClassifier

def fgsm_attack(image, epsilon, data_grad):
    """
    Basic one-step FGSM attack
    """
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()

    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad

    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)

    # Return the perturbed image
    return perturbed_image


# ----------------------------------------------- Horizontal path ------------------------------------------------------


def compute_entropy(logprob):
    prob = torch.exp(logprob)
    return -torch.dot(prob.squeeze(), logprob.squeeze()), prob


def jacobian(model, data, device):
    data.requires_grad = True
    output = model(data)
    entropy, prob = compute_entropy(output)
    pred = output.argmax(dim=1, keepdim=True)
    jac = torch.zeros((10, 784)).to(device)
    for i in range(10):
        loss = F.nll_loss(output, torch.LongTensor([i]))
        model.zero_grad()
        loss.backward(retain_graph=True)
        jac[i, :] = data.grad.data.flatten()
        data.grad.zero_()
    return jac, entropy, pred, prob


def projection(vec, jac):
    """
    Project a vector [vec] onto a subspace spanned by the rows of [jac]
    """
    basis = orth(jac.T).T  # Orthonormal basis of the span of the rows of jac using SVD
    proj = np.dot(vec.T, basis[0, :])*basis[0, :]
    for i in range(1, basis.shape[0]):
        proj += np.dot(vec.T, basis[i, :])*basis[i, :]
    return proj


def compute_angle(org, dest, model, device):
    data = torch.from_numpy(org).reshape((1, 1, 28, 28)).to(device)
    jac, _, _, _ = jacobian(model, data, device)
    distance = np.linalg.norm(dest - org)
    cos_angle = np.linalg.norm(projection((dest - org) / distance, jac))
    return np.arccos(cos_angle)*180/np.pi


def robust_stop(distance, iteration):
    if len(distance) < 2:
        return float('inf')
    return abs((distance[-1] - distance[-2])*iteration/distance[-1])


# ------------------------------------------------- Distances ----------------------------------------------------------


def riemann_distance(x1, x2, model):
    p1 = torch.exp(model(x1.unsqueeze(0)))
    p2 = torch.exp(model(x2.unsqueeze(0)))
    return 2*torch.arccos(torch.sum(torch.sqrt(p1*p2)))


def l2_distance(loader):
    avg = np.zeros((10, 10))
    eff = np.zeros((10, 10))
    tot_avg = 0
    tot_eff = 0
    data, target = next(iter(loader))
    data = data.reshape((data.shape[0], data.shape[2]*data.shape[3])).numpy()
    target = target.numpy()
    for i in range(data.shape[0]):
        for j in range(i+1, data.shape[0]):
            '''
            temp1 = int(target[i])
            temp2 = int(target[j])
            idx1 = temp1 if temp1 <= temp2 else temp2
            idx2 = temp1 if temp1 > temp2 else temp2
            '''
            idx1 = int(target[i])
            idx2 = int(target[j])
            eff[idx1, idx2] += 1
            avg[idx1, idx2] += (np.linalg.norm(data[i]-data[j]) - avg[idx1, idx2])/eff[idx1, idx2]
            avg[idx2, idx1] = avg[idx1, idx2]
            tot_eff += 1
            tot_avg += (np.linalg.norm(data[i]-data[j]) - tot_avg)/tot_eff
    return avg, tot_avg, tot_eff


def fim_distance(loader, model):
    avg = np.zeros((10, 10))
    eff = np.zeros((10, 10))
    tot_avg = 0
    tot_eff = 0
    data, target = next(iter(loader))
    target = target.numpy()
    for i in range(data.shape[0]):
        for j in range(i+1, data.shape[0]):
            idx1 = int(target[i])
            idx2 = int(target[j])
            eff[idx1, idx2] += 1
            avg[idx1, idx2] += (riemann_distance(data[i], data[j], model) - avg[idx1, idx2])/eff[idx1, idx2]
            avg[idx2, idx1] = avg[idx1, idx2]
            tot_eff += 1
            tot_avg += (riemann_distance(data[i], data[j], model) - tot_avg)/tot_eff
    return avg, tot_avg, tot_eff


# ------------------------------------------------- Robust MNIST -------------------------------------------------------


def robust_loss(x, xr, model):
    return torch.norm(model(xr) - model(x))


def pgd(param, x, y, model):
    x.requires_grad = True
    with torch.enable_grad():
        for _iter in range(param['robust_max_iter']):
            loss = robust_loss(x, y, model)
            grads = torch.autograd.grad(loss, x, only_inputs=True)[0]
            x.data -= param['robust_step'] * F.normalize(grads.data)
            x = torch.clamp(x, 0, 1)
    return x


# ------------------------------------------------- FIM pullback -------------------------------------------------------


def get_fim(prob):
    """
    g_ij = delta_ij/p_i + 1/p_m
    """
    m = len(prob)
    eps = 1e-20
    return np.eye(m-1)/np.maximum(prob[:m-1], eps) + 1/np.maximum(prob[m-1], eps)


def jacobian_prob(model, data, device):
    data.requires_grad = True
    output = model(data)
    entropy, prob = compute_entropy(output)
    pred = output.argmax(dim=1, keepdim=True)
    jac = torch.zeros((9, 784)).to(device)
    for i in range(9):
        loss = prob.flatten()[i]
        model.zero_grad()
        loss.backward(retain_graph=True)
        jac[i, :] = data.grad.data.flatten()
        data.grad.zero_()
    return jac.detach().cpu().numpy(), entropy, pred, prob.detach().cpu().numpy().flatten()


def fim_pullback(model, data, device):
    jac, entropy, pred, prob = jacobian_prob(model, data, device)
    fim = get_fim(prob)
    return np.dot(jac.T, np.dot(fim, jac)), entropy, pred, prob


def eig(gx, m):
    try:
        values, vectors = eigh(gx)
    except:
        print(f'Error: {gx}')
    order = np.abs(values).argsort()
    vectors = vectors[:, order]
    values = values[order]
    idx = len(values) - m
    return values[idx:], vectors[:, idx:]


# --------------------------------------------- Adversarial Training ---------------------------------------------------


def project(x, original_x, epsilon, _type='linf'):
    """
    Projection of [x] onto the ball of radius [epsilon] centered on [original_x] with [_type] norm
    """
    if _type == 'linf':
        max_x = original_x + epsilon
        min_x = original_x - epsilon
        x = torch.max(torch.min(x, max_x), min_x)
    elif _type == 'l2':
        dist = (x - original_x)
        dist = dist.view(x.shape[0], -1)
        dist_norm = torch.norm(dist, dim=1, keepdim=True)
        mask = (dist_norm > epsilon).unsqueeze(2).unsqueeze(3)
        # dist = F.normalize(dist, p=2, dim=1)
        dist = dist / dist_norm
        dist *= epsilon
        dist = dist.view(x.shape)
        x = (original_x + dist) * mask.float() + x * (1 - mask.float())
    else:
        raise NotImplementedError
    return x


class FastGradientSignUntargeted:
    """
        Fast gradient sign untargeted adversarial attack, minimizes the initial class activation
        with iterative grad sign updates
    """

    def __init__(self, 
                        model,          # Pytorch model
                        device,         # CPU/GPU
                        epsilon,        # Maximum perturbation
                        alpha,          # Movement multiplier per iteration
                        min_val,        # Minimum value of the pixels
                        max_val,        # Maximum value of the pixels
                        max_iters,      # Maximum numbers of iteration to generated adversaries
                        _type='linf',   # The metric of perturbation size for epsilon
                        _loss='nll'     # Loss function
                        ):

        # Store variables internally
        self.model      = model
        self.device     = device
        self.epsilon    = epsilon
        self.alpha      = alpha
        self.min_val    = min_val
        self.max_val    = max_val
        self.max_iters  = max_iters
        self._type      = _type
        if _loss == 'nll':
            self._loss = F.nll_loss
        elif _loss == 'cross_entropy':
            self._loss = F.cross_entropy
        else:
            raise NotImplementedError

    def perturb(self, original_images, labels, random_start=False, loss_func='nll'):
        # original_images: values are within self.min_val and self.max_val

        # The adversaries created from random close points to the original data
        if random_start:
            rand_perturb = torch.FloatTensor(original_images.shape).uniform_(-self.epsilon, self.epsilon).to(self.device)
            x = original_images + rand_perturb
            x.clamp_(self.min_val, self.max_val)
        else:
            x = original_images.clone()

        # Turn gradients on
        x.requires_grad = True
        with torch.enable_grad():

            # Iterate
            for _iter in range(self.max_iters):

                # Forward pass
                outputs = self.model(x)

                # Calculate loss
                loss = self._loss(outputs, labels)

                # Calculate gradients wrt input
                grads = torch.autograd.grad(loss, x, only_inputs=True)[0]

                # Add perturbation
                x.data += self.alpha * torch.sign(grads.data)

                # Project -- the adversaries' pixel value should within max_x and min_x due to the l_infinity / l2 restriction
                x = project(x, original_images, self.epsilon, self._type)

                # Clamp -- the adversaries' value should be valid pixel value
                x.clamp_(self.min_val, self.max_val)

        return x


class DeepFool:
    """
        Deep Fool uses https://arxiv.org/abs/1511.04599
        and is implemented using adversarial-robustness-toolbox (ART)
    """

    def __init__(self, 
                        model,          # Pytorch model
                        input_shape,    # Size of input image
                        num_classes,    # Size of output
                        device,         # CPU/GPU
                        min_val,        # Minimum value of the pixels
                        max_val,        # Maximum value of the pixels
                        _optimizer,     # Update method
                        max_iters           = 100,      # Maximum numbers of iterations
                        overshoot_param     = 1e-6,     # Overshoot parameter 
                        _type               = 'linf',   # The metric of perturbation size for epsilon
                        _loss               = 'nll'     # Loss function
                        ):

        # Store variables internally
        self.model              = model
        self.input_shape        = input_shape
        self.num_classes        = num_classes
        self.device             = device
        self.overshoot_param    = overshoot_param
        self.min_val            = min_val
        self.max_val            = max_val
        self.max_iters          = max_iters
        self._type              = _type
        self._optimizer         = _optimizer
        if _loss == 'nll':
            self._loss = F.nll_loss
        elif _loss == 'cross_entropy':
            self._loss = F.cross_entropy
        else:
            raise NotImplementedError

        # Load adversarial-robustness-toolbox model wrapper
        self.art_classifier = PyTorchClassifier(
                                            model       = self.model,
                                            clip_values = (self.min_val, self.max_val),
                                            loss        = self._loss,
                                            optimizer   = self._optimizer,
                                            input_shape = self.input_shape,
                                            nb_classes  = self.num_classes,
                                        )

    def perturb(self, original_images, labels):
        # Convert images and labels to numpy arrays
        original_images  = original_images.numpy()
        labels           = labels.numpy()

        # Initialize attacker
        attack = DeepFool(
                            classifier  = self.art_classifier,
                            max_iter    = self.max_iters,
                            epsilon     = self.overshoot_param,
                            nb_grads    = self.num_classes,         # We can make this smaller so that it only computes the top class gradients for faster computation
                            batch_size  = original_images.shape[0],
                            verbose     = False
        )

        # Generate attacks
        x_adv = attack.generate(x = original_images, 
                                y = labels)

        # Retrun as PyTorch tensor
        return torch.from_numpy(x_adv)


class CWLinf:
    """
        The L_inf optimized attack of Carlini and Wagner from https://arxiv.org/abs/1608.04644
        and is implemented using adversarial-robustness-toolbox (ART)
    """

    def __init__(self, 
                        # Model/Data args
                        model,          # Pytorch model
                        input_shape,    # Size of input image
                        num_classes,    # Size of output
                        device,         # CPU/GPU
                        min_val,        # Minimum value of the pixels
                        max_val,        # Maximum value of the pixels
                        _optimizer,     # Update method
                        _loss = 'nll',   # Loss function

                        # CW-Linf Attack args
                        confidence      = 0.0,      # Confidence of adversarial examples: a higher value produces examples that are farther away, from the original input, but classified with higher confidence as the target class.
                        targeted        = False,    # Should the attack target one specific class
                        learning_rate   = 0.01,     # Learn rate
                        max_iter        = 10,       # Maximum number of iterations
                        decrease_factor = 0.9,      # The rate of shrinking tau, values in 0 < decrease_factor < 1 where larger is more accurate.
                        initial_const   = 1e-5,     # The initial value of constant c.
                        largest_const   = 20.0,     # The largest value of constant c.
                        const_factor    = 2.0,      # The rate of increasing constant c with const_factor > 1, where smaller more accurate.
                        
                        ):

        # Store variables internally
        self.model              = model
        self.input_shape        = input_shape
        self.num_classes        = num_classes
        self.device             = device
        self.min_val            = min_val
        self.max_val            = max_val
        self._optimizer         = _optimizer
        if _loss == 'nll':
            self._loss = F.nll_loss
        elif _loss == 'cross_entropy':
            self._loss = F.cross_entropy
        else:
            raise NotImplementedError

        self.confidence      = confidence
        self.targeted        = targeted
        self.learning_rate   = learning_rate
        self.max_iter        = max_iter
        self.decrease_factor = decrease_factor
        self.initial_const   = initial_const
        self.largest_const   = largest_const
        self.const_factor    = const_factor

        # Load adversarial-robustness-toolbox model wrapper
        self.art_classifier = PyTorchClassifier(
                                            model       = self.model,
                                            clip_values = (self.min_val, self.max_val),
                                            loss        = self._loss,
                                            optimizer   = self._optimizer,
                                            input_shape = self.input_shape,
                                            nb_classes  = self.num_classes,
                                        )

    def perturb(self, original_images, labels):
        # Convert images and labels to numpy arrays
        original_images  = original_images.numpy()
        labels           = labels.numpy()

        # Initialize attacker
        attack = CarliniLInfMethod(
                            classifier      = self.art_classifier,
                            confidence      = self.confidence,
                            targeted        = self.targeted,
                            learning_rate   = self.learning_rate,
                            max_iter        = self.max_iter,
                            decrease_factor = self.decrease_factor,
                            initial_const   = self.initial_const,
                            largest_const   = self.largest_const,
                            const_factor    = self.const_factor,
                            verbose              = False
        )

        # Generate attacks
        x_adv = attack.generate(x = original_images, 
                                y = labels)

        # Retrun as PyTorch tensor
        return torch.from_numpy(x_adv)

        
# ---------------------------------------------------- Main ------------------------------------------------------------


def main():
    vec = np.array([2, 3, 4, 5, 6])
    mat = np.eye(4)/vec[:4] + 1/vec[4]
    print(mat)


if __name__ == '__main__':
    main()
