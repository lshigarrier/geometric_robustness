import torch
import torch.nn.functional as F
from torchattacks import GN, FGSM, PGD, PGDL2, DeepFool, CW

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
                        model,                # Pytorch model
                        epsilon,              # Maximum perturbation
                        alpha,                # Movement multiplier per iteration
                        min_val,              # Minimum value of the pixels
                        max_val,              # Maximum value of the pixels
                        max_iters,            # Maximum numbers of iteration to generated adversaries
                        random_start=False,   # Random initialization
                        _type='linf',         # The metric of perturbation size for epsilon
                        _loss='nll'           # Loss function
                        ):

        # Store variables internally
        self.model      = model
        self.epsilon    = epsilon
        self.alpha      = alpha
        self.min_val    = min_val
        self.max_val    = max_val
        self.max_iters  = max_iters
        self._type      = _type
        self.random_start = random_start
        if _loss == 'nll':
            self._loss = F.nll_loss
        elif _loss == 'cross_entropy':
            self._loss = F.cross_entropy
        else:
            raise NotImplementedError

    def set_attacker(self):
        pass

    def perturb(self, original_images, labels):
        # original_images: values are within self.min_val and self.max_val

        # The adversaries created from random close points to the original data
        if self.random_start:
            rand_perturb = torch.FloatTensor(original_images.shape).uniform_(-self.epsilon, self.epsilon)
            x = original_images + rand_perturb
            x.clamp_(self.min_val, self.max_val)
        else:
            x = original_images.clone()

        x.requires_grad = True

        # Turn gradients on
        with torch.enable_grad():

            # Iterate
            for _iter in range(self.max_iters):

                # Forward pass
                outputs = self.model(x)

                # Calculate loss
                loss = self._loss(outputs, labels)

                # Calculate gradients wrt input
                grads = torch.autograd.grad(loss, x)[0]

                # Add perturbation
                diff_tensor = self.alpha * torch.sign(grads.data)
                x.data += diff_tensor
                # For testing purposes
                # print(torch.max(torch.abs(diff_tensor), dim=1)[0].mean())
                # print(torch.linalg.norm(grads.data).mean())

                # Project -- the adversaries' pixel value should within max_x and min_x due to the l_infinity / l2 restriction
                x = project(x, original_images, self.epsilon, self._type)

                # Clamp -- the adversaries' value should be valid pixel value
                x.clamp_(self.min_val, self.max_val)

        return x


class TorchAttackGaussianNoise:
    """
        Arguments:
        model (nn.Module): model to attack.
        std (float): standard deviation (Default: 0.1)
    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`, `H = height` and `W = width`. MUST HAVE RANGE [0, 1]
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.
    """

    def __init__(self,
                        model,
                        std = 0.1,
                        return_type='float'
                        ):
        # Store variables internally
        self.model = model
        self.std   = std
        self.return_type = return_type
        self.attacker = None

        self.set_attacker()

    def set_attacker(self):
        self.attacker = GN(
                            model = self.model,
                            std = self.std
        )
        self.attacker.set_return_type(type=self.return_type)  # float returns [0-1], int returns [0-255]

    def perturb(self, original_images, labels):
        original_images = original_images.detach()
        labels = labels.detach()
        adv_images = self.attacker(original_images, labels)
        return adv_images


class TorchAttackFGSM:
    """
    FGSM in the paper 'Explaining and harnessing adversarial examples'
    [https://arxiv.org/abs/1412.6572]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.
    """

    def __init__(self, 
                        model,                    
                        eps         = 8/255,
                        return_type = 'float',
                        ):

        # Store variables internally
        self.model       = model
        self.eps         = eps
        self.return_type = return_type
        self.attacker = None

        self.set_attacker()
        
    def set_attacker(self):

        self.attacker = FGSM(
                            model = self.model,
                            eps   = self.eps
                            )
        self.attacker.set_return_type(type = self.return_type) # float returns [0-1], int returns [0-255]

    def perturb(self, original_images, labels):
        original_images = original_images.detach()
        original_images.requires_grad = True
        labels = labels.detach()
        with torch.enable_grad():
            adv_images = self.attacker(original_images, labels)
        return adv_images


class TorchAttackPGD:
    """
    ‘Towards Deep Learning Models Resistant to Adversarial Attacks’
    [https://arxiv.org/abs/1706.06083]

    Distance Measure : Linf
        Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation (Default: 0.2)
        alpha (float): step size (Default: 0.01)
        steps (int): number of steps. (Default: 40)
        random_start (bool): using random initialization (Default: False)
        return_type (str): 'float' for [0,1] or 'int' for [0-255] (Default: 'float')
    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`, `H = height` and `W = width`. MUST HAVE RANGE [0, 1]
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.
    """

    def __init__(self,
                        model,
                        eps   = 0.2,
                        alpha = 0.01,
                        steps = 40,
                        random_start = False,
                        return_type = 'float'
                        ):
        # Store variables internally
        self.model        = model
        self.eps          = eps
        self.alpha        = alpha
        self.steps        = steps
        self.random_start = random_start
        self.return_type = return_type
        self.attacker = None

        self.set_attacker()

    def set_attacker(self):
        self.attacker = PGD(
                            model = self.model,
                            eps   = self.eps,
                            alpha = self.alpha,
                            steps = self.steps,
                            random_start = self.random_start
        )
        self.attacker.set_return_type(type=self.return_type)  # float returns [0-1], int returns [0-255]

    def perturb(self, original_images, labels):
        original_images = original_images.detach()
        original_images.requires_grad = True
        labels = labels.detach()
        with torch.enable_grad():
            adv_images = self.attacker(original_images, labels)
        return adv_images


class TorchAttackPGDL2:
    """
    ‘Towards Deep Learning Models Resistant to Adversarial Attacks’
    [https://arxiv.org/abs/1706.06083]

    Distance Measure : Linf
        Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation (Default: 0.2)
        alpha (float): step size (Default: 0.01)
        steps (int): number of steps. (Default: 40)
        random_start (bool): using random initialization (Default: False)
        return_type (str): 'float' for [0,1] or 'int' for [0-255] (Default: 'float')
    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`, `H = height` and `W = width`. MUST HAVE RANGE [0, 1]
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.
    """

    def __init__(self,
                        model,
                        eps   = 0.2,
                        alpha = 0.01,
                        steps = 40,
                        random_start = False,
                        return_type = 'float'
                        ):
        # Store variables internally
        self.model        = model
        self.eps          = eps
        self.alpha        = alpha
        self.steps        = steps
        self.random_start = random_start
        self.return_type = return_type
        self.attacker = None

        self.set_attacker()

    def set_attacker(self):
        self.attacker = PGDL2(
                            model = self.model,
                            eps   = self.eps,
                            alpha = self.alpha,
                            steps = self.steps,
                            random_start = self.random_start
        )
        self.attacker.set_return_type(type=self.return_type)  # float returns [0-1], int returns [0-255]

    def perturb(self, original_images, labels):
        original_images = original_images.detach()
        original_images.requires_grad = True
        labels = labels.detach()
        with torch.enable_grad():
            adv_images = self.attacker(original_images, labels)
        return adv_images


class TorchAttackDeepFool:
    """
    'DeepFool: A Simple and Accurate Method to Fool Deep Neural Networks'
    [https://arxiv.org/abs/1511.04599]

    Distance Measure : L2
    Arguments:
        model (nn.Module): model to attack.
        overshoot (float): parameter for enhancing the noise. (Default: 0.02)
        return_type (str): 'float' for [0,1] or 'int' for [0-255] (Default: 'float')
    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`, `H = height` and `W = width`. MUST HAVE RANGE [0, 1]
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.
    """

    def __init__(self, 
                        model,                    
                        max_iters   =   50,     
                        overshoot   = 0.02,     
                        return_type = 'float', 
                        ):

        # Store variables internally
        self.model          = model
        self.overshoot      = overshoot
        self.max_iters      = max_iters
        self.return_type    = return_type
        self.attacker = None

        self.set_attacker()
        
    def set_attacker(self):
        self.attacker = DeepFool(
                                        model           = self.model, 
                                        steps           = self.max_iters, 
                                        overshoot       = self.overshoot
                                        )
        self.attacker.set_return_type(type=self.return_type)  # float returns [0-1], int returns [0-255]

    def perturb(self, original_images, labels):
        original_images = original_images.detach()
        original_images.requires_grad = True
        labels = labels.detach()
        with torch.enable_grad():
            adv_images = self.attacker(original_images, labels)
        return adv_images


class TorchAttackCWL2:
    """
    CW in the paper 'Towards Evaluating the Robustness of Neural Networks'
    [https://arxiv.org/abs/1608.04644]

    Distance Measure : L2
    Arguments:
        model (nn.Module): model to attack.
        c (float): c in the paper. parameter for box-constraint. (Default: 1)    
            :math:`minimize \Vert\frac{1}{2}(tanh(w)+1)-x\Vert^2_2+c\cdot f(\frac{1}{2}(tanh(w)+1))`
        kappa (float): kappa (also written as 'confidence') in the paper. (Default: 0)
            :math:`f(x')=max(max\{Z(x')_i:i\neq t\} -Z(x')_t, - \kappa)`
        max_iters (int): number of steps. (Default: 50)
        lr (float): learning rate of the Adam optimizer. (Default: 0.01)
    .. warning:: With default c, you can't easily get adversarial images. Set higher c like 1.
    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.
    Examples::
        >> attack = TorchAttackCWL2(model, c=1, kappa=0, steps=50, lr=0.01)
        >> adv_images = attack.perturb(images, labels)
    .. note:: Binary search for c is NOT IMPLEMENTED methods in the paper due to time consuming.
    """

    def __init__(self, 
                        # Model/Data args
                        model,         
                        c           = 10, 
                        kappa       = 0,
                        max_iters    = 50,
                        lr          = 0.01,
                        return_type = 'float'):

        # Store variables internally
        self.model          = model
        self.c              = c
        self.kappa          = kappa
        self.max_iters       = max_iters
        self.lr             = lr
        self.return_type    = return_type
        self.attacker = None

        # Initialize
        self.set_attacker()

    def set_attacker(self):

        self.attacker = CW(
                            model           = self.model, 
                            c               = self.c,
                            kappa           = self.kappa, 
                            steps           = self.max_iters,
                            lr              = self.lr
                            )
        self.attacker.set_return_type(type = self.return_type) # float returns [0-1], int returns [0-255]

    def perturb(self, original_images, labels):
        original_images = original_images.detach()
        original_images.requires_grad = True
        labels = labels.detach()
        with torch.enable_grad():
            adv_images = self.attacker(original_images, labels)
        return adv_images
