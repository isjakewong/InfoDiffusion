import random
import math
import numpy as np
import torch
from torch.optim.lr_scheduler import _LRScheduler
from torch.nn import functional as F
from torch.utils.data import Dataset
from sklearn.datasets import make_swiss_roll


def gaussian_mixture(batch_size, n_dim=2, n_labels=10,
                     x_var=0.5, y_var=0.1, label_indices=None):
    if n_dim % 2 != 0:
        raise Exception("n_dim must be a multiple of 2.")

    def sample(x, y, label, n_labels):
        shift = 1.4
        if label >= n_labels:
            label = np.random.randint(0, n_labels)
        r = 2.0 * np.pi / float(n_labels) * float(label)
        new_x = x * math.cos(r) - y * math.sin(r)
        new_y = x * math.sin(r) + y * math.cos(r)
        new_x += shift * math.cos(r)
        new_y += shift * math.sin(r)
        return np.array([new_x, new_y]).reshape((2,))

    x = np.random.normal(0, x_var, (batch_size, n_dim // 2))
    y = np.random.normal(0, y_var, (batch_size, n_dim // 2))
    z = np.empty((batch_size, n_dim), dtype=np.float32)
    for batch in range(batch_size):
        for zi in range(n_dim // 2):
            if label_indices is not None:
                z[batch, zi*2:zi*2+2] = sample(x[batch, zi], y[batch, zi], label_indices[batch], n_labels)
            else:
                z[batch, zi*2:zi*2+2] = sample(x[batch, zi], y[batch, zi], np.random.randint(0, n_labels), n_labels)

    return z

def swiss_roll(batch_size, noise=0.5):
    return make_swiss_roll(n_samples=batch_size, noise=noise)[0][:, [0, 2]] / 5.

def cos(a, b):
    a = a.view(-1)
    b = b.view(-1)
    a = F.normalize(a, dim=0)
    b = F.normalize(b, dim=0)
    return (a * b).sum()

def generate_exp_string(args) -> str:
    root = f'{args.dataset}_{args.a_dim}d'
    if args.kld_weight != 0:
        root += f'_{args.kld_weight}kld'
        if args.use_C:
            root += f'_{args.C_max}C'
    if args.mmd_weight != 0:
        root += f'_{args.mmd_weight}mmd'
    if args.prior != 'regular':
        root += f'_{args.prior}'
    if args.is_bottleneck:
        root += '_bottleneck'
    return root


def seed_everything(r_seed):
    print("Set seed: ", r_seed)
    random.seed(r_seed)
    np.random.seed(r_seed)
    torch.manual_seed(r_seed)
    torch.cuda.manual_seed(r_seed)
    torch.cuda.manual_seed_all(r_seed)
    torch.backends.cudnn.deterministic = True


@torch.jit.script
def compute_kernel(x, y):
    x_size = x.shape[0]
    y_size = y.shape[0]
    dim = x.shape[1]

    tiled_x = x.view(x_size, 1, dim).repeat(1, y_size, 1)
    tiled_y = y.view(1, y_size, dim).repeat(x_size, 1, 1)

    return torch.exp(-torch.mean((tiled_x - tiled_y)**2, dim=2)/dim*1.0)

@torch.jit.script
def compute_mmd(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    return torch.mean(x_kernel) + torch.mean(y_kernel) - 2*torch.mean(xy_kernel)


class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self. count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]  
        print('\r' + '\t'.join(entries), end='')

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class GradualWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, multiplier, warm_epoch, after_scheduler=None):
        self.multiplier = multiplier
        self.total_epoch = warm_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        self.last_epoch = None
        self.base_lrs = None
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

    def step(self, epoch=None, metrics=None):
        if self.finished and self.after_scheduler:
            if epoch is None:
                self.after_scheduler.step(None)
            else:
                self.after_scheduler.step(epoch - self.total_epoch)
        else:
            return super(GradualWarmupScheduler, self).step(epoch)


class LatentDataset(Dataset):
    def __init__(self, data_path):
        data = np.load(data_path)
        self.x = torch.from_numpy(data['all_a']).float()

    def __getitem__(self, index):
        return self.x[index]

    def __len__(self):
        return len(self.x)