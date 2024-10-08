import torch
import math
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler


def get_lr_scheduler(config, optimizer, net):
    if config.lr_type == 'step_lr':
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.lr_decay_period, gamma=config.lr_decay_lvl)
        return lr_scheduler
    elif config.lr_type == 'cosine_repeat_lr':
        # lr_scheduler = CosineAnnealingLR_with_Restart(optimizer,T_max=config.lr_decay_period,T_mult=config.lr_decay_lvl)
        lr_scheduler = CosineAnnealingLR_with_Restart(optimizer,
                                                      T_max=config.epochs,
                                                      T_mult=1,
                                                      model=net,
                                                      take_snapshot=False,
                                                      eta_min=1e-3)
    else:
        raise Exception('Unknown lr_type')

    return lr_scheduler


class CosineAnnealingLR_with_Restart(_LRScheduler):
    """Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr and
    :math:`T_{cur}` is the number of epochs since the last restart in SGDR:

    .. math::

        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 +
        \cos(\frac{T_{cur}}{T_{max}}\pi))

    When last_epoch=-1, sets initial lr as lr.

    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_. The original pytorch
    implementation only implements the cosine annealing part of SGDR,
    I added my own implementation of the restarts part.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_max (int): Maximum number of iterations.
        T_mult (float): Increase T_max by a factor of T_mult
        eta_min (float): Minimum learning rate. Default: 0.
        last_epoch (int): The index of last epoch. Default: -1.
        model (pytorch model): The model to save.
        out_dir (str): Directory to save snapshots
        take_snapshot (bool): Whether to save snapshots at every restart

    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """

    def __init__(self, optimizer, T_max, T_mult, model=None, take_snapshot=None, out_dir=None, eta_min=0, last_epoch=-1):
        self.T_max = T_max
        self.T_mult = T_mult
        self.Te = self.T_max
        self.eta_min = eta_min
        self.current_epoch = last_epoch

        self.model = model
        self.out_dir = out_dir
        self.take_snapshot = take_snapshot

        self.lr_history = []

        super(CosineAnnealingLR_with_Restart, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        new_lrs = [self.eta_min + (base_lr - self.eta_min) *
                   (1 + math.cos(math.pi * self.current_epoch / self.Te)) / 2
                   for base_lr in self.base_lrs]

        self.lr_history.append(new_lrs)
        return new_lrs

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        self.current_epoch += 1

        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

        ## restart
        if self.current_epoch == self.Te:
            print("restart at epoch {:03d}".format(self.last_epoch + 1))

            if self.take_snapshot:
                torch.save({
                    'epoch': self.T_max,
                    'state_dict': self.model.state_dict()
                }, self.out_dir + "Weight/" + 'snapshot_e_{:03d}.pth.tar'.format(self.T_max))

            ## reset epochs since the last reset
            self.current_epoch = 0

            ## reset the next goal
            self.Te = int(self.Te * self.T_mult)
            self.T_max = self.T_max + self.Te

class LinearAlpha(object):
    def __init__(self, start_epoch, end_epoch, alpha_min, alpha_max):
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max

    def get_alpha(self, epoch):
        if epoch <= self.start_epoch:
            return self.alpha_min
        elif epoch >= self.end_epoch:
            return self.alpha_max
        else:
            epoch_step = self.end_epoch - self.start_epoch
            alpha_step = self.alpha_max - self.alpha_min
            return self.alpha_min + (epoch - self.start_epoch) * alpha_step / epoch_step