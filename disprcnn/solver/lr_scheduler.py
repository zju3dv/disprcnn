# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from bisect import bisect_right

from torch.optim.lr_scheduler import _LRScheduler
import numpy as np
from torch.optim.optimizer import Optimizer

class ConstantScheduler(_LRScheduler):
    def __init__(self, optimizer, last_epoch=-1):
        super(ConstantScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return self.base_lrs


# separating MultiStepLR with WarmupLR
# but the current LRScheduler design doesn't allow it
class WarmupMultiStepLR(_LRScheduler):
    def __init__(
            self,
            optimizer,
            milestones,
            gamma=0.1,
            warmup_factor=1.0 / 3,
            warmup_iters=500,
            warmup_method="linear",
            last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_epoch) / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]


class OneCycleScheduler(_LRScheduler):
    """Sets the learning rate of each parameter group according to
    cyclical learning rate policy (CLR). The policy cycles the learning
    rate between two boundaries with a constant frequency, as detailed in
    the paper `Cyclical Learning Rates for Training Neural Networks`_.
    The distance between the two boundaries can be scaled on a per-iteration
    or per-cycle basis.

    Cyclical learning rate policy changes the learning rate after every batch.
    `step` should be called after a batch has been used for training.

    This class uses annealing cosine policies, as put forth in the paper:

    This implementation was adapted from fastai repo.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        max_lr (float or list): Upper learning rate boundaries in the cycle
            for each parameter group. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore
            max_lr may not actually be reached depending on
            scaling function.
        total_steps(int): total iterations in training phase.
        cycle_momentum (bool): If ``True``, momentum is cycled inversely
            to learning rate between 'base_momentum' and 'max_momentum'.
            Default: True
        base_momentum (float or list): Initial momentum which is the
            lower boundary in the cycle for each parameter group.
            Default: 0.8
        max_momentum (float or list): Upper momentum boundaries in the cycle
            for each parameter group. Functionally,
            it defines the cycle amplitude (max_momentum - base_momentum).
            The momentum at any cycle is the difference of max_momentum
            and some scaling of the amplitude; therefore
            base_momentum may not actually be reached depending on
            scaling function. Default: 0.9
        last_epoch (int): The index of the last batch. This parameter is used when
            resuming a training job. Since `step()` should be invoked after each
            batch instead of after each epoch, this number represents the total
            number of *batches* computed, not the total number of epochs computed.
            When last_epoch=-1, the schedule is started from the beginning.
            Default: -1

    Example:
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        >>> data_loader = torch.utils.data.DataLoader(...)
        >>> epochs = ... # specify epochs
        >>> scheduler = OneCycleScheduler(optimizer, max_lr=0.1,total_steps=epochs*len(data_loader))
        >>> for epoch in range(epochs):
        >>>     for batch in data_loader:
        >>>         train_batch(...)
        >>>         scheduler.step()


    .. _Cyclical Learning Rates for Training Neural Networks: https://arxiv.org/abs/1506.01186
    .. _bckenstler/CLR: https://github.com/bckenstler/CLR
    """

    def __init__(self,
                 optimizer,
                 max_lr,
                 total_steps,
                 pct_start: float = 0.3,
                 div_factor=25.0,
                 final_div_factor=None,
                 cycle_momentum=True,
                 base_momentum=0.85,
                 max_momentum=0.95,
                 last_epoch=-1):

        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        start_lr = max_lr / div_factor
        if final_div_factor is None:
            final_div_factor = div_factor * 1e4
        end_lr = max_lr / final_div_factor
        base_lrs = self._format_param('base_lr', optimizer, start_lr)
        if last_epoch == -1:
            for lr, group in zip(base_lrs, optimizer.param_groups):
                group['lr'] = lr

        self.max_lrs = self._format_param('max_lr', optimizer, max_lr)
        self.end_lrs = self._format_param('end_lr', optimizer, end_lr)

        self.step_size_up = float(total_steps * pct_start)
        self.step_size_down = float(total_steps - self.step_size_up)
        self.total_steps = total_steps
        self.step_ratio = pct_start

        self.cycle_momentum = cycle_momentum
        if cycle_momentum:
            if 'momentum' in optimizer.defaults:
                # use momentum
                self.mom = 'momentum'
            elif 'betas' in optimizer.defaults:
                # use betas[0]
                self.mom = 'beta'
            else:
                raise ValueError('optimizer must support momentum with `cycle_momentum` option enabled')
            base_momentums = self._format_param('base_momentum', optimizer, base_momentum)
            if last_epoch == -1:
                for momentum, group in zip(base_momentums, optimizer.param_groups):
                    if self.mom == 'momentum':
                        group['momentum'] = momentum
                    elif self.mom == 'beta':
                        group['betas'] = (momentum, group['betas'][1])
                    else:
                        raise NotImplementedError()
            if self.mom == 'momentum':
                self.base_momentums = list(map(lambda group: group['momentum'], optimizer.param_groups))
            elif self.mom == 'beta':
                self.base_momentums = list(map(lambda group: group['betas'][0], optimizer.param_groups))
            self.max_momentums = self._format_param('max_momentum', optimizer, max_momentum)

        super(OneCycleScheduler, self).__init__(optimizer, last_epoch)

    def _format_param(self, name, optimizer, param):
        """Return correctly formatted lr/momentum for each param group."""
        if isinstance(param, (list, tuple)):
            if len(param) != len(optimizer.param_groups):
                raise ValueError("expected {} values for {}, got {}".format(
                    len(optimizer.param_groups), name, len(param)))
            return param
        else:
            return [param] * len(optimizer.param_groups)

    def get_lr(self):
        """Calculates the learning rate at batch index. This function treats
        `self.last_epoch` as the last batch index.

        If `self.cycle_momentum` is ``True``, this function has a side effect of
        updating the optimizer's momentum.
        """
        it = self.last_epoch + 1
        x = it / self.total_steps
        lrs = []
        for base_lr, max_lr, end_lr in zip(self.base_lrs, self.max_lrs, self.end_lrs):
            if x <= self.step_ratio:
                lr = annealing_cos(base_lr, max_lr, it / self.step_size_up)
            else:
                lr = annealing_cos(max_lr, end_lr, (it - self.step_size_up) / self.step_size_down)
            lrs.append(lr)

        if self.cycle_momentum:
            momentums = []
            for base_momentum, max_momentum in zip(self.base_momentums, self.max_momentums):
                if x <= self.step_ratio:
                    momentum = annealing_cos(max_momentum, base_momentum, it / self.step_size_up)
                else:
                    momentum = annealing_cos(base_momentum, max_momentum,
                                             (it - self.step_size_up) / self.step_size_down)
                momentums.append(momentum)
            for param_group, momentum in zip(self.optimizer.param_groups, momentums):
                if self.mom == 'momentum':
                    param_group['momentum'] = momentum
                elif self.mom == 'beta':
                    param_group['betas'] = (momentum, param_group['betas'][1])

        return lrs

    def read_momentum(self):
        if self.mom == 'momentum':
            return self.optimizer.param_groups[0]['momentum']
        elif self.mom == 'beta':
            return self.optimizer.param_groups[0]['betas'][0]
        else:
            raise KeyError()


def annealing_cos(start, end, pct: float):
    """Cosine anneal from `start` to `end` as pct goes from 0.0 to 1.0."""
    cos_out = np.cos(np.pi * pct) + 1
    return end + (start - end) / 2 * cos_out
