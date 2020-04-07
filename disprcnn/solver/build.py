# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

from torch.autograd import Variable

from .lr_scheduler import WarmupMultiStepLR, OneCycleScheduler
from torch.optim import SGD, Adam


def make_optimizer(cfg, model):
    # uncert
    if cfg.SOLVER.UNCERT_LOSS_WEIGHT != 0:
        uncert = Variable(torch.rand(cfg.SOLVER.UNCERT_LOSS_WEIGHT).cuda(),
                          requires_grad=True)
        torch.nn.init.constant_(uncert, -1.0)
    else:
        uncert = None
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    if cfg.SOLVER.UNCERT_LOSS_WEIGHT != 0:
        params += [{'params': [uncert], 'lr': lr}]
    if cfg.SOLVER.OPTIMIZER == 'SGD':
        optimizer = SGD(params, lr, momentum=cfg.SOLVER.MOMENTUM)
    elif cfg.SOLVER.OPTIMIZER == 'Adam':
        optimizer = Adam(params, lr)
    else:
        raise NotImplementedError()
    return optimizer, uncert


def make_lr_scheduler(cfg, optimizer):
    if cfg.SOLVER.SCHEDULER == 'WarmupMultiStepLR':
        return WarmupMultiStepLR(
            optimizer,
            cfg.SOLVER.STEPS,
            cfg.SOLVER.GAMMA,
            warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
            warmup_iters=cfg.SOLVER.WARMUP_ITERS,
            warmup_method=cfg.SOLVER.WARMUP_METHOD,
        )
    elif cfg.SOLVER.SCHEDULER == 'OneCycleScheduler':
        return OneCycleScheduler(
            optimizer,
            cfg.SOLVER.BASE_LR,
            cfg.SOLVER.MAX_ITER
        )
        # todo: think a better method for onycycle totalsteps coz maybe training not from iter 0.
    else:
        raise NotImplementedError()
