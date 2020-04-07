import torch
import torch.distributed as dist
from torch import nn
from torch.nn import ModuleList
from collections import OrderedDict
from typing import List, Collection
from disprcnn.utils.comm import get_rank, get_world_size
from fastai.basic_train import LearnerCallback, Learner
from fastai.callbacks import SaveModelCallback
from fastai.torch_core import rank_distrib
from tensorboardX import SummaryWriter


class TensorBoardCallback(LearnerCallback):
    learn: Learner

    def __init__(self, learn):
        super().__init__(learn)
        if get_rank() == 0:
            self.tb_writer = SummaryWriter(learn.model_dir, flush_secs=10)

    def on_batch_end(self, **kwargs) -> None:
        if dist.is_initialized() and rank_distrib() != 0: return
        if kwargs['train']:
            # print('loss', kwargs['last_loss'].item())
            self.tb_writer.add_scalar('Train/loss', kwargs['last_loss'].item(), kwargs['iteration'])


class DistributedSaveModelCallback(SaveModelCallback):
    def on_epoch_end(self, epoch: int, **kwargs) -> None:
        "Compare the value monitored to its best score and maybe save the model."
        if self.every == "epoch":
            self.learn.save(f'{self.name}_{epoch}')
        else:  # every="improvement"
            c = self.get_monitor_value()
            world_size = get_world_size()
            if world_size == 1:
                current = c
                if current is not None and self.operator(current, self.best):
                    print(f'Better model found at epoch {epoch} with {self.monitor} value: {current}.')
                    self.best = current
                    self.learn.save(f'{self.name}')
            else:
                with torch.no_grad():
                    c = torch.tensor(c).cuda()
                    dist.reduce(c, dst=0)
                    if get_rank() == 0:
                        current = c / world_size
                        current = current.data
                        if current is not None and current < self.best:
                            print(f'Better model found at epoch {epoch} with {self.monitor} value: {current}.')
                            self.best = current
                            self.learn.save(f'{self.name}')


class ParameterModule(nn.Module):
    "Register a lone parameter `p` in a module."

    def __init__(self, p: nn.Parameter):
        super().__init__()
        self.val = p

    def forward(self, x):
        return x


def children_and_parameters(m: nn.Module):
    "Return the children of `m` and its direct parameters not registered in modules."
    children = list(m.children())
    children_p = sum([[id(p) for p in c.parameters()] for c in m.children()], [])
    for p in m.parameters():
        if id(p) not in children_p:
            children.append(ParameterModule(p))
    return children


def children(m: nn.Module) -> ModuleList:
    "Get children of `m`."
    return list(m.children())


def num_children(m: nn.Module) -> int:
    "Get number of children modules in `m`."
    return len(children(m))


flatten_model = lambda m: sum(map(flatten_model, children_and_parameters(m)), []) if num_children(m) else [m]

bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
no_wd_types = bn_types + (nn.LayerNorm,)
bias_types = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)


def trainable_params(m: nn.Module) -> Collection[nn.Parameter]:
    "Return list of trainable params in `m`."
    res = filter(lambda p: p.requires_grad, m.parameters())
    return res


def split_no_wd_params(layer_groups: Collection[nn.Module]) -> List[List[nn.Parameter]]:
    "Separate the parameters in `layer_groups` between `no_wd_types` and  bias (`bias_types`) from the rest."
    split_params = []
    for l in layer_groups:
        l1, l2 = [], []
        for c in l.children():
            if isinstance(c, no_wd_types):
                l2 += list(trainable_params(c))
            elif isinstance(c, bias_types):
                bias = c.bias if hasattr(c, 'bias') else None
                l1 += [p for p in trainable_params(c) if not (p is bias)]
                if bias is not None: l2.append(bias)
            else:
                l1 += list(trainable_params(c))
        # Since we scan the children separately, we might get duplicates (tied weights). We need to preserve the order
        # for the optimizer load of state_dict
        l1, l2 = list(OrderedDict.fromkeys(l1).keys()), list(OrderedDict.fromkeys(l2).keys())
        split_params += [l1, l2]
    return split_params
