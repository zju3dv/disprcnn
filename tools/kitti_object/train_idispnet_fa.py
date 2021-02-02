import argparse
import os
from warnings import warn

import torch
import torch.multiprocessing
import torch.nn.functional as F
from fastai.train import fit_one_cycle

from disprcnn.utils.fastai_ext import TensorBoardCallback, DistributedSaveModelCallback
from fastai.basic_data import DataBunch
from fastai.basic_train import Learner
from torch import nn
from torch.distributed import get_rank
from torch.utils.data import DataLoader
from fastai.distributed import *  # do not delete this line!
from disprcnn.data.datasets import KITTIRoiDataset
from disprcnn.modeling.psmnet.stackhourglass import PSMNet
from disprcnn.utils.loss_utils import PSMLoss

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='data/pob_roi')
parser.add_argument('--mode', type=str, default='train_oc')
parser.add_argument('--maxlr', type=float, default=1e-2)
parser.add_argument('--model_dir', type=str, default='models/kitti/object_psmnet_roi_freex_224')
parser.add_argument('--load_model', type=str, default='models/PSMNet/pretrained_model_KITTI2015.tar')
parser.add_argument('--maxdisp', type=int, default=48)
parser.add_argument('--mindisp', type=int, default=-48)
parser.add_argument('--workers', type=int, default=8)
parser.add_argument('--batch_size', type=int, default=12)
parser.add_argument('--load', type=str, default=None)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--local_rank', type=int, default=0)
args = parser.parse_args()
os.makedirs(args.model_dir, exist_ok=True)
num_gpus = torch.cuda.device_count()
if num_gpus > 1:
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
args.ngpus = num_gpus
with open(os.path.join(args.model_dir, 'args.yml'), 'w') as f:
    f.write(str(args))
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def main():
    model = PSMNet(args.maxdisp, args.mindisp).cuda()
    if args.load_model is not None:
        if args.load is not None:
            warn('args.load is not None. load_model will be covered by load.')
        ckpt = torch.load(args.load_model, 'cpu')
        if 'model' in ckpt.keys():
            pretrained = ckpt['model']
        elif 'state_dict' in ckpt.keys():
            pretrained = ckpt['state_dict']
        else:
            raise RuntimeError()
        pretrained = {k.replace('module.', ''): v for k, v in pretrained.items()}
        model.load_state_dict(pretrained)
    train_dl = DataLoader(
        KITTIRoiDataset(args.data_dir, 'train', args.maxdisp, args.mindisp),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers)
    val_dl = DataLoader(
        KITTIRoiDataset(args.data_dir, 'val', args.maxdisp, args.mindisp),
        batch_size=args.batch_size,
        num_workers=args.workers)

    loss_fn = PSMLoss()

    databunch = DataBunch(train_dl, val_dl, device='cuda')
    learner = Learner(databunch, model, loss_func=loss_fn, model_dir=args.model_dir)
    learner.callbacks = [DistributedSaveModelCallback(learner), TensorBoardCallback(learner)]
    if num_gpus > 1:
        learner.to_distributed(get_rank())
    if args.load is not None:
        learner.load(args.load)
    if args.mode == 'train':
        learner.fit(args.epochs, args.maxlr)
    elif args.mode == 'train_oc':
        fit_one_cycle(learner, args.epochs, args.maxlr)
    else:
        raise ValueError('args.mode not supported.')


if __name__ == "__main__":
    main()
