import argparse
import os
from warnings import warn

import torch
import torch.multiprocessing
from disprcnn.utils.average_meter import AverageMeter
from disprcnn.solver.lr_scheduler import OneCycleScheduler, ConstantScheduler
from disprcnn.utils.fastai_ext import split_no_wd_params, flatten_model
from torch.optim import Adam

from disprcnn.engine.psm_trainer import BaseTrainer

from torch import nn
from torch.utils.data import DataLoader
from tqdm import trange

from disprcnn.data.datasets import KITTIRoiDataset
from disprcnn.modeling.psmnet.stackhourglass import PSMNet
from disprcnn.structures.disparity import DisparityMap
from disprcnn.utils.comm import is_main_process
from disprcnn.utils.loss_utils import PSMLoss
from disprcnn.utils.stereo_utils import end_point_error

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default=os.path.expanduser('~/Datasets/kitti/object/pob_roi_freex'))
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--maxlr', type=float, default=1e-2)
parser.add_argument('--model_dir', type=str, default='models/kitti/object_psmnet_roi_freex_224')
parser.add_argument('--load_model', type=str, default=None)
parser.add_argument('--load', type=str, default=None)
parser.add_argument('--batch_size', type=int, default=12)
parser.add_argument('--resolution', type=int, default=224)
parser.add_argument('--maxdisp', type=int, default=48)
parser.add_argument('--mindisp', type=int, default=-48)
parser.add_argument('--workers', type=int, default=4)
parser.add_argument('--weight_decay', type=float, default=0.01)
args = parser.parse_args()
num_gpus = torch.cuda.device_count()
if num_gpus > 1:
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def evaluate(trainer: BaseTrainer, dataset):
    if dataset == 'valid':
        ds: KITTIRoiDataset = trainer.valid_dl.dataset
    else:
        ds: KITTIRoiDataset = trainer.train_dl.dataset
    # debug
    preds = trainer.get_preds(dataset)
    if not is_main_process(): return
    print('Computing epe.')
    am = AverageMeter()
    epes = []
    for i in trange(len(ds)):
        pred = preds[i]
        targets = ds.get_target(i)
        mask, target = targets['mask'], targets['disparity']
        epe = end_point_error(target, mask, pred)
        # epe = rmse(target, mask, pred)
        epes.append(epe)
        am.update(epe, mask.sum().item())
    print('Average epe', am.avg)
    print('Original size...')
    ds = KITTIRoiDataset(ds.root, ds.split, -1, ds.maxdisp, ds.mindisp, ds.length)
    am = AverageMeter()
    epes = []
    for i in trange(len(ds)):
        pred = preds[i]
        targets = ds.get_target(i)
        mask, target = targets['mask'], targets['depth']
        # compute depth
        pred = DisparityMap(pred).resize(mask.shape[::-1]).data + targets['x1'] - targets['x1p']
        pred = targets['fuxb'] / (pred + 1e-6)
        epe = end_point_error(target, mask, pred)
        # epe = rmse(target, mask, pred)
        epes.append(epe)
        am.update(epe, mask.sum().item())
    torch.save(epes, os.path.join(args.model_dir, 'epes.pth'))
    print('Average epe', am.avg)
    print()


def build_solver(model, total_steps):
    split_params = split_no_wd_params([nn.Sequential(*flatten_model(model))])
    optimizer = Adam([{'params': p, 'lr': args.maxlr,
                       'betas': (0.9, 0.99),
                       } for p in split_params])
    if args.mode == 'train_oc':
        scheduler = OneCycleScheduler(optimizer, args.maxlr, total_steps)
    else:
        scheduler = ConstantScheduler(optimizer)
    return optimizer, scheduler


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
        KITTIRoiDataset(args.data_dir, 'train', args.resolution, args.maxdisp, args.mindisp),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers)
    val_dl = DataLoader(
        KITTIRoiDataset(args.data_dir, 'val', args.resolution, args.maxdisp, args.mindisp, ),
        batch_size=args.batch_size,
        num_workers=args.workers)

    loss_fn = PSMLoss()
    optimizer, scheduler = build_solver(model, len(train_dl) * args.epochs)
    trainer = BaseTrainer(model, train_dl, val_dl, args.epochs, loss_fn,
                          optimizer, scheduler, args.model_dir,
                          args.maxlr, args.weight_decay)
    if num_gpus > 1:
        trainer.to_distributed()
    if args.load is not None:
        trainer.load(args.load)
    if args.mode in ['train', 'train_oc']:
        trainer.fit()
    elif args.mode in ['evaluate', 'eval']:
        evaluate(trainer, 'valid')
    elif args.mode == 'eval_train':
        evaluate(trainer, 'train')
    else:
        raise ValueError('args.mode not supported.')


if __name__ == "__main__":
    main()
