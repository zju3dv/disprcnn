# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import time

import torch
import torch.distributed as dist

from tensorboardX import SummaryWriter
from torch.nn.utils import clip_grad_norm_

from disprcnn.solver.lr_scheduler import OneCycleScheduler
from disprcnn.structures.image_list import ImageList
from disprcnn.utils.comm import get_world_size, get_rank
from disprcnn.utils.fix_model import fix_model_training
from disprcnn.utils.metric_logger import MetricLogger


def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def compute_losses(loss_dict, cfg, uncert):
    if cfg.SOLVER.UNCERT_LOSS_WEIGHT == 0:
        loss_reduced = sum(loss for loss in loss_dict.values())
    else:
        assert cfg.SOLVER.UNCERT_LOSS_WEIGHT == len(loss_dict.values()), \
            f'{cfg.SOLVER.UNCERT_LOSS_WEIGHT} != {len(loss_dict.values())}'
        loss_reduced = uncert.sum() + sum(loss * torch.exp(-un) for loss, un in zip(loss_dict.values(), uncert))
    return loss_reduced


def do_train(
        model,
        data_loader,
        optimizer,
        scheduler,
        checkpointer,
        device,
        checkpoint_period,
        arguments,
        uncert,
        cfg
):
    if get_rank() == 0:
        tb_writer = SummaryWriter(cfg.OUTPUT_DIR, flush_secs=20)
    logger = logging.getLogger("disprcnn.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(data_loader)
    start_iter = arguments["iteration"]
    model.train()
    fix_model_training(model, cfg)
    start_training_time = time.time()
    end = time.time()
    grad_norm_clip = cfg.SOLVER.GRAD_CLIP
    if isinstance(scheduler, OneCycleScheduler):
        scheduler = OneCycleScheduler(optimizer, cfg.SOLVER.BASE_LR, cfg.SOLVER.MAX_ITER, last_epoch=start_iter)
    valid_iter = start_iter
    for it, (images, targets, other_fields) in enumerate(data_loader, start_iter):
        if cfg.SOLVER.PRINT_ITERATION:
            print('iteration', it)
        if not check_forward(targets):
            logger.info('check forward failed, not forwarding this iteration.')
            it -= 1
            continue
        data_time = time.time() - end
        iteration = it + 1
        valid_iter += 1
        arguments["iteration"] = iteration
        try:
            images = {k: v.to(device) for k, v in images.items()}
            targets = {k: [t.to(device) for t in v] for k, v in targets.items()}
            if cfg.SOLVER.OFFLINE_2D_PREDICTIONS == '':
                # return idx only
                loss_dict = model(images, targets)
            else:
                _, preds2d = other_fields
                preds2d = {k: [t.to(device) for t in v] for k, v in preds2d.items()}
                loss_dict = model(images, preds2d, targets)
            # torch.cuda.synchronize()
            # print('forward cost', time.time() - begin)
            # losses = sum(loss for loss in loss_dict.values())
            losses = compute_losses(loss_dict, cfg, uncert)
            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = reduce_loss_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            meters.update(loss=losses_reduced, **loss_dict_reduced)
            optimizer.zero_grad()
            losses.backward()
            if cfg.SOLVER.DO_GRAD_CLIP:
                clip_grad_norm_(model.parameters(), grad_norm_clip)
            optimizer.step()
            scheduler.step()
        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            if cfg.SOLVER.ALLOW_EXCEPTION:
                print(e)
                valid_iter -= 1
            else:
                raise e

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % cfg.SOLVER.PRINT_INTERVAL == 0 or iteration == max_iter:
            if get_rank() == 0:
                if 'loss_dict_reduced' in locals():
                    for k, v in loss_dict_reduced.items():
                        tb_writer.add_scalar(k, v.data.cpu().numpy(), iteration)
                if 'losses_reduced' in locals():
                    tb_writer.add_scalar('losses_reduced', losses_reduced.item(), iteration)
                tb_writer.add_scalar('lr', optimizer.param_groups[0]["lr"], iteration)
                tb_writer.add_scalar('batch_time', batch_time, iteration)

                if cfg.SOLVER.UNCERT_LOSS_WEIGHT != 0:
                    for i, a in enumerate(uncert.data.cpu().numpy()):
                        tb_writer.add_scalar('uncert' + str(i), a, iteration)
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "valid_iter: {valid_iter}",
                        "{meters}",
                        "lr: {lr:.8f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    valid_iter=valid_iter,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
        if iteration % checkpoint_period == 0 and iteration != 0:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)
        if iteration == max_iter:
            checkpointer.save("model_final", **arguments)
    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )


def drop_empty_rois(images, targets):
    ret_images = {'left': ImageList(torch.empty(0), [(0, 0)]), 'right': ImageList(torch.empty(0), [(0, 0)])}
    ret_targets = {'left': [], 'right': []}
    num_samples = len(targets['left'])
    for i in range(num_samples):
        if len(targets['left'][i]) != 0 and len(targets['right'][i]) != 0:
            ret_images['left'] = ret_images['left'] + images['left'][i]
            ret_images['right'] = ret_images['right'] + images['right'][i]
            ret_targets['left'].append(targets['left'][i])
            ret_targets['right'].append(targets['right'][i])
    if ret_images['left'].tensors.ndimension() == 3:
        ret_images['left'].tensors = ret_images['left'].tensors.unsqueeze(0)
        ret_images['left'].image_sizes = [ret_images['left'].image_sizes]
    if ret_images['right'].tensors.ndimension() == 3:
        ret_images['right'].tensors = ret_images['right'].tensors.unsqueeze(0)
        ret_images['right'].image_sizes = [ret_images['right'].image_sizes]
    return ret_images, ret_targets


def check_forward(targets):
    return len(targets['left']) != 0 \
           and len(targets['right']) != 0 \
           and all(len(t) != 0 for t in targets['left']) \
           and all(len(t) != 0 for t in targets['right'])
