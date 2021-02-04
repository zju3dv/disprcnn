# from disprcnn.utils.env import setup_environment  # noqa F401 isort:skip
import argparse
import os

from disprcnn.config import cfg
from disprcnn.data import make_data_loader
from disprcnn.solver import make_lr_scheduler
from disprcnn.solver import make_optimizer
from disprcnn.engine.inference import inference
from disprcnn.engine.trainer import do_train
from disprcnn.modeling.detector import build_detection_model
from disprcnn.utils.checkpoint import DetectronCheckpointer
from disprcnn.utils.comm import synchronize, get_rank
from disprcnn.utils.fix_model import fix_parameters
from disprcnn.utils.logger import setup_logger
from disprcnn.utils.miscellaneous import mkdir
import torch.multiprocessing
from torch.nn.parallel import DistributedDataParallel

# torch.multiprocessing.set_sharing_strategy('file_system')


def train(cfg, local_rank, distributed):
    torch.autograd.set_detect_anomaly(True)
    model = build_detection_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)
    optimizer, uncert = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer)
    model = fix_parameters(model, cfg)
    # Initialize mixed-precision training
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
            find_unused_parameters=True
        )

    arguments = {}
    arguments["iteration"] = 0

    output_dir = cfg.OUTPUT_DIR

    save_to_disk = get_rank() == 0
    checkpointer = DetectronCheckpointer(
        cfg, model, optimizer, scheduler, output_dir, save_to_disk
    )
    extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT,
                                              load_optimizer=cfg.SOLVER.LOAD_OPTIMIZER,
                                              load_scheduler=cfg.SOLVER.LOAD_SCHEDULER)
    arguments.update(extra_checkpoint_data)
    data_loader = make_data_loader(
        cfg,
        is_train=True,
        is_distributed=distributed,
        start_iter=arguments["iteration"],
    )

    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD

    do_train(
        model,
        data_loader,
        optimizer,
        scheduler,
        checkpointer,
        torch.device(cfg.MODEL.DEVICE),
        checkpoint_period,
        arguments,
        uncert,
        cfg
    )

    return model


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--config-file",
        default="configs/kitti/e2e_disp_rcnn_R_101_FPN_mf_2d.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("disprcnn", output_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    model = train(cfg, args.local_rank, args.distributed)


if __name__ == "__main__":
    main()
