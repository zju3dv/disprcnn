# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import os

import torch
from tqdm import tqdm

from disprcnn.data.datasets.evaluation import evaluate
from ..utils.comm import all_gather
from ..utils.comm import is_main_process, get_world_size
from ..utils.comm import synchronize
from ..utils.timer import Timer, get_time_str


def compute_on_dataset(model, data_loader, device, timer=None):
    model.eval()
    left_results_dict, right_results_dict = {}, {}
    cpu_device = torch.device("cpu")
    for i, batch in enumerate(tqdm(data_loader)):
        images, targets, image_ids = batch
        images = {k: v.to(device) for k, v in images.items()}
        targets['left'] = [target.to(device) for target in targets['left']]
        if len(image_ids) > 1 and isinstance(image_ids[1], dict):
            image_ids, preds_2d = image_ids
            preds_2d = {k: [t.to(device) for t in v] for k, v in preds_2d.items()}
        else:
            preds_2d = None
        with torch.no_grad():
            if timer:
                timer.tic()
            if preds_2d is None:
                output = model(images, targets)
            else:
                output = model(images, preds_2d, targets)
            if timer:
                torch.cuda.synchronize()
                timer.toc()
            if isinstance(output, dict):
                output = {k: [o.to(cpu_device) for o in v] for k, v in output.items()}
                left_results_dict.update({img_id: result for img_id, result in zip(image_ids, output['left'])})
                right_results_dict.update({img_id: result for img_id, result in zip(image_ids, output['right'])})
            else:
                output = [o.to(cpu_device) for o in output]
                left_results_dict.update(
                    {img_id: result for img_id, result in zip(image_ids, output)}
                )
    if len(right_results_dict) == 0:
        return left_results_dict
    else:
        return left_results_dict, right_results_dict


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = all_gather(predictions_per_gpu)
    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    if len(image_ids) != image_ids[-1] + 1:
        logger = logging.getLogger("disprcnn.inference")
        logger.warning(
            "Number of images that were gathered from multiple processes is not "
            "a contiguous set. Some images might be missing from the evaluation"
        )

    # convert to a list
    predictions = [predictions[i] for i in image_ids]
    return predictions


def inference(
        model,
        data_loader,
        dataset_name,
        iou_types=("bbox",),
        box_only=False,
        device="cuda",
        expected_results=(),
        expected_results_sigma_tol=4,
        output_folder=None,
        class2type=None,
        visualizer=None,
        eval_bbox3d=False,
        force_recompute=True,

):
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = get_world_size()
    logger = logging.getLogger("disprcnn.inference")
    dataset = data_loader.dataset
    logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))
    total_timer = Timer()
    inference_timer = Timer()
    total_timer.tic()
    write_preds_out = True
    if output_folder and os.path.exists(os.path.join(output_folder, 'predictions.pth')) and not force_recompute:
        predictions = torch.load(os.path.join(output_folder, 'predictions.pth'), 'cpu')
        write_preds_out = False
    else:
        predictions = compute_on_dataset(model, data_loader, device, inference_timer)
        # wait for all processes to complete before measuring the time
        synchronize()
        total_time = total_timer.toc()
        total_time_str = get_time_str(total_time)
        logger.info(
            "Total run time: {} ({} s / img per device, on {} devices)".format(
                total_time_str, total_time * num_devices / len(dataset), num_devices
            )
        )
        total_infer_time = get_time_str(inference_timer.total_time)
        logger.info(
            "Model inference time: {} ({} s / img per device, on {} devices)".format(
                total_infer_time,
                inference_timer.total_time * num_devices / len(dataset),
                num_devices,
            )
        )
        if len(predictions) == 1:
            predictions = _accumulate_predictions_from_multiple_gpus(predictions)
        else:
            left_predictions = _accumulate_predictions_from_multiple_gpus(predictions[0])
            right_predictions = _accumulate_predictions_from_multiple_gpus(predictions[1])
            predictions = {'left': left_predictions, 'right': right_predictions}
    if not is_main_process():
        return

    if output_folder and write_preds_out:
        torch.save(predictions, os.path.join(output_folder, "predictions.pth"))

    extra_args = dict(
        class2type=class2type,
        box_only=box_only,
        iou_types=iou_types,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
        eval_bbox3d=eval_bbox3d,
    )
    if predictions['left'][0].has_field('box3d'):
        extra_args['eval_bbox3d'] = True
    # Note: deepcopy dataset because evaluate change dataset
    # to such an extent that dataset cannot be accessed properly.
    result = evaluate(dataset=dataset,  # todo: deepcopy required here?
                      predictions=predictions,
                      output_folder=output_folder,
                      **extra_args)

    return result
