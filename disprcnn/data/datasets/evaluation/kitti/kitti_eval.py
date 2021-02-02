import os

import numpy as np
from tqdm import tqdm
from disprcnn.utils.path import PROJECT_ROOT


def write_txt(dataset, predictions, output_folder, label='Car'):
    output_folder = os.path.join(output_folder, 'txt')
    os.makedirs(output_folder, exist_ok=True)
    for i, prediction in enumerate(tqdm(predictions)):
        imgid = dataset.ids[i]
        size = dataset.infos[int(imgid)]['size']
        # calib = dataset.get_calibration(i)
        prediction = prediction.resize(size)
        preds_per_img = []
        bbox = prediction.bbox.tolist()
        if prediction.has_field('box3d'):
            bbox3d = prediction.get_field('box3d').convert('xyzhwl_ry').bbox_3d.tolist()
            scores_3d = prediction.get_field('scores_3d').tolist()
            scores = prediction.get_field('scores').tolist()
            for b, b3d, s3d, s in zip(bbox, bbox3d, scores_3d, scores):
                sc = s3d
                x1, y1, x2, y2 = b
                x, y, z, h, w, l, ry = b3d
                alpha = ry + np.arctan(-x / z)
                preds_per_img.append(
                    f'{label} -1 -1 {alpha} {x1} {y1} {x2} {y2} {h} {w} {l} {x} {y} {z} {ry} {sc}'
                )
        else:
            scores = prediction.get_field('scores').tolist()
            for b, s in zip(bbox, scores):
                x1, y1, x2, y2 = b
                preds_per_img.append(
                    f'{label} -1 -1 -10 {x1} {y1} {x2} {y2} 0 0 0 0 0 0 0 {s}'
                )
        with open(os.path.join(output_folder, imgid + '.txt'), 'w') as f:
            f.writelines('\n'.join(preds_per_img))
    final_msg = ''
    if label == 'Car':
        iou_thresh = (0.7, 0.5)
    else:
        iou_thresh = (0.5,)
    for iou_thresh in iou_thresh:
        final_msg += '%.1f\n' % iou_thresh
        from termcolor import colored
        print(colored(f'-----using iou thresh{iou_thresh}------', 'red'))
        binary = os.path.join(PROJECT_ROOT, 'tools/kitti_object/kitti_evaluation_lib/evaluate_object_') + str(
            iou_thresh)
        gt_dir = os.path.join(PROJECT_ROOT, 'data/kitti/object/training/label_2')
        eval_command = f'{binary} {output_folder} {gt_dir}'
        os.system(eval_command)
        with open(os.path.join(output_folder, 'stats_%s_detection.txt' % label.lower())) as f:
            lines = np.array(
                [list(map(float, line.split())) for line in f.read().splitlines()]) * 100
            ap = lines[:, ::4].mean(1).tolist()
            final_msg += 'AP 2d %.2f %.2f %.2f\n' % (ap[0], ap[1], ap[2])
        orientation_path = os.path.join(output_folder, 'stats_%s_orientation.txt' % label.lower())
        if os.path.exists(orientation_path):
            with open(orientation_path) as f:
                lines = np.array(
                    [list(map(float, line.split())) for line in f.read().splitlines()]) * 100
            ap = lines[:, ::4].mean(1).tolist()
            final_msg += 'AP ori %.2f %.2f %.2f\n' % (ap[0], ap[1], ap[2])
        bev_path = os.path.join(output_folder, 'stats_%s_detection_ground.txt' % label.lower())
        if os.path.exists(bev_path):
            with open(bev_path) as f:
                lines = np.array(
                    [list(map(float, line.split())) for line in f.read().splitlines()]) * 100
            ap = lines[:, ::4].mean(1).tolist()
            final_msg += 'AP bev %.2f %.2f %.2f\n' % (ap[0], ap[1], ap[2])
        det_3d_path = os.path.join(output_folder, 'stats_%s_detection_3d.txt' % label.lower())
        if os.path.exists(det_3d_path):
            with open(det_3d_path) as f:
                lines = np.array(
                    [list(map(float, line.split())) for line in f.read().splitlines()]) * 100
            ap = lines[:, ::4].mean(1).tolist()
            final_msg += 'AP 3d %.2f %.2f %.2f\n' % (ap[0], ap[1], ap[2])
    print(colored(final_msg, 'red'))


def do_kitti_evaluation(
        dataset,
        left_predictions,
        right_predictions,
        class2type,
        box_only,
        output_folder,
        iou_types,
        expected_results,
        expected_results_sigma_tol,
        eval_bbox3d,
):
    write_txt(dataset, left_predictions, output_folder)


def do_kitti_pedestrian_evaluation(
        dataset,
        left_predictions,
        right_predictions,
        class2type,
        box_only,
        output_folder,
        iou_types,
        expected_results,
        expected_results_sigma_tol,
        eval_bbox3d,
):
    write_txt(dataset, left_predictions, output_folder, 'Pedestrian')


def do_kitti_cyclist_evaluation(
        dataset,
        left_predictions,
        right_predictions,
        class2type,
        box_only,
        output_folder,
        iou_types,
        expected_results,
        expected_results_sigma_tol,
        eval_bbox3d,
):
    write_txt(dataset, left_predictions, output_folder, 'Cyclist')
