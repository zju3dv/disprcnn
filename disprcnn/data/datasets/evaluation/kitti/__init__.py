from .kitti_eval import do_kitti_evaluation, do_kitti_pedestrian_evaluation, do_kitti_cyclist_evaluation


def kitti_evaluation(
        dataset,
        left_predictions,
        right_predictions,
        output_folder,
        class2type,
        box_only,
        iou_types,
        expected_results,
        expected_results_sigma_tol,
        eval_bbox3d,
):
    return do_kitti_evaluation(
        dataset=dataset,
        left_predictions=left_predictions,
        right_predictions=right_predictions,
        box_only=box_only,
        output_folder=output_folder,
        class2type=class2type,
        iou_types=iou_types,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
        eval_bbox3d=eval_bbox3d,
    )
def kitti_pedestrian_evaluation(
        dataset,
        left_predictions,
        right_predictions,
        output_folder,
        class2type,
        box_only,
        iou_types,
        expected_results,
        expected_results_sigma_tol,
        eval_bbox3d,
):
    return do_kitti_pedestrian_evaluation(
        dataset=dataset,
        left_predictions=left_predictions,
        right_predictions=right_predictions,
        box_only=box_only,
        output_folder=output_folder,
        class2type=class2type,
        iou_types=iou_types,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
        eval_bbox3d=eval_bbox3d,
    )


def kitti_cyclist_evaluation(
        dataset,
        left_predictions,
        right_predictions,
        output_folder,
        class2type,
        box_only,
        iou_types,
        expected_results,
        expected_results_sigma_tol,
        eval_bbox3d,
):
    return do_kitti_cyclist_evaluation(
        dataset=dataset,
        left_predictions=left_predictions,
        right_predictions=right_predictions,
        box_only=box_only,
        output_folder=output_folder,
        class2type=class2type,
        iou_types=iou_types,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
        eval_bbox3d=eval_bbox3d,
    )