echo 'generating rois...'
python tools/kitti_object/generate_psmnet_input_inf.py \
--output_dir data/cyclist_roi \
--shape_prior_base notused \
--prediction_template models/kitti/cyclist/mask/inference/kitti_%s_cyclist/predictions.pth \
--cls cyclist

echo 'train iDispNet...'
python -m torch.distributed.launch --nproc_per_node $NGPUS \
tools/kitti_object/train_idispnet_fa.py \
--data_dir data/cyclist_roi \
--model_dir models/kitti/cyclist/idispnet