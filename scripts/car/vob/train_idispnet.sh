echo 'preparing offline predictions...'
python -m torch.distributed.launch --nproc_per_node $NGPUS tools/test_net.py \
--config-file configs/kitti/car/vob/mask.yaml

echo 'generating rois...'
python tools/kitti_object/generate_psmnet_input_inf.py \
--output_dir data/car_vob_roi \
--shape_prior_base vob \
--prediction_template models/kitti/car/vob/mask/inference/kitti_%s_vob_car/predictions.pth \
--cls car

echo 'train iDispNet...'
python -m torch.distributed.launch --nproc_per_node $NGPUS \
tools/kitti_object/train_idispnet_fa.py \
--data_dir data/car_vob_roi \
--model_dir models/kitti/car/vob/idispnet