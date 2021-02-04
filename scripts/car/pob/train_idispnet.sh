echo 'preparing offline predictions...'
python -m torch.distributed.launch --nproc_per_node $NGPUS tools/test_net.py \
--config-file configs/kitti/car/pob/mask.yaml

echo 'generating rois...'
python tools/kitti_object/generate_psmnet_input_inf.py \
--output_dir data/car_pob_roi \
--shape_prior_base pob
--prediction_template models/kitti/car/pob/mask/inference/kitti_%s_pob_car/predictions.pth \
--cls car

echo 'train iDispNet...'
python -m torch.distributed.launch --nproc_per_node $NGPUS \
tools/kitti_object/train_idispnet_fa.py \
--data_dir data/car_pob_roi \
--model_dir models/kitti/car/pob/idispnet