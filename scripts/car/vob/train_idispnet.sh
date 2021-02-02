echo 'preparing offline predictions...'
python -m torch.distributed.launch --nproc_per_node $NGPUS tools/test_net.py --config-file configs/kitti/car/pob/mask.yaml
# generate rois
echo 'generating rois...'
#todo
python tools/kitti_object/generate_psmnet_input_inf.py --output_dir data/pob_roi --prediction_template models/kitti/pob/mask/inference/kitti_%s_pob/predictions.pth --mask_disp_sub_path pob
# train iDispNet
#todo
echo 'train iDispNet...'
python -m torch.distributed.launch --nproc_per_node $NGPUS tools/kitti_object/train_idispnet_fa.py --data_dir data/pob_roi --mode train_oc --model_dir models/kitti/pob/idispnet --load_model models/PSMNet/pretrained_model_KITTI2015.tar