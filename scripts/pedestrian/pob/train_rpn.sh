# load idispnet, inference on train/val set
echo 'preparing offline predictions...'
python -m torch.distributed.launch --nproc_per_node $NGPUS tools/test_net.py \
--config-file configs/kitti/pedestrian/pob/idispnet.yaml
# train RPN
echo 'train RPN...'
python -m torch.distributed.launch --nproc_per_node $NGPUS tools/train_net.py \
--config-file configs/kitti/pedestrian/pob/rpn.yaml