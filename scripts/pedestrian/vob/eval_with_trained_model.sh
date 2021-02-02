python -m torch.distributed.launch --nproc_per_node $NGPUS tools/test_net.py \
--config-file configs/kitti/pedestrian/vob/idispnet.yaml \
DATASETS.TEST "('kitti_val_vob_pedestrian',)"

python -m torch.distributed.launch --nproc_per_node $NGPUS tools/test_net.py \
--config-file configs/kitti/pedestrian/vob/rcnn.yaml \
--ckpt models/kitti/pedestrian/vob/rcnn/pointrcnn.pth \
DATASETS.TEST "('kitti_val_vob_pedestrian',)"