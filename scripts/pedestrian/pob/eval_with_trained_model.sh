python -m torch.distributed.launch --nproc_per_node $NGPUS tools/test_net.py \
--config-file configs/kitti/pedestrian/pob/idispnet.yaml \
DATASETS.TEST "('kitti_val_pob_pedestrian',)"

python -m torch.distributed.launch --nproc_per_node $NGPUS tools/test_net.py \
--config-file configs/kitti/pedestrian/pob/rcnn.yaml \
--ckpt models/kitti/pedestrian/pob/rcnn/pointrcnn.pth \
DATASETS.TEST "('kitti_val_pob_pedestrian',)"