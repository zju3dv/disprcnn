python -m torch.distributed.launch --nproc_per_node $NGPUS tools/test_net.py \
--config-file configs/kitti/car/vob/mask.yaml --ckpt models/kitti/car/vob/mask/smrcnn.pth \
DATASETS.TEST "('kitti_val_vob_car',)"

python -m torch.distributed.launch --nproc_per_node $NGPUS tools/test_net.py \
--config-file configs/kitti/car/vob/idispnet.yaml \
DATASETS.TEST "('kitti_val_vob_car',)"

python -m torch.distributed.launch --nproc_per_node $NGPUS tools/test_net.py \
--config-file configs/kitti/car/vob/rcnn.yaml \
--ckpt models/kitti/car/vob/rcnn/pointrcnn.pth \
DATASETS.TEST "('kitti_val_vob_car',)"