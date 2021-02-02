python -m torch.distributed.launch --nproc_per_node $NGPUS tools/test_net.py \
--config-file configs/kitti/car/pob/mask.yaml --ckpt models/kitti/car/pob/mask/smrcnn.pth \
DATASETS.TEST "('kitti_val_pob_car',)"

python -m torch.distributed.launch --nproc_per_node $NGPUS tools/test_net.py \
--config-file configs/kitti/car/pob/idispnet.yaml \
DATASETS.TEST "('kitti_val_pob_car',)"

python -m torch.distributed.launch --nproc_per_node $NGPUS tools/test_net.py \
--config-file configs/kitti/car/pob/rcnn.yaml \
--ckpt models/kitti/car/pob/rcnn/pointrcnn.pth \
DATASETS.TEST "('kitti_val_pob_car',)"