# copy RPN
mkdir models/kitti/car/pob/rcnn
cd models/kitti/car/pob
cp rpn/model_0027462.pth rcnn/model_0000000.pth
cd rcnn/
python -c "import torch;ckpt=torch.load('model_0000000.pth','cpu');ckpt['iteration']=0;torch.save(ckpt,'model_0000000.pth')"
#todo
echo $(realpath model_0000000.pth) > last_checkpoint
# train rcnn
cd ../../../../../
python -m torch.distributed.launch --nproc_per_node $NGPUS tools/train_net.py \
--config-file configs/kitti/car/pob/rcnn.yaml