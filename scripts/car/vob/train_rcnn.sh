# copy RPN
mkdir models/kitti/pob/rcnn #todo
cp models/kitti/pob/rpn/model_0027462.pth models/kitti/pob/rcnn/model_0000000.pth #todo
#todo
python -c "import torch;ckpt=torch.load('models/kitti/pob/rcnn/model_0000000.pth','cpu');ckpt['iteration']=0;torch.save(ckpt,'models/kitti/pob/rcnn/model_0000000.pth')"
#todo
echo models/kitti/pob/rcnn/model_0000000.pth > models/kitti/pob/rcnn/last_checkpoint
# train rcnn
#todo
python -m torch.distributed.launch --nproc_per_node $NGPUS tools/train_net.py --config-file configs/kitti/pob/rcnn.yaml