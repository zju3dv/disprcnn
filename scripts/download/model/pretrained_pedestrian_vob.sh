mkdir -p models/kitti/pedestrian/vob/idispnet models/kitti/pedestrian/vob/rcnn
# idispnet
gdown --id 1Sb5JXtZDJs5yo9TrErV16zrQrao6dABL -O models/kitti/pedestrian/vob/idispnet/bestmodel.pth
# rcnn
gdown --id 14chVMIpdh3luc1HBSmFVX2xtOVcpFLrL -O models/kitti/pedestrian/vob/rcnn/pointrcnn.pth