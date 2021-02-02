mkdir -p models/kitti/pedestrian/pob/idispnet models/kitti/pedestrian/pob/rcnn
# idispnet
gdown --id 1DvbIwVNV_HD5ELwgE46a5h9-GH1UWazX -O models/kitti/pedestrian/pob/idispnet/bestmodel.pth
# rcnn
gdown --id 1v9fQSbh9GrBA9IfIMHaAu4rL8Z7AjONr -O models/kitti/pedestrian/pob/rcnn/pointrcnn.pth