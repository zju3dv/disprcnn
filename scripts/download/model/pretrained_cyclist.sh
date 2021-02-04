mkdir -p models/kitti/cyclist/idispnet models/kitti/cyclist/rcnn
# idispnet
gdown --id 1ItXomzyNKEOy_nqCKUiIVJ52WWyOdhPe -O models/kitti/cyclist/idispnet/bestmodel.pth
# prcnn
gdown --id 10NL9gyfAI_UMnet01aekIK27g8FrneIV -O models/kitti/cyclist/rcnn/pointrcnn.pth