#!/bin/bash
/bin/rm -r build/ disprcnn.egg-info
python setup.py build develop
cd disprcnn/modeling/pointnet_module/point_rcnn/lib/pointnet2_lib/pointnet2
/bin/rm -r build/ dist/ pointnet2.egg-info
python setup.py install

cd ../../utils/iou3d/
/bin/rm -r build dist iou3d.egg-info
python setup.py install

cd ../roipool3d/
/bin/rm -r build dist roipool3d.egg-info
python setup.py install

cd ../../../../../../../

# optional, for easy evaluation.

# mkdir -p ~/bin/
# cp -r tools/kitti_object/kitti_evaluation_lib ~/bin/
# chmod +x ~/bin/kitti_evaluation_lib/evaluate_object_0.5
# chmod +x ~/bin/kitti_evaluation_lib/evaluate_object_0.7