gdown --id 1KRy7LnMPagIOvoksuYVouAPK8yoaqnnL -O data/kitti/object/training/
mkdir -p data/kitti/object/training/label_3/
tar -zxvf data/kitti/object/training/label3.tar.gz -C data/kitti/object/training/label_3/
rm data/kitti/object/training/label3.tar.gz