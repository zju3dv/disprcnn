mkdir -p models/kitti/car/vob/idispnet models/kitti/car/vob/mask models/kitti/car/vob/rcnn
# smrcnn
gdown --id 1mN4gMSPlsNxhyHekFI11b-c30bEGQKqp -O models/kitti/car/vob/mask/smrcnn.pth
# idispnet
gdown --id 14G3844e_cXC05Hrqts3fr06HO0vq5FhS -O models/kitti/car/vob/idispnet/bestmodel.pth
# prcnn
gdown --id 14wjnsk0DaZigT-URTKdfmhXheEx2AwLf -O models/kitti/car/vob/rcnn/pointrcnn.pth