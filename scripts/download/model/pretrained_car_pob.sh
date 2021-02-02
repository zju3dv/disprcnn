mkdir -p models/kitti/car/pob/idispnet models/kitti/car/pob/mask models/kitti/car/pob/rcnn
gdown --id 1d1SNW7kvWVc1Lj4qBP7okEtou3G5Jl2G -O models/kitti/car/pob/mask/smrcnn.pth

gdown --id 1Bp4VjWcydtawtKjk9BHJS6XWOozZ65yg -O models/kitti/car/pob/idispnet/bestmodel.pth

gdown --id 145T5lo1sgEddbvsYxI6kKqN9Jzo8FYeQ -O models/kitti/car/pob/rcnn/pointrcnn.pth