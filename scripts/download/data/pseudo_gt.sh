#car-pob
mkdir -p data/kitti/object/training/pob/
gdown --id 16vqA5ZFyirqorv_GYkoWn8QdNx-SLeuZ -O data/kitti/object/training/pob/
unzip data/kitti/object/training/pob/car_pob.zip -d data/kitti/object/training/pob/
rm data/kitti/object/training/pob/car_pob.zip

#car-vob
mkdir -p data/kitti/object/training/vob/
gdown --id 13azDR-e71y2w8psSy9CcZHN8i9VwZyQ7 -O data/kitti/object/training/vob/
unzip data/kitti/object/training/vob/car_vob.zip -d data/kitti/object/training/vob/
rm data/kitti/object/training/vob/car_vob.zip

# kins_mask_for pedestrian and cyclist
gdown --id 156SKS_kaShRvzSDcSLNf5mPL_jsySS23 -O data/kitti/object/training/
mkdir -p data/kitti/object/training/kins_mask_2/
unzip data/kitti/object/training/kins_mask_2.zip -d data/kitti/object/training/kins_mask_2
rm data/kitti/object/training/kins_mask_2.zip

#pedestrian-pob
mkdir -p data/kitti/object/training/pob/
gdown --id 19VLb4X4uPB5jmZutOivBrXEJBVWeW0n7 -O data/kitti/object/training/pob/
unzip data/kitti/object/training/pob/pedestrian_pob.zip -d data/kitti/object/training/pob/
rm data/kitti/object/training/pob/pedestrian_pob.zip

#pedestrian-vob
mkdir -p data/kitti/object/training/vob/
gdown --id 1NeL6cuRJiUgiN1bLCJDnEcYadzxB9sGb -O data/kitti/object/training/vob/
unzip data/kitti/object/training/vob/pedestrian_vob.zip -d data/kitti/object/training/vob/
rm data/kitti/object/training/vob/pedestrian_vob.zip

#cyclist
mkdir -p data/kitti/object/training/
gdown --id 1SsRgx57wkvOGscPA1PINwJtejC3CQmje -O data/kitti/object/training/
unzip data/kitti/object/training/cyclist.zip -d data/kitti/object/training/
rm data/kitti/object/training/cyclist.zip

