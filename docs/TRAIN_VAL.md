

## Update in 2021.2

We have made lots of updates in 2021.2, Please pull the commits to get the newest code and data.

1. Pedestrians and cyclists are supported.
2. New pseudo-GT data is released.
   1. Pseudo-GT for pedestrians and cyclists are released.
   2. Pseudo-GT for cars are updated. We add background disparity using depth completion for more robust training for the iDispNet, especially near the edge of segmentation mask.
3. New trained models are released to produce better results.
4. Easy download scripts are provided using [gdown](https://github.com/wkentaro/gdown).

## Setup dataset

1. Setup KITTI object dataset to the following folder structure
   ```bash
   disprcnn/ # project root
   ├─ data
   │  ├─ kitti
   │  │  ├─ object
   │  │  │  ├─ split_set
   │  │  │     ├─ train_set.txt & val_set.txt & test_set.txt & training_set.txt(same as train.txt) # todo
   │  │  │  ├─ training
   │  │  │     ├─ calib & label_2 & label_3 & image_2 & image_3
   │  │  │  ├─ testing
   │  │  │     ├─ calib & image_2 & image_3
   ```
2. Download label_3, segmentation mask and disparity pseudo-ground-truth.
As described in the paper, we generate two types of pseudo-ground-truth: with and without the LiDAR points. These two variants of pseudo-GT correspond to *vob* and *pob* in the following experiments, respectively.
  
   ```bash
   # download label_3 
   sh scripts/download/data/label_3.sh
   # download pseudo-GT
   sh scripts/download/data/pseudo_gt.sh
   ```
## Training and inference on train-val split

As an example, we describe steps to run the *vob* version for the car category here.

Other categories and versions are similar.

Note that the current implementation cannot exit itself automatically, see [Notes](#Notes) for details.

1. Define the number of GPUs.
  
    ```bash
    export NGPUS=8
    ```
    
2. 2D detector.<br>
  
    For the **car** category, download pretrained Stereo R-CNN in Mask R-CNN format.
	
    ```bash
	sh scripts/download/model/srcnn_pretrained_2d_mrcnn_format.sh
    ```
    
    Then train the Stereo Mask R-CNN with the following command.

    ```bash
	sh scripts/car/vob/train_smrcnn.sh # This step cost ~1.5 hours using 4 GPUs.
    ```

		For the pedestrian and cyclist categories, we provide 2D predictions, you can download them instead of training by yourself.

    ```bash
	sh scripts/download/model/pedestrian_2d.sh
	sh scripts/download/model/cyclist_2d.sh
    ```

3. train iDispNet.<br>
   Download `pretrained_model_KITTI2015.tar`
   
   ```bash
   sh scripts/download/model/pretrained_psmnet.sh
   ```
   
   We use the fast.ai framework to train the iDispNet as in `train_idispnet_fa.py`. 

   ```bash
   sh scripts/car/vob/train_idispnet.sh # This step cost ~8 hours using 8 GPUs.
   ```

4. train RPN
   ```bash
   sh scripts/car/vob/train_rpn.sh # This step cost ~5 hours using 8 GPUs.
   ```


5. train RCNN
   ```bash
   sh scripts/car/vob/train_rcnn.sh # This step cost ~13 hours using 8 GPUs.
   ```


6. evaluate RCNN
   ```bash
   sh scripts/eval_rcnn.sh # This step cost ~2min using 8 GPUs.
   ```


## Evaluate with our trained model

The released models are trained on the training split of the KITTI object training set.

As an example, we describe steps to run the *vob* version for the *car* category here. Other categories are similar.

When you have multiple GPUs with more than 12G memory (e.g. RTX TITAN/V100), run the first command to perform distributed inference on multiple GPUs. Otherwise, you are recommended to use only one GPU.

   ```bash
   # download pretrained model.
   sh scripts/download/model/pretrained_car_vob.sh
   # distributed inference with multiple GPUs.
   export NGPUS=8
   sh scripts/car/vob/eval_with_trained_model.sh
   # inference with one GPU.
   export NGPUS=1
   sh scripts/car/vob/eval_with_trained_model.sh
   ```

For pedestrians and cyclists, we provide 2D predictions instead of trained 2D detector. Download them using the following command.

   ```bash
   sh scripts/download/model/pedestrian_2d.sh 
   sh scripts/download/model/cyclist_2d.sh
   ```

## Notes

1. Setting num_workers>0 could speed up training and inference by a large margin. However, due to some bugs in PyTorch, the training process will hang itself after finishing nearly all the iterations. You should interrupt the program manually by "Ctrl+C" when training stucks at the end. You can validate if the training is finished by checking the ETA (usually less than one minute) or iter (close to max_iter in the configs). If the GPU memory is not released after manual interrupting, use the following command to release GPU memory. This imperfection can be potentially fixed via switching the distributed launcher to [`mp.spawn`](https://github.com/facebookresearch/detectron2/blob/master/detectron2/engine/launch.py#L45).

   ```bash
   pkill -e -9 python -u $USER
   ```
If you encouter EOFError or share_memory error during training process using num_workers>0, just relaunch the training scripts with "SOLVER.LOAD_OPTIMIZER True SOLVER.LOAD_SCHEDULER True".
2. We suggest to use 8 GPUs with more than 12G memory each. If you don't have enough GPUs or your GPU memory is less than 12G, there are some alternatives. We provide a script to run inference with one GPU. For training, you can decrease the batch size and learning rate, and increase maximum iteration following [scheduling rules from Detectron](https://github.com/facebookresearch/Detectron/blob/master/configs/getting_started/tutorial_1gpu_e2e_faster_rcnn_R-50-FPN.yaml#L14-L30).
