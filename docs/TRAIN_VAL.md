## Setup dataset
1. Setup KITTI object dataset to the following folder structure
   ```bash
   disprcnn/ # project root
   ├─ data
   │  ├─ kitti
   │  │  ├─ object
   │  │  │  ├─ split_set
   │  │  │     ├─ train_set.txt & val_set.txt & test_set.txt & training_set.txt(same as train.txt)
   │  │  │  ├─ training
   │  │  │     ├─ calib & label_2 & label_3 & image_2 & image_3
   │  │  │  ├─ testing
   │  │  │     ├─ calib & image_2 & image_3
   ```
2. Download segmentation mask and disparity pseudo-ground-truth from [here](https://drive.google.com/drive/folders/1UjHQDs1tU_TQaWsLgAXMEdjY-ioSfOa5).
As described in the paper, we generate two types of pseudo-ground-truth: with and without the LiDAR points. These two variants of pseudo-GT correspond to *pob* and *vob* in the following experiments, respectively.
  
   ```bash
   tar -xvf pob.tar.gz
   mv pob data/kitti/object/training/
   tar -xvf vob.tar.gz
   mv vob data/kitti/object/training/
   ```
## Training and inference on train-val split

We provide bash scripts to run the *pob* version.

To run the *vob* version, replace *vob* with *pob* in every bash script.

Note that the current implementation cannot exit itself automatically, see [Notes](#Notes) for details.

1. Define the number of GPUs.
    
    ```bash
    export NGPUS=8
    ```
    
2. train Stereo Mask R-CNN.<br>
    Download pretrained Stereo R-CNN in Mask R-CNN format from [here](https://drive.google.com/open?id=1PCl94kXSeJeoToS6s0JJ4Zfiu98ukkMd), and put it in `models/kitti/disprcnn_R101_FPN_2d_srcnn_ckpt` directory.

    Then train the Stereo Mask R-CNN with the following command.

    ```bash
    sh scripts/train_msrcnn.sh # This step cost ~2 hours using 4 GPUs.
    ```

3. train iDispNet.<br>
   Download `pretrained_model_KITTI2015.tar` from [here](https://drive.google.com/file/d/1pHWjmhKMG4ffCrpcsp_MTXMJXhgl3kF9/view) and put it in `models/PSMNet` directory.

   We use the fast.ai framework to train the iDispNet as in `train_idispnet_fa.py`. The training script without this dependency is also provided in `train_idispnet.py`.

   ```bash
   sh scripts/train_idispnet.sh # This step cost ~8 hours using 8 GPUs.
   ```

4. train RPN
   ```bash
   sh scripts/train_rpn.sh # This step cost ~5 hours using 8 GPUs.
   ```

5. train RCNN
   ```bash
   sh scripts/train_rcnn.sh # This step cost ~13 hours using 8 GPUs.
   ```

6. evaluate RCNN
   ```bash
   sh scripts/eval_rcnn.sh # This step cost ~2min using 8 GPUs.
   ```

## Evaluate with our trained model

The released models are trained on the training split of the KITTI object training set.

Download [Ours version](https://drive.google.com/open?id=1-Xc0zs--w06xbNaH7Usl99a-Cl5MVXHT) or [Ours(velo) version](https://drive.google.com/open?id=1v31s3sl3lfaKMIxSHr_hhb4Vx8IwIgmY) and put them in `models/kitti/pob` or `models/kitti/vob` directory.

We provide scripts to evaluate with the trained pob model (Ours version). Replace *vob* with *pob* in the script to run Ours (velo) version.

When you have multiple GPUs with more than 12G memory (e.g. RTX TITAN/V100), run the first command to perform distributed inference on multiple GPUs. Otherwise, you are recommended to run the second one, which use only one GPU.

   ```bash
   # distributed inference with multiple GPUs.
   export NGPUS=8 
   sh scripts/eval_with_trained_model_dist.sh
   # inference with one GPU.
   sh scripts/eval_with_trained_model.sh
   ```

## Notes

1. Setting num_workers>0 could speed up training and inference a lot, but due to some bugs in PyTorch, training won't finish all iterations, and inference won't exit automatically after printing out results. You should interrupt the program manually by "Ctrl+C" when training stucks at the end or inference has printed out results. If GPU memories are not released after interrupting, use the following command to release GPU memories. This can be potentially fixed via switching the distributed launcher to [`mp.spawn`](https://github.com/facebookresearch/detectron2/blob/master/detectron2/engine/launch.py#L45).

   ```bash
   pkill -e -9 python -u $USER
   ```

2. We suggest to use 8 GPUs with more than 12G memory each. If you don't have enough GPUs or your GPU memory is less than 12G, there are some alternatives. We provide a script to run inference with one GPU. For training, you can decrease the batch size and learning rate, and increase maximum iteration following [scheduling rules from Detectron](https://github.com/facebookresearch/Detectron/blob/master/configs/getting_started/tutorial_1gpu_e2e_faster_rcnn_R-50-FPN.yaml#L14-L30).
