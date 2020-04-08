# Disp R-CNN: Stereo 3D Object Detection via Shape Prior Guided Instance Disparity Estimation (CVPR 2020)

This project contains the implementation of our CVPR 2020 paper [arxiv](https://arxiv.org/pdf/2004.03572.pdf).

Authors: Jiaming Sun, Linghao Chen, Yiming Xie, Siyu Zhang, Qinhong Jiang, Hujun Bao, Xiaowei Zhou.

<img src="docs/arch0328.png" width = 100% div align=left />


<br/>


## Requirements

- Ubuntu 16.04+
- Python 3.7+
- 8 Nvidia GPU with mem >= 12G (recommended, see [Notes](docs/TRAIN_VAL.md#Notes) for details.)
- GCC >= 4.9
- PyTorch 1.2.0

## Install
```bash
# Install webp support
sudo apt install libwebp-dev
# Install conda environment
conda env create -f environment.yaml
conda activate disprcnn
# Install Disp R-CNN
git clone https://github.com/zju3dv/disprcnn.git
cd disprcnn
sh build_and_install.sh
```

## Training and evaluation
See [TRAIN_VAL.md](docs/TRAIN_VAL.md)

## Sample results

<img src="docs/qualitative.png" width = 100% div align=left />

<br/>

## Citation

If you find this code useful for your research, please use the following BibTeX entry.

```
@inproceedings{sun2020disprcnn,
  title={Disp R-CNN: Stereo 3D Object Detection via Shape Prior Guided Instance Disparity Estimation},
  author={Sun, Jiaming and Chen, Linghao and Xie, Yiming and Zhang, Siyu and Jiang, Qinhong and Zhou, Xiaowei and Bao, Hujun},
  booktitle={CVPR},
  year={2020}
}
```

## Acknowledgment

This repo is built based on the Mask R-CNN implementation from [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark), and we also use the pretrained Stereo R-CNN weight from [here](https://drive.google.com/file/d/1rZ5AsMms7-oO-VfoNTAmBFOr8O2L0-xt/view?usp=sharing) for initialization.


## License
MIT

Copyright (c) 2020 3D Vision Group of State Key Lab at CAD&CG, Zhejiang University


