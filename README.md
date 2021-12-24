# [Unsupervised Joint Learning of Depth, Optical Flow, Ego-motion from Video](https://arxiv.org/abs/2105.14520)

![Depth](img/depth.gif)

![flow](img/flow.gif)

## Introduction

We use a joint self-supervised method to estimate the three geometric elements, depth , optical flow, and ego-motion. Compared with other joint self-supervised methods like EPC++, CC, we achieved more precise results in KITTI dataset.

#### pose evaluation

| method   | kitti seq.09(mean ± std) | kitti seq.10(mean ± std) |
| -------- | ------------------------ | ------------------------ |
| CC       | 0.012 ± 0.007            | 0.012 ± 0.008            |
| EPC++ M  | 0.012 ± 0.007            | 0.013 ± 0.008            |
| GLNet    | 0.011 ± 0.006            | 0.011 ± 0.009            |
| **Ours** | **0.0098 ± 0.0059**      | **0.0090 ± 0.0074**      |

#### optical flow evaluation

| method   | kitti2012 train | kitti2012 test | kitti2015 train | kitti2015 test |
| -------- | --------------- | -------------- | --------------- | -------------- |
| CC-ft    | -               | -              | 5.66            | 25.27%         |
| EPC++    | 2.3             | 2.6            | 5.84            | 21.56%         |
| GLNet    | -               | -              | 8.35            | -              |
| **Ours** | **1.97**        | **2.2**        | **5.66**        | **19.48%**     |

#### depth evaluation (on kitti eigen split)

| method | Abs Rel   | Sq Rel    | RMSE      | RMSE log  |
| ------ | --------- | --------- | --------- | --------- |
| CC     | 0.140     | 1.070     | 5.326     | 0.217     |
| EPC++  | 0.141     | 1.029     | 5.350     | 0.216     |
| GLNet  | 0.135     | 1.070     | **5.230** | **0.210** |
| Ours   | **0.138** | **0.970** | 5.460     | 0.231     |

## Requirements

The code is based on Python3.6. You could use either virtualenv or conda to setup a specified environment. And then run:

`pip install -r requirements.txt`

## Prepare data:

1. Download KITTI raw dataset using the <a href="http://www.cvlibs.net/download.php?file=raw_data_downloader.zip">script</a> provided on the official website. You also need to download <a href="http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow">KITTI 2015 dataset</a> to evaluate the predicted optical flow, and <a href="http://www.cvlibs.net/datasets/kitti/eval_odometry.php">KITTI Odometry</a> to evaluate the pose. 

## Train

1. Modify the configuration file in the ./config directory to set up your path. The config file contains the important paths and default hyper-parameters used in the training process.

```bash
1. python train.py --config_file ./config/kitti_geom.yaml --gpu [gpu_id] --mode flow --prepared_save_dir [name_of_your_prepared_dataset] --model_dir [your/directory/to/save/training/models]
2. python train.py --config_file ./config/kitti_geom.yaml --gpu [gpu_id] --mode depth --prepared_save_dir [name_of_your_prepared_dataset] --model_dir [your/directory/to/save/training/models]
3. python train.py --config_file ./config/kitti_geom.yaml --gpu [gpu_id] --mode geom --flow_pretrained_model [your/file/to/save/training/models/flow_weights] --depth_pretrained_model [your/file/to/save/training/models/depth_weights] --prepared_save_dir [name_of_your_prepared_dataset] --model_dir [your/directory/to/save/training/models]
```

If you are running experiments on the dataset for the first time, it would first process data and save in the [prepared_base_dir] path defined in your config file. 

## Evaluate

the network weights after training on kitti raw data is [here](https://drive.google.com/file/d/1HEXI4v5Xsd1FspYlUE-7p96_pLrG67mJ/view?usp=sharing)

1. To evaluate the optical flow estimation on KITTI 2015 and KITTI 2012, run:
```bash
python test.py --config_file ./config/kitti_geom.yaml --gpu [gpu_id] --mode geom --task [kitti_flow_2015/kitti_flow_2012] --pretrained_model [path/to/your/model]
```

2. To evaluate the depth estimation on KITTI eigen split, run:
```bash
python test.py --config_file ./config/kitti_geom.yaml --gpu [gpu_id] --mode geom --task kitti_depth --pretrained_model [path/to/your/model]
```

3. To evaluate the pose estimation on KITTI Odometry, run:
```bash
python test.py --config_file ./config/kitti_geom.yaml --gpu [gpu_id] --mode geom --task kitti_pose --pretrained_model [path/to/your/model]
```

### Acknowledgement

We implemented our idea based on <a href="https://github.com/B1ueber2y/TrianFlow">TrainFlow</a>

