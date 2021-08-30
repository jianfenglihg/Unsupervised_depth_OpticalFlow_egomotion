# Unsupervised_geometry

## Requirements

## KITTI
### Evaluate
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

### To be continued