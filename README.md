# PolarBEVU

Official code of the papar "PolarBEVU: Multi-Camera 3D Object Detection in Polar Bird's-Eye View via Unprojection".

## Get Started
### Environment
- CUDA 11.1
- Python 3.7.15
- PyTorch 1.9.0+cu111
- mmcv-full 1.5.2
- mmdet 2.24.0
- mmengine 0.7.2
- mmsegmentation 0.24.0

### Installation
Refer [here](docs/en/getting_started.md).
### Data Preparation

Refer [here](docs/en/datasets/nuscenes_det.md)

    └── data
        └── nuscenes
            ├── v1.0-trainval 
            ├── sweeps
            ├── samples 
            └── maps    

``` bash
python tools/create_data_polarbevu.py
```
### Train model
```bash
python tools/train.py $config
```
### Test model
```bash
python tools/test.py $config $checkpoint --eval mAP
```
