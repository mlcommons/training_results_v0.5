# 1. Problem
Object detection.

## Requirements
* [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)
* [PyTorch 18.11-py3 NGC container](https://ngc.nvidia.com/registry/nvidia-pytorch)

# 2. Directions

### Steps to download data
```
cd reference/single_stage_detector/
source download_dataset.sh
```

## Steps to launch training

### NVIDIA DGX-1 (single node)
Launch configuration and system-specific hyperparameters for the NVIDIA DGX-1
single node submission are in the `config_DGX1.sh` script.

Steps required to launch single node training on NVIDIA DGX-1:

```
docker build . -t mlperf-nvidia:single_stage_detector
DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> DGXSYSTEM=DGX1 ./run.sub
```

### NVIDIA DGX-2 (single node)
Launch configuration and system-specific hyperparameters for the NVIDIA DGX-2
single node submission are in the `config_DGX2.sh` script.

Steps required to launch single node training on NVIDIA DGX-2:

```
docker build . -t mlperf-nvidia:single_stage_detector
DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> DGXSYSTEM=DGX2 ./run.sub
```

### NVIDIA DGX-1 (multi node)
Launch configuration and system-specific hyperparameters for the NVIDIA DGX-1
multi node submission are in the `config_DGX1_multi.sh` script.

Steps required to launch multi node training on NVIDIA DGX-1:

```
docker build . -t mlperf-nvidia:single_stage_detector
DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> DGXSYSTEM=DGX1_multi sbatch -N $DGXNNODES -t $WALLTIME run.sub
```

### NVIDIA DGX-2 (multi node)
Launch configuration and system-specific hyperparameters for the NVIDIA DGX-2
multi node submission are in the `config_DGX2_multi.sh` script.

Steps required to launch multi node training on NVIDIA DGX-2:

```
docker build . -t mlperf-nvidia:single_stage_detector
DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> DGXSYSTEM=DGX2_multi sbatch -N $DGXNNODES -t $WALLTIME run.sub
```

### Hyperparameter settings

Hyperparameters are recorded in the `config_*.sh` files for each configuration and in `run_and_time.sh`.

# 3. Dataset/Environment
### Publiction/Attribution.
Microsoft COCO: COmmon Objects in Context. 2017.

### Training and test data separation
Train on 2017 COCO train data set, compute mAP on 2017 COCO val data set.

# 4. Model.
### Publication/Attribution
Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg. SSD: Single Shot MultiBox Detector. In the Proceedings of the European Conference on Computer Vision (ECCV), 2016.

Backbone is ResNet34 pretrained on ImageNet (from torchvision).

# 5. Quality.
### Quality metric
Metric is COCO box mAP (averaged over IoU of 0.5:0.95), computed over 2017 COCO val data.

### Quality target
mAP of 0.212

### Evaluation frequency
The model is evaluated at epochs 33, 44, 49, 55, 60 and 65.

### Evaluation thoroughness
All the images in COCO 2017 val data set.
