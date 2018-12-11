# 1. Problem

This problem uses the ResNet-50 CNN to do image classification.

## Requirements
* [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)
* [MXNet 18.11-py3 NGC container](https://ngc.nvidia.com/registry/nvidia-mxnet)

# 2. Directions
## Steps to download and verify data
Download the data using the following command:

```
cd ..
bash download_dataset.sh
cd -
```

Verify data with:

```
cd ..
bash verify_dataset.sh
cd -
```
## Steps to launch training

### NVIDIA DGX-1 (single node)
Launch configuration and system-specific hyperparameters for the NVIDIA DGX-1
single node submission are in the `config_DGX1.sh` script.

Steps required to launch single node training on NVIDIA DGX-1:

```
docker build . -t mlperf-nvidia:image_classification
DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> DGXSYSTEM=DGX1 ./run.sub
```

### NVIDIA DGX-2 (single node)
Launch configuration and system-specific hyperparameters for the NVIDIA DGX-2
single node submission are in the `config_DGX2.sh` script.

Steps required to launch single node training on NVIDIA DGX-2:

```
docker build . -t mlperf-nvidia:image_classification
DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> DGXSYSTEM=DGX2 ./run.sub
```

### NVIDIA DGX-1 (multi node)
Launch configuration and system-specific hyperparameters for the NVIDIA DGX-1
multi node submission are in the `config_DGX1_multi.sh` script.

Steps required to launch multi node training on NVIDIA DGX-1:

```
docker build . -t mlperf-nvidia:image_classification
DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> DGXSYSTEM=DGX1_multi sbatch -N $DGXNNODES -t $WALLTIME run.sub
```

### NVIDIA DGX-2 (multi node)
Launch configuration and system-specific hyperparameters for the NVIDIA DGX-2
multi node submission are in the `config_DGX2_multi.sh` script.

Steps required to launch multi node training on NVIDIA DGX-2:

```
docker build . -t mlperf-nvidia:image_classification
DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> DGXSYSTEM=DGX2_multi sbatch -N $DGXNNODES -t $WALLTIME run.sub
```





