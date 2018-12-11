# Problem
This task benchmarks image classification on ImageNet dataset.

# Directions
## 1. Downloading and Building Intel Caffe
Ensure GCC4.8.5 is installed on the machine.
#### SKX 8180 2 socket single node run (skx_8180_2s_1x):
      `./setup_skx_8180_2s_1x.sh`
#### SKX 8180 2 socket multinode node run (skx_8180_2s_8x):
	  `./setup_skx_8180_2s_8x.sh`
#### SKX 8180 4 socket multinode node run (skx_8180_4s_4x):	
	  `./setup_skx_8180_4s_4x.sh`
An “intelcaffe” folder will be created with built caffe.

## 2. Prepare for multi node training
#### Configure password less SSH
      (1) Generate ssh key and append public key to authorized_keys
        `ssh-keygen -t rsa`
        `cat /home/user/.ssh/id_rsa.pub >> authorized_keys`
      (2) Disable strict key checking
        `sudo vim /etc/ssh/ssh_config`
    Add “StrictHostKeyChecking no” to ssh_config file
Additional notes in https://github.com/intel/caffe/wiki/Multinode-guide

## 3. Prepare ImageNet dataset for Intel Caffe
  Create ImageNet LMDB based on the instructions in https://github.com/intel/caffe/wiki/How-to-create-ImageNet-LMDB
      * Note: In our test ImageNet LMDB was created with encoding but without resizing, thus make sure setting “ENCODE=true” and “RESIZE=false” in create_imagenet.sh script

## 4. Prepare solver.prototxt (hyperparameters) and train_val.prototxt(model)
	The prototxt's are under prototxt/ folder

## 5. Running
### a. Prepare the hostfile
  (1) Put the list of host names in a temp file, then sort the host names, one hostname one line. 
	`sort -V temp > hostfile`
  (2) Upload hostfile to master node “intelcaffe” folder. 

### b. Login to master node and run Intel Caffe training
#### SKX 8180 2 socket single node run (skx_8180_2s_1x):
    `cp run_and_time_skx_8180_2s_1x.sh intelcaffe`
	`cp –r prototxt/skx_8180_2s_1x intelcaffe`
	`cd intelcaffe`
	`sh run_and_time_skx_8180_2s_1x.sh skx_8180_2s_1x/solver.prototxt  hostfile 0.749`
	After test finish the MLPv0.5_resnet_intelcaffe_skx_8180_2s_1x.txt log will be generated at working folder.
#### SKX 8180 2 socket multinode node run (skx_8180_2s_8x): 
	`cp run_and_time_ skx_8180_2s_8x.sh intelcaffe`
	`cp –r prototxt/skx_8180_2s_8x intelcaffe`
	`cd intelcaffe`
	`sh run_and_time_skx_8180_2s_8x.sh skx_8180_2s_8x/solver.prototxt  hostfile 0.749`
	After test finish the MLPv0.5_resnet_intelcaffe_ skx_8180_2s_8x. txt log will be generated at working folder.
#### SKX 8180 4 socket multinode node run (skx_8180_4s_4x):  
	`cp run_and_time_skx_8180_4s_4x.sh intelcaffe`
	`cp –r prototxt/skx_8180_4s_4x intelcaffe`
	`cd intelcaffe`
	`sh run_and_time_skx_8180_4s_4x.sh skx_8180_4s_4x/solver.prototxt  hostfile 0.749`  
	After test finish the MLPv0.5_resnet_intelcaffe_ skx_8180_4s_4x. txt log will be generated at working folder.

## Best performance solution
Please read [our Wiki](https://github.com/intel/caffe/wiki/Recommendations-to-achieve-best-performance) for our recommendations and configuration to achieve best performance on Intel CPUs. 

# Quality
## Quality metric
Top-1 accuracy.

## Quality target
Top-1 accuracy: 74.9

# Evaluation frequency
After every epoch through the training data.

# Customizations
The LR policy uses a base_lr=0.1. For multinode training, the warmup_lr=0.1 and reaches base_lr=0.4(for 1K batch size) with warm-up period=4 epochs

