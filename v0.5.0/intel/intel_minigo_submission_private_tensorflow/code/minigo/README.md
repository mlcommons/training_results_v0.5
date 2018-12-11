# 1. Problem
This task benchmarks on policy reinforcement learning for the 9x9 version of the board game go. The model plays games against itself and uses these games to improve play.
This implementation is derived from original `mlperf/training/reinforcement` reference implementation, and optimized for Intel® Xeon® scalable processors.

# 2. Directions
### Steps to configure machine
To setup the environment on CentOS 7.4 (112 CPUs, 100 GB disk), you can use these commands. This may vary on a different operating system and hardware environment.

1. Install Python3

    Install Python3 on your Linux distribution, we verified our implementation on Python 3.4, but any minor version should work.

2. Install requirements

    `pip3 install -r tensorflow/minigo/requirements.txt`

3. Install MLPerf Compliance Logging Utilities and Helper Functions

    `git clone https://github.com/mlperf/training`
    `cd training/compliance`
    `python3 setup.py install`

4. Install TensorFlow fork with sampling dataset

   Our method use a sampling dataset to replace the filter dataset with a filter function.   This is mathematical equivalent but will make training faster.  To use the sampling dataset, get TensorFlow fork from the following branch:
   https://github.com/delock/tensorflow/tree/sampling_dataset
   This branch bases on TensorFlow branch r1.10 with sampling dataset.  To build this branch, follow the 'Build TensorFlow from Source with Intel MKL' section in the following link:
   https://software.intel.com/en-us/articles/intel-optimization-for-tensorflow-installation-guide

### Steps to download and verify data
Unlike other benchmarks, there is no data to download. All training data comes from games played during benchmarking.

### Steps to run and time

To run, this assumes you get code from minigo submission package and put it to `~/mlperf_minigo/` directory

If you are running on 2 socket Intel® Xeon® Platinum 8180 Processors, run the following
    `cd ~/mlperf_minigo/`
    `./run_and_time_skx_8180_2s_1x.sh | tee result_<N>.txt`

If you are running on 4 socket Intel® Xeon® Platinum 8180 Processors, run the following
    `cd ~/mlperf_minigo/`
    `./run_and_time_skx_8180_4s_1x.sh | tee result_<N>.txt`

# 3. Model
### Publication/Attribution

This benchmark is based on a fork of the minigo project (https://github.com/tensorflow/minigo); which is inspired by the work done by Deepmind with ["Mastering the Game of Go with Deep Neural Networks and
Tree Search"](https://www.nature.com/articles/nature16961), ["Mastering the Game of Go without Human
Knowledge"](https://www.nature.com/articles/nature24270), and ["Mastering Chess and Shogi by
Self-Play with a General Reinforcement Learning
Algorithm"](https://arxiv.org/abs/1712.01815). Note that minigo is an
independent effort from AlphaGo, and that this fork is minigo is independent from minigo itself. 

### Reinforcement Setup

This benchmark includes both the environment and training for 9x9 go. There are four primary phases in this benchmark, these phases are repeated in order:

 - Selfplay: the *current best* model plays games against itself to produce board positions for training.
 - Training: train the neural networks selfplay data from several recent models. 
 - Target Evaluation: the termination criteria for the benchmark is checked using the provided record of professional games. 
 - Model Evaluation: the *current best* and the most recently trained model play a series of games. In order to become the new *current best*, the most recently trained model must win 55% of the games.

### Structure

This task has a non-trivial network structure, including a search tree. A good overview of the structure can be found here: https://medium.com/applied-data-science/alphago-zero-explained-in-one-diagram-365f5abf67e0. 

### Weight and bias initialization and Loss Function
Network weights are initialized randomly. Initialization and loss are described here;
["Mastering the Game of Go with Deep Neural Networks and Tree Search"](https://www.nature.com/articles/nature16961)

### Optimizer
We use a MomentumOptimizer to train the primary network. 

# 4. Model optimization
This implementation is derived from reference TensorFlow based minigo python implementation at https://github.com/mlperf/training/tree/master/reinforcement

This implementation do the following optimizations on top of reference implementation:
1. concurrent selfplay and evaluation
    Reference implementation use 16 worker instances for selfplay, assuming running on a 16 core machine.  This implementation use number of worker instances that equals to the number of hyperthreads on the system.  We also pin each worker to different logical CPUs to maximize performance.
    Reference implementation use single process for selfplay evaluation (old vs. new stage).  This implemenation use 100 instances to play all the 50+50 selfplay evaluation games concurrently.
    Reference implementation use single process for predict in evaluation.  There are 4 sgf files, each sgf file needs to be evaluated twice.  This implementation run 8 instances concurrently, each instances to evaluate one sgf file for one time.
2. Batch size optimized for training
    Reference implementation use batch size = 16 in training stage.  This implementation use batch size = 64 for better training performance.  Both learning rate and learning rate step is adjusted accordingly.
3. Use freezed graph for selfplay
    For selfplay in evaluation and selfplay in training data generation, this implemenation freeze the graph before inference.  Consequently, constants are folded together so there is performance improvement in selfplay stages.
4. Pipeline predict evaluation stage
    We run predict evaluate stage in parallel with selfplay evaluation stage, selfplay stage of next epoch, and training of next epoch.  So predict evaluation stage overhead is totally hidden.

# 5. Quality

Due to the difficulty of training a highly proficient go model, our quality metric and termination criteria is based on predicting moves from human reference games. Currently published results indicate that it takes weeks of time and/or cluster sized resources to achieve a high level of play. Given more limited time and resources, it is possible to predict a significant number of moves from professional or near-professional games. 

### Quality metric

Provided in with this benchmark are records of human games and the quality metric is the percent of the time the model chooses the same move the human chose in each position. Each position is attempted twice by the model (keep in mind the model's choice is non-deterministic). The metric is calculated as the number of correct predictions divided by the number of predictions attempted. 

The particular games we use are from Iyama Yuta 6 Title Celebration, between contestants Murakawa Daisuke, Sakai Hideyuki, Yamada Kimio, Hyakuta Naoki, Yuki Satoshi, and Iyama Yuta.



### Quality target
The quality target is predicting 40% of the moves.

### Evaluation frequency
Evaluation should be preformed for every model which is trained (regardless if it wins the "model evaluation" round). 
    

### Evaluation thoroughness
All positions should be considered in each evaluation phase.
