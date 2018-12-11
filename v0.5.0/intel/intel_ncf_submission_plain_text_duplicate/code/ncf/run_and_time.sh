#!/bin/bash
# runs benchmark and reports time to convergence
# to use the script:
#   run_and_time.sh <parallelism> <random seed> <learning rate> <batch size> <beta1> <beta2> <max epoch> <eps>

THRESHOLD=0.635
BASEDIR=$(dirname -- "$0")

# start timing
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"

# Get command line parallelism
parallelism=${1:-28}

# Get command line seed
seed=${2:-1}

# learning rate
lr=${3:-0.003821}

# BatchSize
b=${4:-98304}

# beta1
beta1=${5:-0.783529}

# beta2
beta2=${6:-0.909003}

# maxEpoch
e=${7:-20}

# epsilon
eps=${8:-1.45439E-07}

# useLazyAdam
useLazyAdam=${9:-false}

echo "parallelism=$parallelism, random seed=$seed, learning rate=$lr, batch size=$b, beta1=$beta1, beta2=$beta2, max epoch=$e, epsilon=$eps, useLazyAdam=$useLazyAdam"
echo "unzip ml-20m.zip"
if unzip -o ml-20m.zip
then
    echo "Start training"
    t0=$(date +%s)
    spark-submit --master "local[$parallelism]" --driver-memory 40g \
      --conf "spark.driver.extraJavaOptions=-Dbigdl.utils.Engine.defaultPoolSize=$parallelism -Dbigdl.localMode=true -Dbigdl.coreNumber=$parallelism" \
      --class com.intel.analytics.zoo.examples.mlperf.recommendation.NeuralCFexample \
      target/ncf-0.1.0-SNAPSHOT-jar-with-dependencies.jar \
      --inputDir ml-20m -b $b -e $e --valNeg 999 --layers 256,256,128,64 --numFactors 64 \
      --dataset ml-20m -l $lr --seed $seed --threshold $THRESHOLD --beta1 $beta1 --beta2 $beta2 --eps $eps --useLazyAdam $useLazyAdam
    t1=$(date +%s)
	delta=$(( $t1 - $t0 ))
    echo "Finish training in $delta seconds"

	# end timing
	end=$(date +%s)
	end_fmt=$(date +%Y-%m-%d\ %r)
	echo "ENDING TIMING RUN AT $end_fmt"


	# report result
	result=$(( $end - $start ))
	result_name="recommendation"


	echo "RESULT,$result_name,$seed,$result,$USER,$start_fmt"
else
	echo "Problem unzipping ml-20.zip"
	echo "Please run 'download_data.sh && verify_datset.sh' first"
fi

