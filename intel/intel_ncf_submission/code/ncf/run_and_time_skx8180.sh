#!/bin/bash
times=${1:-100}
times=`expr $times - 1`
mkdir logs
for i in $(seq 0 $times)
do
  seed=`date +%s`
  echo "sh run_and_time.sh 28 $seed"
  sh run_and_time.sh 28 $seed > logs/result_$i.txt
done
