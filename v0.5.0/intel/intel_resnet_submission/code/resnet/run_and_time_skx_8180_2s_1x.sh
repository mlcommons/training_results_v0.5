WORKDIR=`pwd`

PREFIX="$PREFIX" 
TESTNAME="skx_8180_2s_1x"

solver=$1
hostfile=$2
target=$3
caffelog=$4
ppn=4
index=$5

if [ ! -e $solver ];then
   echo "Cannot find $solver!"
fi

if [ ! -e $hostfile ];then
   echo "Cannot find $hostfile!"
fi

if [[ -z $caffelog ]];then
if  [ ! -e "./scripts/run_intelcaffe.sh" ];then
   echo "Cannot find run_intelcaffe.sh, please run under Caffe working folder"
fi 

if  [ ! -e "./build/tools/caffe" ];then
   echo "Cannot find caffe binary, please build and run under Caffe working folder"
fi
fi
#extract hyperparameters from prototxt
mlperf_log="MLPv0.5_resnet_intelcaffe_${TESTNAME}_${index}.txt"
if [ -e $mlperf_log ];then
 rm $mlperf_log
fi

timestamp()
{
 if [[ -f $caffelog ]];then
    starttime=$(cat $caffelog | grep "\[0\].*MLPerf:.*run_start" | awk -F' ' '{ print $3 }')
    date_run=$(cat $caffelog | grep "\[0\].*MLPerf:.*run_start" | awk -F' ' '{ print $2 }' | sed "s/I//g")
    stamp=$(date --date="2018$date_run $starttime" +%s)
 else
    stamp=$(date +%s)
 fi
 echo $stamp
 return $stamp
}


PREFIX=":::MLPv0.5.0 resnet" 
prototxt=$(cat $solver|grep "net:"|awk -F':' '{ print $2 }'|awk -F'"' '{ print $2 }' ) 
echo "$PREFIX $(timestamp) ($prototxt:27) preproc_num_train_examples: 1281167"  >> $mlperf_log
echo "$PREFIX $(timestamp) ($prototxt:57) preproc_num_eval_examples: 50000" >> $mlperf_log
batch_size=$(cat $prototxt|grep -m 1 "batch_size"|awk -F':' '{ print $2 }')
echo "$PREFIX $(timestamp) ($prototxt:28) input_batch_size: $batch_size" >> $mlperf_log
#echo "$PREFIX $(timestamp) ($prototxt:28) input_size: $((batch_size*ppn))" >> $mlperf_log
max_iter=$(cat $solver|grep "max_iter"|awk -F':' '{ print $2 }'|awk -F'#' '{ print $1 }')
test_offset=$(cat $solver|grep "test_offset"|awk -F':' '{ print $2 }')
eval_epoch_offset=$(($test_offset / 1281167))
echo "$PREFIX $(timestamp) ($solver:12) eval_offset: $eval_epoch_offset" >> $mlperf_log
echo "$PREFIX $(timestamp) ($prototxt:31) input_order" >> $mlperf_log
crop=$(cat $prototxt|grep -m 1 "crop"|awk -F':' '{ print $2 }')
echo "$PREFIX $(timestamp) ($prototxt:31) input_central_crop: [$crop, $crop]" >> $mlperf_log
min_area_ratio=$(cat $prototxt|grep "min_area_ratio"|awk -F':' '{ print $2 }')
max_area_ratio=$(cat $prototxt|grep "max_area_ratio"|awk -F':' '{ print $2 }')
echo "$PREFIX $(timestamp) ($prototxt:17) input_distorted_crop_area_range: [$min_area_ratio, $max_area_ratio]"
max_attempt=$(cat $prototxt|grep "max_attempt"|awk -F':' '{ print $2 }')
echo "$PREFIX $(timestamp) ($prototxt:20) input_distorted_crop_max_attempts: $max_attempt" >> $mlperf_log
ratio=$(cat $prototxt|grep "aspect_ratio_change"|awk -F':' '{ print $2 }')
ratio2=$(echo "scale=1; 1 / $ratio" | bc -l)
echo "$PREFIX $(timestamp) ($prototxt:19) input_distorted_crop_aspect_ratio_range [$ratio, $ratio2]"
mirror=$(cat $prototxt|grep -m 1 "mirror"|awk -F': ' '{ print $2 }')
echo "$PREFIX $(timestamp) ($prototxt:17) input_random_flip: $mirror" >> $mlperf_log
echo "$PREFIX $(timestamp) ($prototxt:49) input_resize_aspect_preserving: {\"min\": 256}" >> $mlperf_log
input_resize=$(cat $prototxt|grep -m 1 "crop_size"|awk -F': ' '{ print $2 }')
echo "$PREFIX $(timestamp) ($prototxt:12) input_size: $input_resize" >> $mlperf_log
echo "$PREFIX $(timestamp) ($prototxt:13) input_mean_subtraction: [124, 117, 104]" >> $mlperf_log
echo "$PREFIX $(timestamp) ($solver:15) opt_name: \"stochastic_gradient_descent_with_momentum\"" >> $mlperf_log
momentum=$(cat $solver|grep "momentum:"|awk -F':' '{ print $2 }')
echo "$PREFIX $(timestamp) ($solver:15) opt_momentum:$momentum" >> $mlperf_log
lr=$(cat $solver|grep "base_lr:"|awk -F':' '{ print $2 }')
echo "$PREFIX $(timestamp) ($solver:15) opt_lr:$lr" >> $mlperf_log
echo "$PREFIX $(timestamp) ($prototxt:3298) model_hp_loss_fn: \"categorical_cross_entropy\"" >> $mlperf_log
weight_decay=$(cat $solver|grep "weight_decay:"|awk -F':' '{ print $2 }')
echo "$PREFIX $(timestamp) ($solver:16) model_l2_regularization: $weight_decay " >> $mlperf_log
#Set Radom seed
echo "$PREFIX 0000000000 (empty:0) run_set_random_seed: 0" >> $mlperf_log

run_intelcaffe="run_intelcaffe_${TESTNAME}"
if [[ -z $caffelog ]];then
./scripts/run_intelcaffe.sh --hostfile $hostfile --mode train --debug off --network opa  --engine MKLDNN --msg_priority off --caffe_bin ./build/tools/caffe --ppn $ppn --solver $solver --output run_intelcaffe_${TESTNAME} --benchmark none --num_mlsl_servers 1
fi

#extract MLPerf log from caffe log 
nodes=$(wc $hostfile -l|awk -F' ' '{ print $1 }')
log="$run_intelcaffe/outputCluster-skx-${nodes}.txt"
if [[ -f $caffelog ]];then
   log=$caffelog
fi
output=$(cat $log | grep -m 1 "\[0\].*output data size" | awk -F ':' '{ print $NF }' | awk -F',' '{ print $3 "," $4 "," $2 }')
ln=$(cat $log | grep -m 1 -n "\[0\].*output data size" | awk -F ':' '{ print $1 }')
echo "$PREFIX $(timestamp) ($(basename $log):$ln) model_hp_initial_shape: [$output]" >> $mlperf_log
echo "$nodes $ppn $batch_size" 
iter_in1epoch=$((1281167 / ($nodes * $ppn * $batch_size))) #1281167
ln=0
echo "1 epoch is $iter_in1epoch iterations"
epoch=0

timestamp2()
{
   t2=$1
   date_curr=$2
   date_curr=$(echo $date_curr |sed "s/I//g")
   st2=$(date --date="2018$date_curr $t2" +%s)
   echo $st2
   return $st2
}

if [ -e $log ];then

   while read line; do
   ln=$(($ln+1))

   fileln=$(echo $line |awk -F' ' '{ print $5 }'|sed "s/]//g")  
   t=$(echo $line |awk -F' ' '{ print $3 }')
   md=$(echo $line |awk -F' ' '{ print $2 }')

   #parse normal log on node 0
   reg="^\[0\].*Creating layer res[2-6][a-f]$" 
   if [[ $line =~ $reg ]];then
      echo "$PREFIX $(timestamp2 $t $md) ($(basename $log):$ln) model_hp_shorcut_add" >> $mlperf_log
   fi
 
   reg="\[0\].*Iteration\s[0-9]+,\sloss\s="
   if [[  $line =~ $reg ]];then
      iter=$(echo $line |grep "\[0\].*Iteration.*, loss =" |awk -F' ' '{ print $7 }'|awk -F',' '{ print $1 }')
      batch_size=$(cat $prototxt|grep -m 1 "batch_size"|awk -F':' '{ print $2 }')

      if [ "$iter" -ge $(( ( epoch + 1 ) * iter_in1epoch )) ];then
         echo "$PREFIX $(timestamp2 $t $md) ($(basename $log):$ln) train_epoch: $epoch" >> $mlperf_log
         epoch=$((epoch+1))
      fi
   fi

   reg="\[0\].*Iteration\s[0-9]+,\slr\s="
   if [[ $line =~ $reg ]];then
      lr=$(echo $line |grep "\[0\].*Iteration.*, lr =" |awk -F' ' '{ print $10 }')
   fi

   reg="\[0\].*Test\snet\soutput.*loss3/top-1"
   if [[ $line =~ $reg ]];then
      accuracy=$(echo $line | awk -F' ' '{ print $12 }')
      st=$(date --date="$t" +%s)
      echo "$PREFIX $(timestamp2 $t $md) ($fileln) eval_accuracy: {\"epoch\": $((epoch-1)), \"value\": $accuracy}" >> $mlperf_log

      if [ "$target" != "" ] && [ $(echo $accuracy'>'$target | bc -l) -eq 1 ];then
          echo "$PREFIX $(timestamp2 $t $md) (caffe.cpp:350) run_stop: {\"success\": true}" >> $mlperf_log
          echo "$PREFIX 0000000000 (empty:0) run_clear_caches" >> $mlperf_log
          echo "$PREFIX $(timestamp2 $t $md) (caffe.cpp:351) run_final" >> $mlperf_log
          echo "reached accuracy on iteration $iter"
          exit 0
      fi
   fi

   #only handle node 0 MLPerf log
   reg="\[0\].*MLPerf:"
   if [[ ! $line =~ $reg ]];then
      continue
   fi

   reg="eval_start"
   if [[ $line =~ $reg ]];then
      st=$(date --date="$t" +%s)
      echo "$PREFIX $(timestamp2 $t $md) ($fileln) opt_learning_rate: {\"value\": $lr}" >> $mlperf_log
   fi 
 
   reg="\[0\].*MLPerf:.*<.*>"
   if [[ $line =~ $reg ]];then
      st=$(date --date="$t" +%s)
      layer=$(echo $line |sed "s/\] MLPerf://g" |sed "s/.*MLPerf://g" |awk -F'<' '{ print $2 }' |awk -F'>' '{ print $1 }')  
  
      reg="MLPerf.*Top shape"
      reg2="MLPerf.*\"shape\""
      reg3="model_hp_conv2d_fixed_padding"
      if [[ $line =~ $reg ]];then
         top=$(echo $line |grep "\[0\].*MLPerf.*Top shape"|awk -F' ' '{ for(i=12;i<=13;i++) printf("%s, ", $i); print $14 }')  
         tag=$(echo $line |grep "\[0\].*MLPerf.*Top shape"|awk -F' ' '{ print $8}')  
         echo "$PREFIX $(timestamp2 $t $md) ($fileln) ${tag} \"($top)\"" >> $mlperf_log
         if [[ $line =~ "model_hp_conv2d_fixed_padding" ]];then
            if [ -z "$line_conv" ];then
               echo "unexpected convolution tag"
            fi
            echo $line_conv >> $mlperf_log
         fi
         reg_shortcut="res[2-5]a_branch1" 
         if [[ $line =~ $reg_shortcut ]];then
            echo "$PREFIX $(timestamp2 $t $md) ($fileln) model_hp_projection_shortcut: \"($top)\"" >> $mlperf_log 
         fi
      elif [[ $line =~ $reg2 ]];then 
          line_bn=$(echo $line|sed "s/($batch_size /(/" | sed "s/ (.*))/)/") 
          tmp=$(timestamp2 $t $md)
          echo $line |grep "\[0\].*MLPerf"|sed "s/\] MLPerf://g" |awk -F' ' -v p="$PREFIX" -v d="$tmp"  'END{ printf("%s %s (%s) ", p, d, $5); for(i=7;i<=NF;i++) printf("%s%s",$i,i==NF?RS:OFS); }'   >> $mlperf_log
      elif [[ $line =~ $reg3 ]];then
          tmp=$(timestamp2 $t $md)
          line_conv=$(echo $line |grep "\[0\].*MLPerf"|sed "s/\] MLPerf://g" |awk -F' ' -v p="$PREFIX" -v d="$tmp"  'END{ printf("%s %s (%s) ", p, d, $5); for(i=7;i<=NF;i++) printf("%s%s",$i,i==NF?RS:OFS); }')
      else
           tmp=$(timestamp2 $t $md)
           echo $line |grep "\[0\].*MLPerf"|sed "s/\] MLPerf://g" |awk -F' ' -v p="$PREFIX" -v d="$tmp" 'END{ printf("%s %s (%s) ", p, d, $5); for(i=7;i<=NF;i++) printf("%s%s",$i,i==NF?RS:OFS); }'   >> $mlperf_log
      fi

      if [[ "$layer" =~ "pool1" ]];then

         echo "$PREFIX 0000000000 (empty:0) model_hp_block_type: \"None\"" >> $mlperf_log
         echo "$PREFIX $(timestamp2 $t $md) ($fileln) model_hp_begin_block: {\"block_type\": \"bottleneck_block\"}" >> $mlperf_log
         echo "$PREFIX $(timestamp2 $t $md) ($fileln) model_hp_resnet_topology: \" Block Input: ($top)\"" >> $mlperf_log
      fi

      reg_res2="res2[a-c]_relu"
      reg_res3="res3[a-d]_relu"
      reg_res4="res4[a-f]_relu"
      reg_res5="res5[a-c]_relu"
      if [[ $layer =~ $reg_res2 || $layer =~ $reg_res3 || $layer =~ $reg_res4 || $layer =~ $reg_res5 ]];then
         echo "$PREFIX $(timestamp2 $t $md) ($fileln) model_hp_end_block: \" Block Output: ($top)\"" >> $mlperf_log
         reg_res5c="res5c_relu"
         if [[ ! $layer =~ $reg_res5c ]];then 
            echo "$PREFIX $(timestamp2 $t $md) ($fileln) model_hp_begin_block: {\"block_type\": \"bottleneck_block\"}" >> $mlperf_log
            echo "$PREFIX $(timestamp2 $t $md) ($fileln) model_hp_resnet_topology: \" Block Input: ($top)\"" >> $mlperf_log
         fi 
      fi
     
      reg_fc="fc1000" 
      if [[ $layer =~ $reg_fc ]];then
         output_num=$(echo $line | awk -F' ' '{ print $9 }')
         echo "$PREFIX $(timestamp2 $t $md) ($fileln) model_hp_final_shape: [$output_num]" >> $mlperf_log
      fi
   else
      reg_eval_target="eval_target"
      if [[ $line =~ $reg_eval_target ]];then
         tmp=$(timestamp2 $t $md)
         echo $line |grep "\[0\].*MLPerf:"|sed "s/\] MLPerf://g" |awk -F' ' -v p="$PREFIX" -v d="$tmp" 'END{ printf("%s %s (%s) ", p, d, $5); for(i=6;i<=NF;i++) printf("%s%s",$i,i==NF?RS:OFS); }' |sed "s/0.749/$target/"   >> $mlperf_log
      else 
         reg_eval_acc="eval_accuracy"
         if [[ ! $line =~ $reg_eval_acc ]];then
            echo $line
            tmp=$(timestamp2 $t $md)
            #echo $line |grep "\[0\].*MLPerf:"|sed "s/\] MLPerf://g" |awk -F' ' -v p="$PREFIX" -v d="$tmp" 'END{ printf("%s %s (%s) ", p, d, $5); for(i=6;i<=NF;i++) printf("%s%s",$i,i==NF?RS:OFS); }'   >> $mlperf_log
            echo $line |grep "\[0\].*MLPerf:"|sed "s/\] MLPerf://g" |awk -F' ' -v p="$PREFIX" -v d="$tmp" 'END{ printf("%s %s (%s) ", p, d, $5); for(i=6;i<=NF;i++) printf("%s%s",$i,i==NF?RS:OFS); }' 
            echo $line |grep "\[0\].*MLPerf:"|sed "s/\] MLPerf://g" |awk -F' ' -v p="$PREFIX" -v d="$tmp" 'END{ printf("%s %s (%s) ", p, d, $5); for(i=6;i<=NF;i++) printf("%s%s",$i,i==NF?RS:OFS); }'   >> $mlperf_log
         fi
      fi
   fi

   done <$log
 fi

cp $mlperf_log run_intelcaffe_${TESTNAME}/mlperf_intelcaffe_${index}.txt
