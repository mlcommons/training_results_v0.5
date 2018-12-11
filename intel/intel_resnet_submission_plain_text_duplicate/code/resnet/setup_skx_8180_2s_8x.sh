Build_IntelCaffe ()
{
echo "Build intelcaffe ..."
git clone https://github.com/intel/caffe.git intelcaffe
cd intelcaffe
echo "patch the code for mlperf log"
patch -p1 < ../mlperf2018.patch
sh ./scripts/prepare_env.sh --compiler gcc
sh ./scripts/build_intelcaffe.sh --compiler gcc
}

Build_IntelCaffe

echo "Next step: "
echo "1. Please allocate 8 skx-8180 nodes with 8 sockets on each nodes" 
echo "2. Please refer to the README section 2 to setup No password access between nodes."
echo "3. Please refer to the README section 5.a to create the hostfile"


