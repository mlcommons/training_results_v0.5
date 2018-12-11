import numpy as np
from mlperf_compliance import mlperf_log

class MPIWrapper(object):
    def __init__(self):
        self.comm = None
        self.MPI = None

    def get_comm(self):
        if self.comm is None:
            from mpi4py import MPI
            self.comm = MPI.COMM_WORLD
            self.MPI = MPI

        return self.comm

    def barrier(self):
        c = self.get_comm()
        c.Barrier()

    def allreduce(self, x):
        c = self.get_comm()
        rank = c.Get_rank()
        val = np.array(x, dtype=np.int32)
        result = np.zeros_like(val, dtype=np.int32)
        c.Allreduce([val, self.MPI.INT], [result, self.MPI.INT]) #, op=self.MPI.SUM)
        return result

    def rank(self):
        c = self.get_comm()
        return c.Get_rank()

mpiwrapper=MPIWrapper()

def all_reduce(v):
    return mpiwrapper.allreduce(v)


def mx_resnet_print(key, val=None, sync=False, uniq=True, stack_offset=1, deferred=False):
    rank = mpiwrapper.rank()
    if sync:
        mpiwrapper.barrier()

    if uniq and (rank != 0):
        return

    mlperf_log.resnet_print(key=key, value=val,
            stack_offset=stack_offset,
            deferred=deferred)
    return


def resnet_max_pool_log(input_shape, stride):
    downsample = 2 if stride == 2 else 1
    output_shape = (input_shape[0], 
                    input_shape[1]/downsample, 
                    input_shape[2]/downsample)

    mx_resnet_print(
            key=mlperf_log.MODEL_HP_INITIAL_MAX_POOL,
            val="{} -> {}".format(input_shape, output_shape))

    return output_shape


def resnet_begin_block_log(input_shape, block_type):
    mx_resnet_print(
            key=mlperf_log.MODEL_HP_BEGIN_BLOCK,
            val={"block_type": block_type})

    mx_resnet_print(
            key=mlperf_log.MODEL_HP_BLOCK_TYPE,
            val=block_type)

    mx_resnet_print(
            key=mlperf_log.MODEL_HP_RESNET_TOPOLOGY,
            val=" Block Input: {}".format(input_shape))

    return input_shape


def resnet_end_block_log(input_shape):
    mx_resnet_print(
            key=mlperf_log.MODEL_HP_END_BLOCK,
            val=" Block Output: {}".format(input_shape))
    return input_shape


def resnet_projection_log(input_shape, output_shape):
    mx_resnet_print(
            key=mlperf_log.MODEL_HP_PROJECTION_SHORTCUT,
            val="{} -> {}".format(input_shape, output_shape))

    return output_shape


def resnet_conv2d_log(input_shape, stride, out_channels, initializer, bias):
    downsample = 2 if (stride == 2 or stride == (2, 2)) else 1
    output_shape = (out_channels, 
                    input_shape[1]/downsample, 
                    input_shape[2]/downsample)

    mx_resnet_print(
            key=mlperf_log.MODEL_HP_CONV2D_FIXED_PADDING,
            val="{} -> {}".format(input_shape, output_shape))

    mx_resnet_print(
            key=mlperf_log.MODEL_HP_CONV2D_FIXED_PADDING,
            val={
                "stride":stride, 
                "filters":out_channels, 
                "initializer":initializer,
                "use_bias":bias})
    return output_shape


def resnet_relu_log(input_shape):
    mx_resnet_print(key=mlperf_log.MODEL_HP_RELU)
    return input_shape


def resnet_dense_log(input_shape, out_features):
    shape = (out_features)
    mx_resnet_print(key=mlperf_log.MODEL_HP_DENSE, val=out_features)
    return shape


def resnet_batchnorm_log(shape, momentum, eps, center=True, scale=True, training=True):
    mx_resnet_print(
            key=mlperf_log.MODEL_HP_BATCH_NORM,  
            val={
                "shape":shape,
                "momentum":momentum, "epsilon":eps,
                "center":center, "scale":scale, "training":training})
    return shape
