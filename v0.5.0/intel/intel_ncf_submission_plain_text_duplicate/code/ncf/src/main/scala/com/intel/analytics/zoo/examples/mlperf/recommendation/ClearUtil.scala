/*
 * Copyright 2018 The BigDL Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.intel.analytics.bigdl.examples.recommendation

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.tensor.{QuantizedTensor, QuantizedType, Storage, Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

object ClearUtil {
  private def getAndClear[T: ClassTag](tensors: Array[Tensor[T]])
                                      (implicit ev: TensorNumeric[T]): Array[Tensor[T]] = {
    if (tensors.length != 0) {
      val newTensors = new Array[Tensor[T]](tensors.length)
      val isQuantized = tensors.exists(_.getTensorType == QuantizedType)
      val totalElements = tensors.map(_.nElement()).sum
      val (isCompacted, storage) = if (!isQuantized) {
        val storageArray = tensors(0).storage.array()
        if (tensors.map(_.storage().array().eq(storageArray)).reduce(_ & _) &&
          totalElements <= storageArray.length) {
          val storage = Storage(storageArray)
          (true, storage)
        } else {
          (false, Storage[T](totalElements))
        }
      } else {
        (false, null)
      }
      var i = 0
      // get weight and bias
      while (i < tensors.length) {
        if (tensors(i) != null) {
          val ithTensor = tensors(i)
          ithTensor.getTensorType match {
            case QuantizedType =>
              val quantTensor = ithTensor.asInstanceOf[QuantizedTensor[T]]
              newTensors(i) = QuantizedTensor[T](quantTensor.getStorage, quantTensor.maxOfRow,
                quantTensor.minOfRow, quantTensor.sumOfRow, quantTensor.size(), quantTensor.params)
            case _ =>
              newTensors(i) = if (isCompacted) {
                Tensor[T](storage, ithTensor.storageOffset(), ithTensor.size(), ithTensor.stride())
              } else {
                Tensor[T](storage, ithTensor.storageOffset(), ithTensor.size(), ithTensor.stride())
                  .copy(ithTensor)
              }
          }
          i += 1
        }
      }
      // clear parameters
      clearTensor(tensors)
      newTensors
    } else {
      // just return an empty array when parameters is empty.
      Array()
    }
  }

  private[bigdl] def getAndClearWeightBiasGrad[T: ClassTag]
  (parameters: (Array[Tensor[T]], Array[Tensor[T]]))(implicit ev: TensorNumeric[T])
  : (Array[Tensor[T]], Array[Tensor[T]]) = {
    (getAndClear(parameters._1), getAndClear(parameters._2))
  }

  private[bigdl] def putGradWeightBias[T: ClassTag](
                                                       broadcastGradWeightBias: Array[Tensor[T]],
                                                       localModel: Module[T])(implicit ev: TensorNumeric[T]): Unit = {
    val localWeightBias = localModel.parameters()._2
    var i = 0
    while (i < localWeightBias.length) {
      if (localWeightBias(i) != null) {
        localWeightBias(i).set(broadcastGradWeightBias(i))
      }
      i += 1
    }
  }

  def clearTensor[T: ClassTag](tensors: Array[Tensor[T]])
                              (implicit ev: TensorNumeric[T]): Unit = {
    var i = 0
    while (i < tensors.length) {
      if (tensors(i) != null) {
        if (tensors(i).getTensorType == QuantizedType) {
          tensors(i).toQuantizedTensor.release()
        }

        tensors(i).set()
      }
      i += 1
    }
  }
}
