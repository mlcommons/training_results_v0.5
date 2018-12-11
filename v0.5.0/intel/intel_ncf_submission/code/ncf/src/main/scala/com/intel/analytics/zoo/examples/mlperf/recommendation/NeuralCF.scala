/*
 * Copyright 2016 The BigDL Authors.
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

package com.intel.analytics.zoo.examples.mlperf.recommendation

import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

/**
 * The model is for neural collaborative filtering.
 *
 * @param numClasses   The number of classes. Positive integer.
 * @param userCount    The number of users. Positive integer.
 * @param itemCount    The number of items. Positive integer.
 * @param userEmbed    Units of user embedding. Positive integer.
 * @param itemEmbed    Units of item embedding. Positive integer.
 * @param hiddenLayers Units hidenLayers of MLP part. Array of positive integer.
 * @param includeMF    Include Matrix Factorization or not. Boolean.
 * @param mfEmbed      Units of matrix factorization embedding. Positive integer.
 * @tparam T Numeric type of parameter(e.g. weight, bias). Only support float/double now.
 */

class NeuralCF[T: ClassTag] private(val userCount: Int,
                                    val itemCount: Int,
                                    val numClasses: Int,
                                    val userEmbed: Int = 20,
                                    val itemEmbed: Int = 20,
                                    val hiddenLayers: Array[Int] = Array(40, 20, 10),
                                    val includeMF: Boolean = true,
                                    val mfEmbed: Int = 20
      )(implicit ev: TensorNumeric[T]) extends Container[Tensor[T], Tensor[T], T] {

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    output = ncfModel.forward(input).toTensor[T]
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    gradInput = ncfModel.updateGradInput(input, gradOutput).toTensor[T]
    gradInput
  }

  override def accGradParameters(input: Tensor[T], gradOutput: Tensor[T]): Unit = {
    ncfModel.accGradParameters(input, gradOutput)
  }

  override def backward(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    gradInput = ncfModel.backward(input, gradOutput).toTensor[T]
    gradInput
  }

  override def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = {
    (embeddingModel.parameters()._1 ++ ncfLayers.parameters()._1,
      embeddingModel.parameters()._2 ++ ncfLayers.parameters()._2)
  }

  var embeddingModel: Graph[T] = _
  var ncfLayers: Graph[T] = _
  var ncfModel: Sequential[T] = _

  def buildModel(): this.type = {
    val input = Identity().inputs()
    val squeeze = Squeeze(1).inputs(input) // delete batch during evaluate
    val userId = Select(2, 1).setName("userId").inputs(squeeze)
    val itemId = Select(2, 2).setName("itemId").inputs(squeeze)
    val mlpUserTable = LookupTable[T](userCount, userEmbed)
      .setName("mlpUserEmbedding")
      .setInitMethod(RandomNormal(0, 0.01))
      .inputs(userId)
    val mlpItemTable = LookupTable[T](itemCount, itemEmbed)
      .setName("mlpItemEmbedding")
      .setInitMethod(RandomNormal(0, 0.01))
      .inputs(itemId)

    val mfUserTable = LookupTable[T](userCount, mfEmbed)
      .setName("mfUserEmbedding")
      .setInitMethod(RandomNormal(0, 0.01))
      .inputs(userId)
    val mfItemTable = LookupTable[T](itemCount, mfEmbed)
      .setName("mfItemEmbedding")
      .setInitMethod(RandomNormal(0, 0.01))
      .inputs(itemId)
    embeddingModel = Graph(input, Array(mlpUserTable, mlpItemTable, mfUserTable, mfItemTable))
      .setName("embeddings")

    val mlpUser = Identity().inputs()
    val mlpItem = Identity().inputs()
    val mfUser = Identity().inputs()
    val mfItem = Identity().inputs()

    val mlpMerge = JoinTable(2, 2).inputs(mlpUser, mlpItem)
    val mfMerge = CMulTable().inputs(mfUser, mfItem)

    var linear = Linear[T](itemEmbed + userEmbed, hiddenLayers(0))
        .setName(s"fc${itemEmbed + userEmbed}->${hiddenLayers(0)}")
      .setInitMethod(Xavier)
      .inputs(mlpMerge)
    var relu = ReLU[T]().inputs(linear)
    for (i <- 1 to hiddenLayers.length - 1) {
      linear = Linear(hiddenLayers(i - 1), hiddenLayers(i))
        .setName(s"fc${hiddenLayers(i-1)}->${hiddenLayers(i)}")
        .setInitMethod(Xavier)
        .inputs(relu)
      relu = ReLU().inputs(linear)
    }

    val merge = JoinTable(2, 2).inputs(mfMerge, relu)
    val stdv = math.sqrt(3.toDouble / hiddenLayers.last)
    val finalLinear = Linear(mfEmbed + hiddenLayers.last, numClasses)
        .setName(s"fc${mfEmbed + hiddenLayers.last}->$numClasses")
      .setInitMethod(Xavier.setVarianceNormAverage(false))
      .inputs(merge)
    val sigmoid = Sigmoid().inputs(finalLinear)

    ncfLayers = Graph(Array(mlpUser, mlpItem, mfUser, mfItem), sigmoid)
      .setName("linears")

    ncfModel = Sequential[T]()

    ncfModel.add(embeddingModel).add(ncfLayers)

    modules.clear()
    modules.append(ncfModel)

    this
  }
}

object NeuralCF {

  def apply[@specialized(Float, Double) T: ClassTag]
  (userCount: Int,
   itemCount: Int,
   numClasses: Int,
   userEmbed: Int,
   itemEmbed: Int,
   hiddenLayers: Array[Int],
   includeMF: Boolean = true,
   mfEmbed: Int = 20
  )(implicit ev: TensorNumeric[T]): NeuralCF[T] = {
    new NeuralCF[T](
      userCount, itemCount, numClasses, userEmbed, itemEmbed, hiddenLayers, includeMF, mfEmbed)
      .buildModel()
  }

}

