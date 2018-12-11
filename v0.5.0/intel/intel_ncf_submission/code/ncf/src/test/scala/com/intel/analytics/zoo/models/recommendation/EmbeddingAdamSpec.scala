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
package com.intel.analytics.zoo.models.recommendation

import com.intel.analytics.bigdl.mkl.MklDnn
import com.intel.analytics.bigdl.optim.{Adam, EmbeddingAdam2}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{Engine, T}
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

class EmbeddingAdamSpec extends FlatSpec with Matchers with BeforeAndAfter {
  "adam result" should "be same for one update" in {
    System.setProperty("bigdl.localMode", "true")
    Engine.init
    val userCount = 10
    val itemCount = 5
    val embedding1 = 3
    val embedding2 = 4
    val testAdam = new EmbeddingAdam2[Float](userCount = userCount, itemCount = itemCount,
      embedding1 = embedding1, embedding2 = embedding2, parallelism = Some(1))
    val refAdam = new Adam[Float]()
    val length = itemCount * embedding1 + userCount * embedding1 + itemCount * embedding2 +
      userCount * embedding2
    val weight1 = Tensor[Float](length).fill(1.0f)
    val weight2 = weight1.clone()

    MklDnn.isLoaded
    testAdam.updateWeight(Tensor[Float](T(1.0, 1.0)), weight1)
    testAdam.gradients(3)(0) = (Tensor[Float](T(1.0)), Tensor[Float](T(T(1.0, 2.0, 3.0))))
    testAdam.gradients(2)(0) = (Tensor[Float](T(2.0)), Tensor[Float](T(T(1.0, 2.0, 3.0))))
    testAdam.gradients(1)(0) = (Tensor[Float](T(1.0)), Tensor[Float](T(T(1.0, 1.0, 1.0, 1.0))))
    testAdam.gradients(0)(0) = (Tensor[Float](T(1.0)), Tensor[Float](T(T(1.0, 1.0, 1.0, 1.0))))
    testAdam.optimize(null, weight1)

    var offset = 0
    val denseGradient = weight1.clone().zero()
    denseGradient.setValue(offset + 1, 1.0f)
    denseGradient.setValue(offset + 2, 2.0f)
    denseGradient.setValue(offset + 3, 3.0f)
    offset += itemCount * embedding1
    denseGradient.setValue(offset + 4, 1.0f)
    denseGradient.setValue(offset + 5, 2.0f)
    denseGradient.setValue(offset + 6, 3.0f)
    offset += userCount * embedding1
    denseGradient.narrow(1, offset + 1, 4).fill(1.0f)
    offset += itemCount * embedding2
    denseGradient.narrow(1, offset + 1, 4).fill(1.0f)

    refAdam.optimize(_ => (1.0f, denseGradient), weight2)
    weight1 should be (weight2)
  }

  "adam result" should "be same for one update with multiple ids" in {
    System.setProperty("bigdl.localMode", "true")
    Engine.init
    val userCount = 10
    val itemCount = 5
    val embedding1 = 3
    val embedding2 = 4
    val testAdam = new EmbeddingAdam2[Float](userCount = userCount, itemCount = itemCount,
      embedding1 = embedding1, embedding2 = embedding2, parallelism = Some(1))
    val refAdam = new Adam[Float]()
    val length = itemCount * embedding1 + userCount * embedding1 + itemCount * embedding2 +
      userCount * embedding2
    val weight1 = Tensor[Float](length).fill(1.0f)
    val weight2 = weight1.clone()
    testAdam.gradients(3)(0) = (Tensor[Float](T(1.0, 1.0)),
      Tensor[Float](T(T(1.0, 2.0, 3.0), T(2.0, 3.0, 4.0))))
    testAdam.gradients(2)(0) = (Tensor[Float](T(1.0, 2.0)),
      Tensor[Float](T(T(1.0, 2.0, 3.0), T(2.0, 3.0, 4.0))))
    testAdam.gradients(1)(0) = (Tensor[Float](T(1.0)), Tensor[Float](T(T(1.0, 1.0, 1.0, 1.0))))
    testAdam.gradients(0)(0) = (Tensor[Float](T(1.0)), Tensor[Float](T(T(1.0, 1.0, 1.0, 1.0))))
    MklDnn.isLoaded
    testAdam.updateWeight(Tensor[Float](T(1.0, 1.0)), weight1)
    testAdam.optimize(null, weight1)

    var offset = 0
    val denseGradient = weight1.clone().zero()
    denseGradient.setValue(offset + 1, 1.5f)
    denseGradient.setValue(offset + 2, 2.5f)
    denseGradient.setValue(offset + 3, 3.5f)
    offset += itemCount * embedding1
    denseGradient.narrow(1, offset + 1, 3)
    denseGradient.setValue(offset + 1, 1.0f)
    denseGradient.setValue(offset + 2, 2.0f)
    denseGradient.setValue(offset + 3, 3.0f)
    denseGradient.setValue(offset + 4, 2.0f)
    denseGradient.setValue(offset + 5, 3.0f)
    denseGradient.setValue(offset + 6, 4.0f)
    offset += userCount * embedding1
    denseGradient.narrow(1, offset + 1, 4).fill(1.0f)
    offset += itemCount * embedding2
    denseGradient.narrow(1, offset + 1, 4).fill(1.0f)

    refAdam.optimize(_ => (1.0f, denseGradient), weight2)
    weight1 should be(weight2)
  }

  "adam result" should "be same for two update" in {
    System.setProperty("bigdl.localMode", "true")
    Engine.init
    val userCount = 10
    val itemCount = 5
    val embedding1 = 3
    val embedding2 = 4
    val testAdam = new EmbeddingAdam2[Float](userCount = userCount, itemCount = itemCount,
      embedding1 = embedding1, embedding2 = embedding2, parallelism = Some(1))
    val refAdam = new Adam[Float]()
    val length = itemCount * embedding1 + userCount * embedding1 + itemCount * embedding2 +
      userCount * embedding2
    val weight1 = Tensor[Float](length).fill(1.0f)
    val weight2 = weight1.clone()
    testAdam.gradients(3)(0) = (Tensor[Float](T(1.0)), Tensor[Float](T(T(1.0, 1.0, 1.0))))
    testAdam.gradients(2)(0) = (Tensor[Float](T(1.0)), Tensor[Float](T(T(1.0, 1.0, 1.0))))
    testAdam.gradients(1)(0) = (Tensor[Float](T(1.0)), Tensor[Float](T(T(1.0, 1.0, 1.0, 1.0))))
    testAdam.gradients(0)(0) = (Tensor[Float](T(1.0)), Tensor[Float](T(T(1.0, 1.0, 1.0, 1.0))))

    MklDnn.isLoaded
    testAdam.updateWeight(Tensor[Float](T(1.0, 1.0)), weight1)
    testAdam.optimize(null, weight1)


    testAdam.updateWeight(Tensor[Float](T(2.0, 2.0)), weight1)
    testAdam.gradients(3)(0) = (Tensor[Float](T(2.0)), Tensor[Float](T(T(1.0, 1.0, 1.0))))
    testAdam.gradients(2)(0) = (Tensor[Float](T(2.0)), Tensor[Float](T(T(1.0, 1.0, 1.0))))
    testAdam.gradients(1)(0) = (Tensor[Float](T(2.0)), Tensor[Float](T(T(1.0, 1.0, 1.0, 1.0))))
    testAdam.gradients(0)(0) = (Tensor[Float](T(2.0)), Tensor[Float](T(T(1.0, 1.0, 1.0, 1.0))))
    testAdam.optimize(null, weight1)

    testAdam.updateWeight(Tensor[Float](T(1.0, 1.0)), weight1)

    var offset = 0
    val denseGradient = weight1.clone().zero()
    denseGradient.narrow(1, offset + 1, 3).fill(1.0f)
    offset += itemCount * embedding1
    denseGradient.narrow(1, offset + 1, 3).fill(1.0f)
    offset += userCount * embedding1
    denseGradient.narrow(1, offset + 1, 4).fill(1.0f)
    offset += itemCount * embedding2
    denseGradient.narrow(1, offset + 1, 4).fill(1.0f)
    refAdam.optimize(_ => (1.0f, denseGradient), weight2)

    offset = 0
    denseGradient.zero()
    denseGradient.narrow(1, offset + 3 + 1, 3).fill(1.0f)
    offset += itemCount * embedding1
    denseGradient.narrow(1, offset + 3 + 1, 3).fill(1.0f)
    offset += userCount * embedding1
    denseGradient.narrow(1, offset + 4 + 1, 4).fill(1.0f)
    offset += itemCount * embedding2
    denseGradient.narrow(1, offset + 4 + 1, 4).fill(1.0f)
    refAdam.optimize(_ => (1.0f, denseGradient), weight2)


    weight1 should be(weight2)
  }

  "adam result" should "be same for three update" in {
    System.setProperty("bigdl.localMode", "true")
    Engine.init
    val userCount = 10
    val itemCount = 5
    val embedding1 = 3
    val embedding2 = 4
    val testAdam = new EmbeddingAdam2[Float](userCount = userCount, itemCount = itemCount,
      embedding1 = embedding1, embedding2 = embedding2, parallelism = Some(1))
    val refAdam = new Adam[Float]()
    val length = itemCount * embedding1 + userCount * embedding1 + itemCount * embedding2 +
      userCount * embedding2
    val weight1 = Tensor[Float](length).fill(1.0f)
    val weight2 = weight1.clone()
    MklDnn.isLoaded
    testAdam.updateWeight(Tensor[Float](T(1.0, 1.0)), weight1)
    testAdam.gradients(3)(0) = (Tensor[Float](T(1.0)), Tensor[Float](T(T(1.0, 1.0, 1.0))))
    testAdam.gradients(2)(0) = (Tensor[Float](T(1.0)), Tensor[Float](T(T(1.0, 1.0, 1.0))))
    testAdam.gradients(1)(0) = (Tensor[Float](T(1.0)), Tensor[Float](T(T(1.0, 1.0, 1.0, 1.0))))
    testAdam.gradients(0)(0) = (Tensor[Float](T(1.0)), Tensor[Float](T(T(1.0, 1.0, 1.0, 1.0))))
    testAdam.optimize(null, weight1)


    testAdam.updateWeight(Tensor[Float](T(2.0, 2.0)), weight1)
    testAdam.gradients(3)(0) = (Tensor[Float](T(2.0)), Tensor[Float](T(T(1.0, 1.0, 1.0))))
    testAdam.gradients(2)(0) = (Tensor[Float](T(2.0)), Tensor[Float](T(T(1.0, 1.0, 1.0))))
    testAdam.gradients(1)(0) = (Tensor[Float](T(2.0)), Tensor[Float](T(T(1.0, 1.0, 1.0, 1.0))))
    testAdam.gradients(0)(0) = (Tensor[Float](T(2.0)), Tensor[Float](T(T(1.0, 1.0, 1.0, 1.0))))
    testAdam.optimize(null, weight1)

    testAdam.updateWeight(Tensor[Float](T(2.0, 2.0)), weight1)
    testAdam.gradients(3)(0) = (Tensor[Float](T(2.0)), Tensor[Float](T(T(1.0, 1.0, 1.0))))
    testAdam.gradients(2)(0) = (Tensor[Float](T(2.0)), Tensor[Float](T(T(1.0, 1.0, 1.0))))
    testAdam.gradients(1)(0) = (Tensor[Float](T(2.0)), Tensor[Float](T(T(1.0, 1.0, 1.0, 1.0))))
    testAdam.gradients(0)(0) = (Tensor[Float](T(2.0)), Tensor[Float](T(T(1.0, 1.0, 1.0, 1.0))))
    testAdam.optimize(null, weight1)

    testAdam.updateWeight(Tensor[Float](T(1.0, 1.0)), weight1)

    var offset = 0
    val denseGradient = weight1.clone().zero()
    denseGradient.narrow(1, offset + 1, 3).fill(1.0f)
    offset += itemCount * embedding1
    denseGradient.narrow(1, offset + 1, 3).fill(1.0f)
    offset += userCount * embedding1
    denseGradient.narrow(1, offset + 1, 4).fill(1.0f)
    offset += itemCount * embedding2
    denseGradient.narrow(1, offset + 1, 4).fill(1.0f)
    refAdam.optimize(_ => (1.0f, denseGradient), weight2)

    offset = 0
    denseGradient.zero()
    denseGradient.narrow(1, offset + 3 + 1, 3).fill(1.0f)
    offset += itemCount * embedding1
    denseGradient.narrow(1, offset + 3 + 1, 3).fill(1.0f)
    offset += userCount * embedding1
    denseGradient.narrow(1, offset + 4 + 1, 4).fill(1.0f)
    offset += itemCount * embedding2
    denseGradient.narrow(1, offset + 4 + 1, 4).fill(1.0f)
    refAdam.optimize(_ => (1.0f, denseGradient), weight2)
    refAdam.optimize(_ => (1.0f, denseGradient), weight2)


    weight1 should be(weight2)
  }

}
