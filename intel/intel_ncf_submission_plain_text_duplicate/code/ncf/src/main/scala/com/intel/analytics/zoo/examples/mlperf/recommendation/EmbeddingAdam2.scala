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
package com.intel.analytics.bigdl.optim

import com.intel.analytics.bigdl.mkl.MklDnn
import com.intel.analytics.bigdl.mkl.hardware.Affinity
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{Engine, Table}
import org.apache.log4j.Logger

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.math.{pow, sqrt}
import scala.reflect.ClassTag

class EmbeddingAdam2[@specialized(Float, Double) T: ClassTag](
  var learningRate: Double = 1e-3,
  var learningRateDecay: Double = 0.0,
  var beta1: Double = 0.9,
  var beta2: Double = 0.999,
  var eps: Double = 1e-8,
  val userCount: Int = 138493,
  val itemCount: Int = 26744,
  val embedding1: Int = 64,
  val embedding2: Int = 128,
  val parallelism: Option[Int] = None
)(implicit ev: TensorNumeric[T]) extends OptimMethod[T] {

  EmbeddingAdam2.initBetas(beta1, beta2)
  val modelParallelism = Engine.coreNumber() // model parallelism, average gradient
  val parallelNum = parallelism.getOrElse(Engine.coreNumber())

  val gradients: Array[Array[(Tensor[T], Tensor[T])]] = new Array[Array[(Tensor[T], Tensor[T])]](4)
  for(i <- 0 until 4) {
    gradients(i) = new Array[(Tensor[T], Tensor[T])](parallelNum)
  }

  val userTimestep = new Array[Int](userCount)
  val itemTimestep = new Array[Int](itemCount)

  val userTaskSize = userCount / parallelNum
  val extraUserTask = userCount % parallelNum
  val itemTaskSize = itemCount / parallelNum
  val extraItemTask = itemCount % parallelNum

  val times = new Array[Long](parallelNum)

  (0 until parallelNum).foreach{tid =>
    if (state.get[Tensor[T]](s"buffer1$tid").isEmpty) {
      val userLength = userTaskSize + (if (tid < extraUserTask) 1 else 0)
      val itemLength = itemTaskSize + (if (tid < extraItemTask) 1 else 0)
      state(s"buffer1$tid") = Tensor[T](itemLength * embedding1 * 3)
      state(s"buffer2$tid") = Tensor[T](userLength * embedding1 * 3)
      state(s"buffer3$tid") = Tensor[T](itemLength * embedding2 * 3)
      state(s"buffer4$tid") = Tensor[T](userLength * embedding2 * 3)
    }
  }

  /**
   * An implementation of Adam http://arxiv.org/pdf/1412.6980.pdf
   *
   * @param feval     a function that takes a single input (X), the point of a evaluation, and
   *                  returns f(X) and df/dX
   * @param parameter the initial point
   * @return the new x vector and the function list {fx}, evaluated before the update
   */
  override def optimize(feval: (Tensor[T]) => (T, Tensor[T]),
    parameter: Tensor[T]): (Tensor[T], Array[T]) = {
    MklDnn.isLoaded

    var timestep = state.getOrElse[Int]("evalCounter", 0)

    val clr = learningRate / (1 + timestep*learningRateDecay)

    timestep = timestep + 1

    val start = System.nanoTime()

    Engine.default.invokeAndWait((0 until parallelNum).map(tid => () => {
      Affinity.setAffinity()
      val start = System.nanoTime()
      var offset = 0
      EmbeddingAdam2.updateSparse(tid, itemTaskSize, extraItemTask, embedding1,
        state[Tensor[T]](s"buffer1$tid"), clr, beta1, beta2, eps, gradients(3),
        parameter, offset, itemTimestep, timestep, modelParallelism)
      offset += itemCount * embedding1
      EmbeddingAdam2.updateSparse(tid, userTaskSize, extraUserTask, embedding1,
        state[Tensor[T]](s"buffer2$tid"), clr, beta1, beta2, eps, gradients(2),
        parameter, offset, userTimestep, timestep, modelParallelism)
      offset += userCount * embedding1
      EmbeddingAdam2.updateSparse(tid, itemTaskSize, extraItemTask, embedding2,
        state[Tensor[T]](s"buffer3$tid"), clr, beta1, beta2, eps, gradients(1),
        parameter, offset, null, timestep, modelParallelism)
      offset += itemCount * embedding2
      EmbeddingAdam2.updateSparse(tid, userTaskSize, extraUserTask, embedding2,
        state[Tensor[T]](s"buffer4$tid"), clr, beta1, beta2, eps, gradients(0),
        parameter, offset, null, timestep, modelParallelism)

      times(tid) = (System.nanoTime() - start) / 1000000
    }))

    state("evalCounter") = timestep // A tmp tensor to hold the sqrt(v) + epsilon

    (parameter, null)
  }

  def updateWeight(indexes: Tensor[T], parameter: Tensor[T]): Unit = {
    // first column is user, the second column is item
    val indexData = indexes.storage().array()
    val indexOffset = indexes.storageOffset() - 1
    val indexLength = indexes.nElement()
    val userRecord = new mutable.HashSet[Int]()
    val itemRecord = new mutable.HashSet[Int]()
    var i = 0
    while(i < indexLength) {
      userRecord.add(ev.toType[Int](indexData(i + indexOffset)))
      itemRecord.add(ev.toType[Int](indexData(i + indexOffset + 1)))
      i += 2
    }

    val timestep = state.getOrElse[Int]("evalCounter", 0)
    val clr = learningRate / (1 + timestep*learningRateDecay)

    Engine.default.invokeAndWait((0 until parallelNum).map(tid => () => {
      Affinity.setAffinity()
      val start = System.nanoTime()
      var offset = 0
      EmbeddingAdam2.lazyUpdate(itemRecord, tid, itemTaskSize, extraItemTask, embedding1, parameter,
        state[Tensor[T]](s"buffer1$tid"), clr, beta1, beta2, eps, offset,
        itemTimestep, timestep, false)
      offset += itemCount * embedding1
      EmbeddingAdam2.lazyUpdate(userRecord, tid, userTaskSize, extraUserTask, embedding1, parameter,
        state[Tensor[T]](s"buffer2$tid"), clr, beta1, beta2, eps, offset,
        userTimestep, timestep, false)
      offset += userCount * embedding1
      EmbeddingAdam2.lazyUpdate(itemRecord, tid, itemTaskSize, extraItemTask, embedding2, parameter,
        state[Tensor[T]](s"buffer3$tid"), clr, beta1, beta2, eps, offset,
        itemTimestep, timestep, true)
      offset += itemCount * embedding2
      EmbeddingAdam2.lazyUpdate(userRecord, tid, userTaskSize, extraUserTask, embedding2, parameter,
        state[Tensor[T]](s"buffer4$tid"), clr, beta1, beta2, eps, offset,
        userTimestep, timestep, true)

      times(tid) = (System.nanoTime() - start) / 1000000
    }))
  }

  override def loadFromTable(config: Table): this.type = {
    this.learningRate = config.get[Double]("learningRate").getOrElse(this.learningRate)
    this.learningRateDecay = config.get[Double]("learningRateDecay")
      .getOrElse(this.learningRateDecay)
    this.beta1 = config.get[Double]("beta1").getOrElse(this.beta1)
    this.beta2 = config.get[Double]("beta2").getOrElse(this.beta2)
    this.eps = config.get[Double]("Epsilon").getOrElse(this.eps)
    this
  }

  override def clearHistory(): Unit = {
    state.delete("s")
    state.delete("r")
  }

  override def getLearningRate(): Double = this.learningRate
}

object EmbeddingAdam2 {
  val logger = Logger.getLogger(this.getClass)

  private[optim] def updateSparse[T: ClassTag](
      tid: Int,
      taskSize: Int,
      extraTask: Int,
      embedding: Int,
      buffer: Tensor[T],
      clr: Double,
      beta1: Double,
      beta2: Double,
      eps: Double,
      gradient: Array[(Tensor[T], Tensor[T])],
      parameter: Tensor[T],
      parameterOffset: Int,
      timestamps: Array[Int],
      timestamp: Int,
      modelParallelism: Int
  )(implicit ev: TensorNumeric[T]): Unit = {
    val idStart = tid * taskSize + math.min(tid, extraTask)
    val idLength = taskSize + (if (tid < extraTask) 1 else 0)
    val offset = idStart * embedding
    val length = idLength * embedding
    val _s = buffer.narrow(1, 1, length)
    val _r = buffer.narrow(1, length + 1, length)
    val _denom = buffer.narrow(1, 2 * length + 1, embedding)

    val record = new ArrayBuffer[(Int, Int, Int, Int)]()
    var i = 0
    while(i < gradient.length) {
      val indexes = gradient(i)._1
      val values = gradient(i)._2
      val indexData = indexes.storage().array()
      val indexOffset = indexes.storageOffset() - 1
      var j = 0
      while(j < indexes.size(1)) {
        val ind = ev.toType[Int](indexData(indexOffset + j)) - 1
        if (ind >= idStart && ind < idLength + idStart) {
          if (timestamps != null) {
            timestamps(ind) = timestamp
          }
          record.append((ind * embedding - offset, i, j, ind))
        }
        j += 1
      }
      i += 1
    }
    val recordArray = record.toArray.sortWith(_._1 < _._1)
    val dfdx = Tensor[T](embedding)
    i = 0
    while(i < recordArray.length) {
      val values = gradient(recordArray(i)._2)._2
      dfdx.add(values.select(1, recordArray(i)._3 + 1))
      if (i == recordArray.length - 1 || recordArray(i)._4 != recordArray(i + 1)._4) {
        dfdx.div(ev.fromType(modelParallelism))
        val curS = _s.narrow(1, recordArray(i)._1 + 1, embedding).mul(ev.fromType[Double](beta1))
          .add(ev.fromType[Double](1 - beta1), dfdx)
        _denom.cmul(dfdx, dfdx)
        val curR = _r.narrow(1, recordArray(i)._1 + 1, embedding).mul(ev.fromType[Double](beta2))
          .add(ev.fromType[Double](1 - beta2), _denom)
        _denom.sqrt(curR)
        _denom.add(ev.fromType(eps))
        val biasCorrection1 = 1 - pow(beta1, timestamp)
        val biasCorrection2 = 1 - pow(beta2, timestamp)
        val stepSize = clr * sqrt(biasCorrection2) / biasCorrection1
        _denom.cdiv(curS, _denom)
        parameter.narrow(1, parameterOffset + recordArray(i)._4 * embedding + 1, embedding)
          .add(ev.fromType[Double](-stepSize), _denom)
        dfdx.zero()
      }
      i += 1
    }
  }

  private[optim] def lazyUpdate[T: ClassTag](
    record: mutable.HashSet[Int],
    tid: Int,
    taskSize: Int,
    extraTask: Int,
    embedding: Int,
    parameter: Tensor[T],
    buffer: Tensor[T],
    clr: Double,
    beta1: Double,
    beta2: Double,
    eps: Double,
    parameterOffset: Int,
    timestamps: Array[Int],
    timestamp: Int,
    updateTimeStamp: Boolean)(implicit ev: TensorNumeric[T]): Unit = {

    val idStart = (tid * taskSize + math.min(tid, extraTask))
    val idLength = (taskSize + (if (tid < extraTask) 1 else 0))

    val iter = record.iterator
    while(iter.hasNext) {
      val id = iter.next() - 1
      if (id >= idStart && id < idStart + idLength) {
        val lastTimestamp = timestamps(id)
        val t = timestamp - lastTimestamp
        require(t >= 0, s"t is $t")
        if (t > 0 && lastTimestamp != 0) {
          val currentParameter = parameter.narrow(1, parameterOffset + id * embedding + 1, embedding)
          val _s = buffer.narrow(1, (id - idStart) * embedding + 1, embedding)
          val _r = buffer.narrow(1, (idLength + id - idStart) * embedding + 1, embedding)
          val _denom = buffer.narrow(1, (2 * idLength + id - idStart) * embedding + 1, embedding)
          _denom.sqrt(_r)
          _denom.add(ev.fromType(eps))
          _denom.cdiv(_s, _denom)

          var i = lastTimestamp + 1
          var stepSizeSum = 0.0
          while(i <= timestamp) {
            val biasCorrection1 = 1 - pow1N(i)
            val biasCorrection2 = 1 - pow2N(i)
            val stepSize = clr * sqrt(biasCorrection2) / biasCorrection1
            val b1 = pow1N(i - lastTimestamp)
            val b2 = pow2N(i - lastTimestamp)
            stepSizeSum += stepSize * b1 / sqrt(b2)
            i += 1
          }
          currentParameter.add(ev.fromType[Double](-stepSizeSum), _denom)

          val beta1t = pow1N(timestamp - lastTimestamp)
          val beta2t = pow2N(timestamp - lastTimestamp)
          _s.mul(ev.fromType(beta1t))
          _r.mul(ev.fromType(beta2t))
        }
        if (timestamps != null && updateTimeStamp) {
          timestamps(id) = timestamp
        }
      }
    }
  }

  @inline
  private def pow1N(n: Int): Double = {
    require(n > 0 && n < cap)
    beta1Powers(n)
  }

  @inline
  private def pow2N(n: Int): Double = {
    require(n < cap)
    beta2Powers(n)
  }

  // TODO: hold by each embedding
  val cap = 1000000
  val beta1Powers = new Array[Double](cap)
  beta1Powers(0) = 1
  val beta2Powers = new Array[Double](cap)
  beta2Powers(0) = 1
  def initBetas(beta1: Double, beta2: Double): Unit = {
    logger.info("init power start")
    var i = 1
    while(i < cap) {
      beta1Powers(i) = beta1Powers(i - 1) * beta1
      beta2Powers(i) = beta2Powers(i - 1) * beta2
      i += 1
    }
    logger.info("init power done")
  }
}
