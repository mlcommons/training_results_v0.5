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

package com.intel.analytics.bigdl.examples.mlperf.recommendation
import java.util
import java.util.concurrent.atomic.AtomicInteger
import com.intel.analytics.bigdl.dataset.{LocalDataSet, MiniBatch}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{Engine, RandomGenerator}
import com.intel.analytics.zoo.examples.mlperf.recommendation.NcfLogger
import scala.collection.parallel.ParSeq
import scala.util.Random

class NCFDataSet (trainSet: Seq[(Int, Set[Int])], trainNegatives: Int, batchSize: Int, userCount: Int,
    itemCount: Int, var seed: Int = 1, val processes: Int = 10) extends LocalDataSet[MiniBatch[Float]] {
  val trainSize = trainSet.map(_._2.size).sum
  val trainPositiveBuffer = new Array[Float](trainSize * 2)
  NCFDataSet.copy(trainSet, trainPositiveBuffer)
  val inputBuffer = new Array[Float](trainSize * (1 + trainNegatives) * 2 )
  val labelBuffer = new Array[Float](trainSize * (1 + trainNegatives))

  val generateTasks = NCFDataSet.generateTasks(trainSet, trainNegatives, processes, seed)

  override def shuffle(): Unit = {
    NcfLogger.info("input_hp_num_neg", trainNegatives)
    NCFDataSet.generateNegativeItems(
      trainSet,
      inputBuffer,
      generateTasks,
      trainNegatives,
      itemCount)
    System.arraycopy(trainPositiveBuffer, 0, inputBuffer,
      trainSize * trainNegatives * 2, trainSize * 2)
    util.Arrays.fill(labelBuffer, 0, trainSize * trainNegatives, 0)
    util.Arrays.fill(labelBuffer, trainSize * trainNegatives, trainSize * (1 + trainNegatives), 1)
    NCFDataSet.shuffle(inputBuffer, labelBuffer, seed, processes)
//    NCFDataSet.shuffle(inputBuffer, labelBuffer, seed)
    seed += processes
  }

  override def data(train: Boolean): Iterator[MiniBatch[Float]] = {
    new Iterator[MiniBatch[Float]] {
      val input = Tensor[Float](batchSize, 2)
      val label = Tensor[Float](batchSize, 1)
      val miniBatch = MiniBatch(Array(input), Array(label))

      private val index = new AtomicInteger()
      private val numOfSample = inputBuffer.length / 2
      private val numMiniBatch = math.ceil(numOfSample.toFloat / batchSize).toInt

      override def hasNext: Boolean = {
        index.get() < numMiniBatch
      }

      override def next(): MiniBatch[Float] = {
        val curIndex = index.getAndIncrement()  % numMiniBatch
        if (curIndex < numMiniBatch - 1) {
          System.arraycopy(inputBuffer, curIndex * 2 * batchSize,
            input.storage().array(), 0, batchSize * 2)
          System.arraycopy(labelBuffer, curIndex * batchSize,
            label.storage().array(), 0, batchSize)
          miniBatch
        } else if (curIndex == numMiniBatch - 1) {
          // left padding
          val leftPaddingOffset = numOfSample - batchSize
          System.arraycopy(inputBuffer, leftPaddingOffset * 2,
            input.storage().array(), 0, batchSize * 2)
          System.arraycopy(labelBuffer, leftPaddingOffset,
            label.storage().array(), 0, batchSize)
          miniBatch
        } else {
          null
        }
      }
    }
  }

  override def size(): Long = inputBuffer.length / 2
}

object NCFDataSet {

  def copy(trainSet: Seq[(Int, Set[Int])], trainBuffer: Array[Float]): Unit = {
    var i = 0
    var offset = 0
    while(i < trainSet.size) {
      val userId = trainSet(i)._1
      val itemIds = trainSet(i)._2.toIterator

      while(itemIds.hasNext) {
        val itemId = itemIds.next()
        trainBuffer(offset) = userId
        trainBuffer(offset + 1) = itemId

        offset += 2
      }

      i += 1
    }

  }

  def shuffle(inputBuffer: Array[Float],
              labelBuffer: Array[Float],
              seed: Int): Unit = {
    val rand = new Random(seed)
    var i = 0
    val length = inputBuffer.length / 2
    while (i < length) {
      val exchange = rand.nextInt(length - i) + i
      val tmp1 = inputBuffer(exchange * 2)
      val tmp2 = inputBuffer(exchange * 2 + 1)
      inputBuffer(exchange * 2) = inputBuffer(i * 2)
      inputBuffer(exchange * 2 + 1) = inputBuffer(i * 2 + 1)
      inputBuffer(2 * i) = tmp1
      inputBuffer(2 * i + 1) = tmp2

      val labelTmp = labelBuffer(exchange)
      labelBuffer(exchange) = labelBuffer(i)
      labelBuffer(i) = labelTmp
      i += 1
    }
  }

  /**
   * shuffle two times with multithread
   * @param inputBuffer
   * @param labelBuffer
   * @param seed
   * @param parallelism
   */
  def shuffle(inputBuffer: Array[Float],
              labelBuffer: Array[Float],
              seed: Int,
              parallelism: Int): Unit = {
    val length = inputBuffer.length / 2
    val extraSize = length % parallelism
    val taskSize = math.floor(length / parallelism).toInt

    val seeds = Array.tabulate(parallelism)(i =>{
      val rand = new Random(seed + i)
      val length = if (i < extraSize) taskSize + 1 else taskSize
      (i, length, rand)
    }).par
    // first shuffle
    seeds.foreach{v =>
      val offset = v._1
      val length = v._2
      val rand = v._3
      var i = 0
      while(i < length) {
        val ex = rand.nextInt(length) * parallelism + offset
        val current = i * parallelism + offset
        if (ex != current) {
          exchange(inputBuffer, labelBuffer,
            current, ex)
        }
        i += 1
      }
    }
    // second shuffle
    seeds.foreach{v =>
      val offset = v._1 * taskSize + {
        if (v._1 < extraSize) v._1 else extraSize
      }
      val length = v._2
      val rand = v._3
      var i = 0
      while (i < length) {
        val ex = rand.nextInt(length) + offset
        val current = i + offset
        if (ex != current) {
          exchange(inputBuffer, labelBuffer,
            current, ex)
        }
        i += 1
      }
    }
  }

  private def exchange(inputBuffer: Array[Float],
                       labelBuffer: Array[Float],
                       current: Int, exchange: Int): Unit = {
    val tmp1 = inputBuffer(exchange * 2)
    val tmp2 = inputBuffer(exchange * 2 + 1)
    inputBuffer(exchange * 2) = inputBuffer(current * 2)
    inputBuffer(exchange * 2 + 1) = inputBuffer(current * 2 + 1)
    inputBuffer(2 * current) = tmp1
    inputBuffer(2 * current + 1) = tmp2

    val labelTmp = labelBuffer(exchange)
    labelBuffer(exchange) = labelBuffer(current)
    labelBuffer(current) = labelTmp
  }

  def generateTasks(trainSet: Seq[(Int, Set[Int])],
                            trainNeg: Int,
                            processes: Int,
                            seed: Int): ParSeq[(Int, Int, Int, RandomGenerator)] = {
    val size = Math.ceil(trainSet.size / processes).toInt
    val lastOffset = size * (processes - 1)
    val processesOffset = Array.tabulate[Int](processes)(_ * size)

    val tasks = processesOffset.map{ offset =>
      val numberOfUser = if(offset == lastOffset) {
        trainSet.length - offset
      } else {
        size
      }
      var numItem = 0
      var i = 0
      while (i < numberOfUser) {
        numItem += trainSet(i + offset)._2.size
        i += 1
      }
      (numberOfUser, offset, numItem)
    }

    val numItemAndOffset = (0 until processes).map{p =>
      (tasks(p)._1, tasks(p)._2,
        tasks.slice(0, p).map(_._3).sum * trainNeg, new RandomGenerator().setSeed(p + seed))
    }.par

    numItemAndOffset
  }

  def generateNegativeItems(trainSet: Seq[(Int, Set[Int])],
                            buffer: Array[Float],
                            tasks: ParSeq[(Int, Int, Int, RandomGenerator)],
                            trainNeg: Int,
                            itemCount: Int): Unit = {

    tasks.foreach{ v =>
      val numberOfUser = v._1
      var userStart = v._2
      var bufferOffset = v._3
      val rand = v._4

      while (userStart < v._2 + numberOfUser) {
        val userId = trainSet(userStart)._1
        val items = trainSet(userStart)._2
        var i = 0
        while (i < items.size * trainNeg) {
          var negItem = Math.floor(rand.uniform(0, itemCount)).toInt + 1
          while (items.contains(negItem)) {
            negItem = Math.floor(rand.uniform(0, itemCount)).toInt + 1
          }
          val negItemOffset = bufferOffset * 2
          buffer(negItemOffset) = userId
          buffer(negItemOffset + 1) = negItem

          i += 1
          bufferOffset += 1
        }
        userStart += 1
      }
    }
  }
}
