package com.intel.analytics.zoo.models.recommendation

import com.intel.analytics.bigdl.examples.mlperf.recommendation.NCFDataSet
import com.intel.analytics.bigdl.nn.BCECriterion
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{Engine, RandomGenerator, T}
import com.intel.analytics.zoo.examples.mlperf.recommendation.{NcfLogger, NeuralCF, NeuralCFexample}
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

import scala.io.Source

class NcfDatasetSpec extends FlatSpec with Matchers with BeforeAndAfter {
  "dataset" should "generate right result" in {
    val trainSet = Array(
      (1, Set(2, 3)),
      (2, Set(3, 4)),
      (3, Set(4, 5)),
      (4, Set(5, 6)),
      (5, Set(6, 7)),
      (6, Set(7, 8)),
      (7, Set(8, 9)),
      (8, Set(9, 10)))

    val valPos = Map(
      1 -> 1,
      2 -> 2,
      3 -> 3,
      4 -> 4,
      5 -> 5,
      6 -> 6,
      7 -> 7,
      8 -> 8
    )

    val trainNegatives = 3
    val batchSize = 3
    val userCount = 8
    val itemCount = 10

    val ncfD = new NCFDataSet(trainSet,
      trainNegatives, batchSize, userCount, itemCount, processes = 3)

    RandomGenerator.RNG.setSeed(10)
    ncfD.shuffle()
    val ite = ncfD.data(true)
    var count = 0
    while(ite.hasNext) {
      val batch = ite.next()
      val input = batch.getInput().toTensor[Float]
      val target = batch.getTarget().toTensor[Float]
      (1 to input.size(1)).foreach{i =>
        val userID = input.valueAt(i, 1)
        if (target.valueAt(i, 1) == 1) {
          Some(input.valueAt(i, 2)) should contain oneOf (userID + 1, userID + 2)
        } else {
          Some(input.valueAt(i, 2)) should not contain oneOf (userID + 1, userID + 2)
        }
      }
      count += 1
    }
   count should be (22)

  }

  "dataset" should "generate right result 2" in {
    val trainSet = Array(
      (1, Set(2, 3)),
      (2, Set(3, 4)),
      (3, Set(4, 5)),
      (4, Set(5, 6)),
      (5, Set(6, 7)),
      (6, Set(7, 8)),
      (7, Set(8, 9)),
      (8, Set(9, 10)))

    val valPos = Map(
      1 -> 1,
      2 -> 2,
      3 -> 3,
      4 -> 4,
      5 -> 5,
      6 -> 6,
      7 -> 7,
      8 -> 8
    )

    val trainNegatives = 3
    val batchSize = 4
    val userCount = 8
    val itemCount = 10

    val ncfD = new NCFDataSet(trainSet,
      trainNegatives, batchSize, userCount, itemCount, processes = 3)

    RandomGenerator.RNG.setSeed(10)
    ncfD.shuffle()
    var ite = ncfD.data(true)
    while(ite.hasNext) {
      val batch = ite.next()
      val input = batch.getInput().toTensor[Float]
      val target = batch.getTarget().toTensor[Float]
      (1 to input.size(1)).foreach{i =>
        val userID = input.valueAt(i, 1)
        if (target.valueAt(i, 1) == 1) {
          Some(input.valueAt(i, 2)) should contain oneOf (userID + 1, userID + 2)
        } else {
          Some(input.valueAt(i, 2)) should not contain oneOf (userID + 1, userID + 2)
        }
      }
    }

    RandomGenerator.RNG.setSeed(12)
    ncfD.shuffle()
    ite = ncfD.data(true)
    while(ite.hasNext) {
      val batch = ite.next()
      val input = batch.getInput().toTensor[Float]
      val target = batch.getTarget().toTensor[Float]
      (1 to input.size(1)).foreach{i =>
        val userID = input.valueAt(i, 1)
        if (target.valueAt(i, 1) == 1) {
          Some(input.valueAt(i, 2)) should contain oneOf (userID + 1, userID + 2)
        } else {
          Some(input.valueAt(i, 2)) should not contain oneOf (userID + 1, userID + 2)
        }
      }
    }
  }

  "dataset" should "generate right result 3" in {
    val trainSet = Array(
      (1, Set(2, 3)),
      (2, Set(3, 4)),
      (3, Set(4, 5)),
      (4, Set(5, 6)),
      (5, Set(6, 7)),
      (6, Set(7, 8)),
      (7, Set(8, 9)),
      (8, Set(9, 10)))

    val valPos = Map(
      1 -> 1,
      2 -> 2,
      3 -> 3,
      4 -> 4,
      5 -> 5,
      6 -> 6,
      7 -> 7,
      8 -> 8
    )

    val trainNegatives = 3
    val batchSize = 4
    val userCount = 8
    val itemCount = 10

    val ncfD = new NCFDataSet(trainSet,
      trainNegatives, batchSize, userCount, itemCount, processes = 3)
    val userCounts = new Array[Int](8)

    RandomGenerator.RNG.setSeed(10)
    ncfD.shuffle()
    var ite = ncfD.data(true)
    while(ite.hasNext) {
      val batch = ite.next()
      val input = batch.getInput().toTensor[Float]
      val target = batch.getTarget().toTensor[Float]
      (1 to input.size(1)).foreach{i =>
        val userID = input.valueAt(i, 1)
        userCounts(userID.toInt - 1) += 1
        if (target.valueAt(i, 1) == 1) {
          Some(input.valueAt(i, 2)) should contain oneOf (userID + 1, userID + 2)
        } else {
          Some(input.valueAt(i, 2)) should not contain oneOf (userID + 1, userID + 2)
        }
      }
    }

    userCounts should be (Array.tabulate(8)(_ => 8))

    ite = ncfD.data(true)
    while(ite.hasNext) {
      val batch = ite.next()
      val input = batch.getInput().toTensor[Float]
      val target = batch.getTarget().toTensor[Float]
      (1 to input.size(1)).foreach{i =>
        val userID = input.valueAt(i, 1)
        userCounts(userID.toInt - 1) += 1
        if (target.valueAt(i, 1) == 1) {
          Some(input.valueAt(i, 2)) should contain oneOf (userID + 1, userID + 2)
        } else {
          Some(input.valueAt(i, 2)) should not contain oneOf (userID + 1, userID + 2)
        }
      }
    }

    userCounts should be (Array.tabulate(8)(_ => 16))
  }

  "dataset shuffled" should "have different result" in {
    val trainSet = Array(
      (1, Set(2, 3)),
      (2, Set(3, 4)),
      (3, Set(4, 5)),
      (4, Set(5, 6)),
      (5, Set(6, 7)),
      (6, Set(7, 8)),
      (7, Set(8, 9)),
      (8, Set(9, 10)))

    val valPos = Map(
      1 -> 1,
      2 -> 2,
      3 -> 3,
      4 -> 4,
      5 -> 5,
      6 -> 6,
      7 -> 7,
      8 -> 8
    )

    val trainNegatives = 3
    val batchSize = 4
    val userCount = 8
    val itemCount = 10

    val ncfD = new NCFDataSet(trainSet,
      trainNegatives, batchSize, userCount, itemCount, processes = 3)

    ncfD.shuffle()
    val tensor1 = Tensor(ncfD.inputBuffer.clone(), Array(64, 2))
    ncfD.shuffle()
    val tensor2 = Tensor(ncfD.inputBuffer.clone(), Array(64, 2))
    tensor1 should not be (tensor2)

  }

  "two datasets with same seed" should "generate the same data" in {
    val trainSet = Array(
      (1, Set(2, 3)),
      (2, Set(3, 4)),
      (3, Set(4, 5)),
      (4, Set(5, 6)),
      (5, Set(6, 7)),
      (6, Set(7, 8)),
      (7, Set(8, 9)),
      (8, Set(9, 10)))

    val valPos = Map(
      1 -> 1,
      2 -> 2,
      3 -> 3,
      4 -> 4,
      5 -> 5,
      6 -> 6,
      7 -> 7,
      8 -> 8
    )

    val trainNegatives = 3
    val batchSize = 4
    val userCount = 8
    val itemCount = 10

    val ncfD = new NCFDataSet(trainSet,
      trainNegatives, batchSize, userCount, itemCount, processes = 3)
    ncfD.shuffle()
    ncfD.shuffle()
    val tensor1 = Tensor(ncfD.inputBuffer, Array(64, 2))
    val ncfD2 = new NCFDataSet(trainSet,
      trainNegatives, batchSize, userCount, itemCount, processes = 3)
    ncfD2.shuffle()
    val tensor2 = Tensor(ncfD2.inputBuffer, Array(64, 2))
    tensor1 should not be (tensor2)
    ncfD2.shuffle()
    tensor1 should be (tensor2)
  }

  "dataset" should "run with ncfoptimizer" in {
    System.setProperty("bigdl.localMode", "true")
    Engine.init(1, 1, false)
    val trainSet = Array(
      (1, Set(2, 3)),
      (2, Set(3, 4)),
      (3, Set(4, 5)),
      (4, Set(5, 6)),
      (5, Set(6, 7)),
      (6, Set(7, 8)),
      (7, Set(8, 9)),
      (8, Set(9, 10)))

    val valPos = Map(
      1 -> 1,
      2 -> 2,
      3 -> 3,
      4 -> 4,
      5 -> 5,
      6 -> 6,
      7 -> 7,
      8 -> 8
    )

    val trainNegatives = 3
    val batchSize = 11
    val userCount = 8
    val itemCount = 10
    val numFactors = 8
    val learningRate = 1e-3

    val trainDataset = new NCFDataSet(trainSet,
      trainNegatives, batchSize, userCount, itemCount, processes = 3)
    trainDataset.shuffle()

    val hiddenLayers = Array(16, 16, 8, 4)

    val optimMethod = Map(
      "embeddings" -> new EmbeddingAdam2[Float](
        learningRate = learningRate,
        userCount = userCount,
        itemCount = itemCount,
        embedding1 = hiddenLayers(0) / 2,
        embedding2 = numFactors),
      "linears" -> new ParallelAdam[Float](
        learningRate = learningRate))

    val ncf = NeuralCF[Float](
      userCount = userCount,
      itemCount = itemCount,
      numClasses = 1,
      userEmbed = hiddenLayers(0) / 2,
      itemEmbed = hiddenLayers(0) / 2,
      hiddenLayers = hiddenLayers.slice(1, hiddenLayers.length),
      mfEmbed = numFactors)

    val optimizer = new NCFOptimizer2[Float](ncf,
      trainDataset, BCECriterion[Float]())

    optimizer
        .setEndWhen(Trigger.maxEpoch(1))
      .setOptimMethods(optimMethod)
      .optimize()

    trainDataset.shuffle()

    optimizer
      .setEndWhen(Trigger.maxEpoch(2))
      .optimize()

    trainDataset.shuffle()

    optimizer
      .setEndWhen(Trigger.maxEpoch(3))
      .optimize()

  }

  "dataset" should "run with ncfoptimizer and localOptimizer" in {
    System.setProperty("bigdl.localMode", "true")
    Engine.init(1, 1, false)
    val maxEpoch = 10
    val trainSet = Array(
      (1, Set(2, 3)),
      (2, Set(3, 4)),
      (3, Set(4, 5)),
      (4, Set(5, 6)),
      (5, Set(6, 7)),
      (6, Set(7, 8)),
      (7, Set(8, 9)),
      (8, Set(9, 10)))

    val trainNegatives = 3
    val batchSize = 16
    val userCount = 8
    val itemCount = 10
    val numFactors = 8
    val learningRate = 1e-3

    val trainDataset = new NCFDataSet(trainSet,
      trainNegatives, batchSize, userCount, itemCount, processes = 1, seed = 1)

    val hiddenLayers = Array(16, 16, 8, 4)

    val optimMethod = Map(
      "embeddings" -> new EmbeddingAdam2[Float](
        learningRate = learningRate,
        userCount = userCount,
        itemCount = itemCount,
        embedding1 = hiddenLayers(0) / 2,
        embedding2 = numFactors),
      "linears" -> new ParallelAdam[Float](
        learningRate = learningRate))

    val ncf = NeuralCF[Float](
      userCount = userCount,
      itemCount = itemCount,
      numClasses = 1,
      userEmbed = hiddenLayers(0) / 2,
      itemEmbed = hiddenLayers(0) / 2,
      hiddenLayers = hiddenLayers.slice(1, hiddenLayers.length),
      mfEmbed = numFactors)

    val optimizer = new NCFOptimizer2[Float](ncf.cloneModule(),
      trainDataset, BCECriterion[Float]())

    optimizer
      .setEndWhen(Trigger.maxEpoch(maxEpoch))
      .setOptimMethods(optimMethod)
      .optimize()

    val localDataset = new NCFDataSet(trainSet,
      trainNegatives, batchSize, userCount, itemCount, processes = 1, seed = 1)
    val localOptimizer = new LocalOptimizer[Float](ncf.cloneModule(),
      localDataset, BCECriterion[Float]())
    val localOptimMethod = new ParallelAdam[Float](
      learningRate = learningRate)
    localOptimizer
      .setEndWhen(Trigger.maxEpoch(maxEpoch))
      .setOptimMethod(localOptimMethod)
      .optimize()

  }

  "trigger" should "works fine" in {
    val trigger = NeuralCFexample.maxEpochAndHr(5, 0.45f)
    val state = T()
    state("epoch") = 1
    state("HitRatio@10") = 0.1f

    while(!trigger(state)) {
      state("epoch") = state[Int]("epoch") + 1
      state("HitRatio@10") = state[Float]("HitRatio@10") + 0.1f
    }

    state[Int]("epoch") should be (5)
    state[Float]("HitRatio@10") should be (0.5f)
  }

  "trigger" should "works fine 2" in {
    val trigger = NeuralCFexample.maxEpochAndHr(5, 0.9f)
    val state = T()
    state("epoch") = 1
    state("HitRatio@10") = 0.1f

    while(!trigger(state)) {
      state("epoch") = state[Int]("epoch") + 1
      state("HitRatio@10") = state[Float]("HitRatio@10") + 0.1f
    }

    state[Int]("epoch") should be (6)
    state[Float]("HitRatio@10") should be (0.6f)
  }

  "log" should "works fine" in {
    NcfLogger.info("123")

  }

//  "generate" should "works" in {
//    val posFile = "/home/xin/datasets/ncf/test-ratings.csv"
//    val trainFile = "/home/xin/datasets/ncf/0.txt"
//    val testPositives = Source.fromFile(posFile).getLines()
//      .map{line =>
//        val pos = line.split("\t")
//        val userId = pos(0).toInt
//        val posItem = pos(1).toInt
//        (userId, posItem)
//      }.toMap
//
//    var i = 0
//    val trainData = Source.fromFile(trainFile).getLines()
//      .foreach{line =>
//        val pos = line.split(",")
//        val userId = pos(0).toInt
//        val item = pos(1).toInt
//        val label = pos(2).toInt
//        if (label == 0 && testPositives(userId) == item) {
//          println(s"userId: $userId, itemId $item")
//          i += 1
//        }
//      }
//
//    println(i)
//
//  }
}
