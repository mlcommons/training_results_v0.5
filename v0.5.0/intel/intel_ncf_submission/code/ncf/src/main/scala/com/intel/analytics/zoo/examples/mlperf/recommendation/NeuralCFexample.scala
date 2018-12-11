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

package com.intel.analytics.zoo.examples.mlperf.recommendation

import java.io.File
import java.util

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.dataset.{DataSet, MiniBatch, Sample, SampleToMiniBatch}
import com.intel.analytics.bigdl.examples.mlperf.recommendation.{GenerateData, NCFDataSet}
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.nn.{BCECriterion, ClassNLLCriterion}
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils._
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkConf
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import org.apache.spark.sql.functions._
import scopt.OptionParser

import scala.collection.JavaConverters._
import scala.collection.mutable.ArrayBuffer
import scala.collection.parallel.mutable.ParArray
import scala.io.Source
import scala.reflect.ClassTag
import scala.util.{Random, Sorting}

case class NeuralCFParams(val inputDir: String = "./data/ml-1m",
                          val dataset: String = "ml-1m",
                          val batchSize: Int = 256,
                          val nEpochs: Int = 2,
                          val learningRate: Double = 1e-3,
                          val learningRateDecay: Double = 0.0,
                          val trainNegtiveNum: Int = 4,
                          val valNegtiveNum: Int = 100,
                          val layers: String = "64,32,16,8",
                          val numFactors: Int = 8,
                          val seed: Int = 1,
                          val threshold: Float = 0.635f,
                          val beta1: Double = 0.9,
                          val beta2: Double = 0.999,
                          val eps: Double = 1e-8,
                          val lazyAdam: Boolean = false
                    )

case class Rating(userId: Int, itemId: Int, label: Int, timestamp: Int, train: Boolean)


object NeuralCFexample {

  def main(args: Array[String]): Unit = {
    NcfLogger.info("run_start")
    NcfLogger.info("run_clear_caches")
    val defaultParams = NeuralCFParams()
    // run with ml-20m, please use
    val parser = new OptionParser[NeuralCFParams]("NCF Example") {
      opt[String]("inputDir")
        .text(s"inputDir")
        .action((x, c) => c.copy(inputDir = x))
      opt[String]("dataset")
        .text(s"dataset, ml-20m or ml-1m, default is ml-1m")
        .action((x, c) => c.copy(dataset = x))
      opt[Int]('b', "batchSize")
        .text(s"batchSize")
        .action((x, c) => c.copy(batchSize = x))
      opt[Int]('e', "nEpochs")
        .text("epoch numbers")
        .action((x, c) => c.copy(nEpochs = x))
      opt[Double]('l', "lRate")
        .text("learning rate")
        .action((x, c) => c.copy(learningRate = x))
      opt[Double]("lrd")
        .text("learning rate decay")
        .action((x, c) => c.copy(learningRateDecay = x))
      opt[Int]("trainNeg")
        .text("The Number of negative instances to pair with a positive train instance.")
        .action((x, c) => c.copy(trainNegtiveNum = x))
      opt[Int]("valNeg")
        .text("The Number of negative instances to pair with a positive validation instance.")
        .action((x, c) => c.copy(valNegtiveNum = x))
      opt[String]("layers")
        .text("The sizes of hidden layers for MLP. Default is 64,32,16,8")
        .action((x, c) => c.copy(layers = x))
      opt[Int]("seed")
        .text("Random seed to generate data and model")
        .action((x, c) => c.copy(seed = x))
      opt[Double]("threshold")
        .text("End training when hit this threshold")
        .action((x, c) => c.copy(threshold = x.toFloat))
      opt[Double]("beta1")
        .text("coefficients used for computing running averages of gradient in adam")
        .action((x, c) => c.copy(beta1 = x))
      opt[Double]("beta2")
        .text("coefficients used for computing running averages of square gradient in adam")
        .action((x, c) => c.copy(beta2 = x))
      opt[Double]("eps")
        .text("eps in adam")
        .action((x, c) => c.copy(eps = x))
      opt[Int]("numFactors")
        .text("The Embedding size of MF model.")
        .action((x, c) => c.copy(numFactors = x))
      opt[Boolean]("useLazyAdam")
        .text("If use lazyAdam")
        .action((x, c) => c.copy(lazyAdam = x))
    }
   parser.parse(args, defaultParams).map {
      params =>
        run(params)
    } getOrElse {
      System.exit(1)
    }
  }

  def run(param: NeuralCFParams): Unit = {
    NcfLogger.info("seed", param.seed)
    Logger.getLogger("org").setLevel(Level.ERROR)
    Engine.init


    NcfLogger.info("opt_learning_rate", param.learningRate)
    NcfLogger.info("create_optim_method", Array(("name", """"Adam""""),
      ("lr", param.learningRate.toString),
      ("beta1", param.beta1.toString),
      ("beta2", param.beta2.toString),
      ("eps", param.eps.toString)))


    val optimMethod = Map(
      "embeddings" -> new EmbeddingAdam2[Float](
        learningRate = param.learningRate,
        learningRateDecay = param.learningRateDecay,
        beta1 = param.beta1,
        beta2 = param.beta2,
        eps = param.eps),
      "linears" -> new ParallelAdam[Float](
        learningRate = param.learningRate,
        learningRateDecay = param.learningRateDecay,
        beta1 = param.beta1,
        beta2 = param.beta2,
        Epsilon = param.eps))


    val validateBatchSize = optimMethod("linears").asInstanceOf[ParallelAdam[Float]].parallelNum

    val hiddenLayers = param.layers.split(",").map(_.toInt)

    val (ratings, userCount, itemCount, itemMapping) = loadPublicData(param.inputDir, param.dataset)




    NcfLogger.info("preproc_hp_num_eval", param.valNegtiveNum)
    NcfLogger.info("preproc_hp_sample_eval_replacement")
    // TODO: As reference pytorch code doesn't pass seed to generate test negative, we don't either.
    val (evalPos, trainSet, valSample) = GenerateData.generateTrainValSetLocal(ratings, itemCount,
        trainNegNum = param.trainNegtiveNum, valNegNum = param.valNegtiveNum)
    val trainDataset = new NCFDataSet(trainSet.sortBy(_._1),
      param.trainNegtiveNum, param.batchSize, userCount, itemCount,
      seed = param.seed, processes = validateBatchSize)
    val valDataset = (DataSet.array(valSample) ->
      SampleToMiniBatch[Float](validateBatchSize)).toLocal()





    RandomGenerator.RNG.setSeed(param.seed)
    NcfLogger.info("model_hp_mf_dim", param.numFactors)
    NcfLogger.info("model_hp_mlp_layer_sizes", s"[${hiddenLayers.mkString(", ")}]")
    val ncf = NeuralCF[Float](
      userCount = userCount,
      itemCount = itemCount,
      numClasses = 1,
      userEmbed = hiddenLayers(0) / 2,
      itemEmbed = hiddenLayers(0) / 2,
      hiddenLayers = hiddenLayers.slice(1, hiddenLayers.length),
      mfEmbed = param.numFactors)



    NcfLogger.info("model_hp_loss_fn", "binary_cross_entropy")
    val criterion = BCECriterion[Float]()

    val optimizer = new NCFOptimizer2[Float](ncf,
      trainDataset, criterion)

    optimizer
      .setOptimMethods(optimMethod)
      .setValidation(Trigger.everyEpoch, valDataset,
          Array(new HitRate[Float](negNum = param.valNegtiveNum)))
    if (param.lazyAdam) {
      NcfLogger.info("enable_lazy_adam")
      optimizer.enableLazyAdam()
    }
    val endTrigger = maxEpochAndHr(param.nEpochs, param.threshold)
    optimizer
      .setEndWhen(endTrigger)
      .optimize()

//    var e = 2
//    while(e <= param.nEpochs) {
//      println(s"Starting epoch $e/${param.nEpochs}")
//      val endTrigger = Trigger.maxEpoch(e)
//      val newTrainDataset = (DataSet.array[MiniBatch[Float]](
//        loadPytorchTrain(s"${e - 1}.txt", param.batchSize))).toLocal()
//
//      optimizer
//        .setTrainData(newTrainDataset)
//        .setEndWhen(endTrigger)
//        .optimize()
//
//      e += 1
//    }


    NcfLogger.info("run_final")
    System.exit(0)
  }




























































  def loadPytorchTest(posFile: String, negFile: String): Array[Sample[Float]] = {
    val startTime = System.currentTimeMillis()
    val testSet = new ArrayBuffer[Sample[Float]]()
    val positives = Source.fromFile(posFile).getLines()
    val negatives = Source.fromFile(negFile).getLines()
    while(positives.hasNext && negatives.hasNext) {
      val pos = positives.next().split("\t")
      val userId = pos(0).toFloat
      val posItem = pos(1).toFloat
      val neg = negatives.next().split("\t").map(_.toFloat)
      val distinctNegs = neg.distinct
      val testFeature = Tensor[Float](1 + neg.size, 2)
      testFeature.select(2, 1).fill(userId + 1)
      val testLabel = Tensor[Float](1 + neg.size).fill(0)
      var i = 1
      while(i <= distinctNegs.size) {
        testFeature.setValue(i, 2, distinctNegs(i - 1) + 1)
        i += 1
      }
      testFeature.setValue(i, 2, posItem + 1)
      testLabel.setValue(i, 1)
      testFeature.narrow(1, i + 1, neg.size - distinctNegs.size).fill(1)
      testLabel.narrow(1, i + 1, neg.size - distinctNegs.size).fill(-1)

      testSet.append(Sample(testFeature, testLabel))
    }
    println(s"load path: ${System.currentTimeMillis() - startTime}ms")
    testSet.toArray
  }


  def loadPytorchTrain(path: String, batchSize: Int = 2048): Array[MiniBatch[Float]] = {
    val startTime = System.currentTimeMillis()
    val lines = Source.fromFile(path).getLines()
    val miniBatches = new ArrayBuffer[MiniBatch[Float]]()
    while(lines.hasNext) {
      var i = 1
      val input = Tensor(batchSize, 2)
      val target  = Tensor(batchSize, 1)
      while(i <= batchSize && lines.hasNext) {
        val line = lines.next().split(",").map(_.toFloat)
        input.setValue(i, 1, line(0) + 1)
        input.setValue(i, 2, line(1) + 1)
        target.setValue(i, 1, line(2))
        i += 1
      }
      if (i <= batchSize) {
        input.narrow(1, i, batchSize + 1 - i).copy(
          miniBatches(0).getInput().toTensor.narrow(1, 1, batchSize + 1 - i))
        target.narrow(1, i, batchSize + 1 - i).copy(
          miniBatches(0).getTarget().toTensor.narrow(1, 1, batchSize + 1 - i))
        miniBatches.append(MiniBatch(input, target))
      } else {
        miniBatches.append(MiniBatch(input, target))
      }
    }
    println(s"load path: ${System.currentTimeMillis() - startTime}ms")
    miniBatches.toArray
  }

  def loadPublicData(sqlContext: SQLContext, dataPath: String,
                     dataset: String): (DataFrame, Int, Int, Map[Int, Int]) = {
    import sqlContext.implicits._
    val ratings = dataset match {
      case "ml-1m" =>
        loadMl1mData(sqlContext, dataPath)
      case "ml-20m" =>
        loadMl20mData(sqlContext, dataPath)
      case _ =>
        throw new IllegalArgumentException(s"Only support dataset ml-1m and ml-20m, but got ${dataset}")
    }

    val minMaxRow = ratings.agg(max("userId")).collect()(0)
    val userCount = minMaxRow.getInt(0)

    val uniqueMovie = ratings.rdd.map(_.getAs[Int]("itemId")).distinct().collect().sortWith(_ < _)
    val mapping = uniqueMovie.zip(1 to uniqueMovie.length).toMap

    val bcMovieMapping = sqlContext.sparkContext.broadcast(mapping)

    val mappingUdf = udf((itemId: Int) => {
     bcMovieMapping.value(itemId)
    })
    val mappedItemID = mappingUdf.apply(col("itemId"))
    val mappedRating = ratings//.drop(col("itemId"))
      .withColumn("itemId", mappedItemID)
    mappedRating.show()

    (mappedRating, userCount, uniqueMovie.length, mapping)
  }

  case class Row(userId: Int, itemId: Int, label: Float, timeStamp: Long)
  trait RowOrdering extends Ordering[Row] {
    def compare(x: Row, y: Row): Int = x.timeStamp compare y.timeStamp
  }
  implicit object Row extends RowOrdering

  def loadPublicData(dataPath: String, dataset: String): (Array[Row], Int, Int, Map[Int, Int]) = {
    val start = System.nanoTime()
    var userCount = 0

    val path = new File(dataPath, "/ratings.csv").getAbsolutePath
    val bufferedSource = scala.io.Source.fromFile(path)
    val rows = bufferedSource.getLines().drop(1).map { line =>
      val row = line.split(",").map(_.trim)
      val userId = row(0).toInt
      val itemId = row(1).toInt

      if (userId > userCount) userCount = userId

      Row(userId, itemId, row(2).toFloat, row(3).toLong)
    }.toArray.par
    bufferedSource.close

    val uniqueMovies = rows.toArray.map(_.itemId).distinct
    val length = uniqueMovies.length

    val mapping = uniqueMovies.zip(1 to uniqueMovies.length).toMap
    val parMapping = mapping.par
    val ratings = rows.map { row =>
      Row(row.userId, parMapping(row.itemId), row.label, row.timeStamp)
    }

    (ratings.seq.toArray, userCount, length, mapping)
  }

  def loadMl1mData(sqlContext: SQLContext, dataPath: String): DataFrame = {
    import sqlContext.implicits._
    sqlContext.read.text(dataPath + "/ratings.dat").as[String]
      .map(x => {
        val line = x.split("::")
        Rating(line(0).toInt, line(1).toInt, 1, line(3).toInt, true)
      }).toDF()
  }

  def loadMl20mData(sqlContext: SQLContext, dataPath: String): DataFrame = {
    val ratings = sqlContext.read
      .option("inferSchema", "true")
      .option("header", "true")
      .option("delimiter", ",")
      .csv(dataPath + "/ratings.csv")
      .toDF()
    println(ratings.schema)
    val result = ratings.withColumnRenamed("movieId", "itemId").withColumn("rating", lit(1))
      .withColumnRenamed("rating", "label").withColumn("train", lit(true))
    println(result.schema)
    result
  }

  def generateTrainValData(rating: DataFrame, userCount: Int, itemCount: Int,
                           trainNegNum: Int = 4, valNegNum: Int = 100): (DataFrame, DataFrame) = {
    val maxTimeStep = rating.groupBy("userId").max("timestamp").collect().map(r => (r.getInt(0), r.getInt(1))).toMap
    val bcT = rating.sparkSession.sparkContext.broadcast(maxTimeStep)
    val evalPos = rating.filter(r => bcT.value.apply(r.getInt(0)) == r.getInt(3)).dropDuplicates("userId")
      .collect().toSet
    val bcEval = rating.sparkSession.sparkContext.broadcast(evalPos)

    val negDataFrame = rating.sqlContext.createDataFrame(
      rating.rdd.groupBy(_.getAs[Int]("userId")).flatMap{v =>
        val userId = v._1
        val items = scala.collection.mutable.Set(v._2.map(_.getAs[Int]("itemId")).toArray: _*)
        val itemNumOfUser = items.size
        val gen = new Random(userId + System.currentTimeMillis())
        var i = 0
        val totalNegNum = trainNegNum * (itemNumOfUser - 1) + valNegNum

        val negs = new Array[Rating](totalNegNum)
        // gen negative sample to validation
        while(i < valNegNum) {
          val negItem = Random.nextInt(itemCount) + 1
          if (!items.contains(negItem)) {
            negs(i) = Rating(userId, negItem, 0, 0, false)
            i += 1
          }
        }

        // gen negative sample for train
        while(i < totalNegNum) {
          val negItem = gen.nextInt(itemCount) + 1
          if (!items.contains(negItem)) {
            negs(i) = Rating(userId, negItem, 0, 0, true)
            i += 1
          }
        }
        negs.toIterator
    })
//    println("neg train" + negDataFrame.filter(_.getAs[Boolean]("train")).count())
//    println("neg eval" + negDataFrame.filter(!_.getAs[Boolean]("train")).count())

    (negDataFrame.filter(_.getAs[Boolean]("train"))
      .union(rating.filter(r => !bcEval.value.contains(r))),
      negDataFrame.filter(!_.getAs[Boolean]("train"))
        .union(rating.filter(r => bcEval.value.contains(r))))

  }

  def maxEpochAndHr(maxEpoch: Int, maxHr: Float): Trigger = {
    new Trigger() {
      protected var runStop = false

      override def apply(state: Table): Boolean = {
        if (runStop) return runStop
        val hrEnd = if (state.contains("HitRatio@10")) {
          state[Float]("HitRatio@10") > maxHr
        } else {
          false
        }
        val epochEnd = state[Int]("epoch") > maxEpoch
        if (hrEnd || epochEnd) {
          // print this log only once
          NcfLogger.info("eval_target", maxHr)
          if (hrEnd) {
            NcfLogger.info("run_stop", Array(("success", "true")))
          } else {
            NcfLogger.info("run_stop", Array(("success", "false")))
          }
          runStop = true
        }
        hrEnd || epochEnd
      }
    }
  }

}

class HitRate[T: ClassTag](k: Int = 10, negNum: Int = 100)(
    implicit ev: TensorNumeric[T])
  extends ValidationMethod[T] {
  override def apply(output: Activity, target: Activity):
  ValidationResult = {
    val o = output.toTensor[T].resize(1 + negNum)
    val t = target.toTensor[T].resize(1 + negNum)
    var exceptedTarget = 0
    var i = 1
    while(i <= t.nElement()) {
      if (t.valueAt(i) == 1) {
        exceptedTarget = i
      }
      i += 1
    }
    require(exceptedTarget != 0, s"No positive sample")

    val hr = hitRate(exceptedTarget,
      o.narrow(1, 1, exceptedTarget), k)

    new ContiguousResult(hr, 1, s"HitRatio@$k")
  }

  def hitRate(index: Int, o: Tensor[T], k: Int): Float = {
    var topK = 1
    var i = 1
    val precision = ev.toType[Float](o.valueAt(index))
    while (i < o.nElement() && topK <= k) {
      if (ev.toType[Float](o.valueAt(i)) > precision) {
        topK += 1
      }
      i += 1
    }

    if(topK <= k) {
      1
    } else {
      0
    }
  }

  override def format(): String = s"HitRatio@$k"
}

class Ndcg[T: ClassTag](k: Int = 10, negNum: Int = 100)(
    implicit ev: TensorNumeric[T])
  extends ValidationMethod[T] {
  override def apply(output: Activity, target: Activity):
  ValidationResult = {
    val o = output.toTensor[T].resize(1 + negNum)
    val t = target.toTensor[T].resize(1 + negNum)
    var exceptedTarget = 0
    var i = 1
    while(i <= t.nElement()) {
      if (t.valueAt(i) == 1) {
        exceptedTarget = i
      }
      i += 1
    }
    require(exceptedTarget != 0, s"No positive sample")

    val n = ndcg(exceptedTarget, o.narrow(1, 1, exceptedTarget), k)

    new ContiguousResult(n, 1, s"NDCG")
  }

  def ndcg(index: Int, o: Tensor[T], k: Int): Float = {
    var ranking = 1
    var i = 1
    val precision = ev.toType[Float](o.valueAt(index))
    while (i < o.nElement() && ranking <= k) {
      if (ev.toType[Float](o.valueAt(i)) > precision) {
        ranking += 1
      }
      i += 1
    }

    if(ranking <= k) {
      (math.log(2) / math.log(ranking + 1)).toFloat
    } else {
      0
    }
  }

  override def format(): String = "NDCG"
}
