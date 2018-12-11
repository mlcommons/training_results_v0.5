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

object NcfLogger {
  val header = ":::MLPv0.5.0 ncf"

  def info(message: String, mapping: Array[(String, String)]): Unit = {
    val ste = Thread.currentThread().getStackTrace()
    println(f"$header ${System.currentTimeMillis() / 1e3}%10.3f " +
      f"(${ste(2).getFileName}:${ste(2).getLineNumber}) $message: {" +
      f"${mapping.map(m => s""""${m._1}": ${m._2}""").mkString(", ")}}")
  }

  def info(message: String, value: Float): Unit = {
    val ste = Thread.currentThread().getStackTrace()
    println(f"$header ${System.currentTimeMillis() / 1e3}%10.3f " +
      f"(${ste(2).getFileName}:${ste(2).getLineNumber}) $message: $value")
  }

  def info(message: String, value: Double): Unit = {
    val ste = Thread.currentThread().getStackTrace()
    println(f"$header ${System.currentTimeMillis() / 1e3}%10.3f " +
      f"(${ste(2).getFileName}:${ste(2).getLineNumber}) $message: $value")
  }

  def info(message: String, value: String): Unit = {
    val ste = Thread.currentThread().getStackTrace()
    println(f"$header ${System.currentTimeMillis() / 1e3}%10.3f " +
      s"""(${ste(2).getFileName}:${ste(2).getLineNumber}) $message: "$value"""")
  }

  def info(message: String, value: Int): Unit = {
    val ste = Thread.currentThread().getStackTrace()
    println(f"$header ${System.currentTimeMillis() / 1e3}%10.3f " +
      f"(${ste(2).getFileName}:${ste(2).getLineNumber}) $message: $value")
  }

  def info(message: String, value: Boolean): Unit = {
    val ste = Thread.currentThread().getStackTrace()
    println(f"$header ${System.currentTimeMillis() / 1e3}%10.3f " +
      f"(${ste(2).getFileName}:${ste(2).getLineNumber}) $message: $value")
  }

  def info(message: String): Unit = {
    val ste = Thread.currentThread().getStackTrace()
    println(f"$header ${System.currentTimeMillis() / 1e3}%10.3f " +
      f"(${ste(2).getFileName}:${ste(2).getLineNumber}) $message")
  }

}
