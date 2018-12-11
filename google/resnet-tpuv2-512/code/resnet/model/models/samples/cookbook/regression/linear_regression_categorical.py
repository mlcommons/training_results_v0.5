# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Linear regression with categorical features."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import tensorflow as tf

import automobile_data

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int,
                    help='number of training steps')
parser.add_argument('--price_norm_factor', default=1000., type=float,
                    help='price normalization factor')


def main(argv):
  """Builds, trains, and evaluates the model."""
  args = parser.parse_args(argv[1:])

  (train_x,train_y), (test_x, test_y) = automobile_data.load_data()

  train_y /= args.price_norm_factor
  test_y /= args.price_norm_factor

  # Provide the training input dataset.
  train_input_fn = automobile_data.make_dataset(args.batch_size, train_x, train_y, True, 1000)

  # Provide the validation input dataset.
  test_input_fn = automobile_data.make_dataset(args.batch_size, test_x, test_y)

  # The following code demonstrates two of the ways that `feature_columns` can
  # be used to build a model with categorical inputs.

  # The first way assigns a unique weight to each category. To do this, you must
  # specify the category's vocabulary (values outside this specification will
  # receive a weight of zero).
  # Alternatively, you can define the vocabulary in a file (by calling
  # `categorical_column_with_vocabulary_file`) or as a range of positive
  # integers (by calling `categorical_column_with_identity`)
  body_style_vocab = ["hardtop", "wagon", "sedan", "hatchback", "convertible"]
  body_style_column = tf.feature_column.categorical_column_with_vocabulary_list(
      key="body-style", vocabulary_list=body_style_vocab)

  # The second way, appropriate for an unspecified vocabulary, is to create a
  # hashed column. It will create a fixed length list of weights, and
  # automatically assign each input category to a weight. Due to the
  # pseudo-randomness of the process, some weights may be shared between
  # categories, while others will remain unused.
  make_column = tf.feature_column.categorical_column_with_hash_bucket(
      key="make", hash_bucket_size=50)

  feature_columns = [
      # This model uses the same two numeric features as `linear_regressor.py`
      tf.feature_column.numeric_column(key="curb-weight"),
      tf.feature_column.numeric_column(key="highway-mpg"),
      # This model adds two categorical colums that will adjust the price based
      # on "make" and "body-style".
      body_style_column,
      make_column,
  ]

  # Build the Estimator.
  model = tf.estimator.LinearRegressor(feature_columns=feature_columns)

  # Train the model.
  # By default, the Estimators log output every 100 steps.
  model.train(input_fn=train_input_fn, steps=args.train_steps)

  # Evaluate how the model performs on data it has not yet seen.
  eval_result = model.evaluate(input_fn=test_input_fn)

  # The evaluation returns a Python dictionary. The "average_loss" key holds the
  # Mean Squared Error (MSE).
  average_loss = eval_result["average_loss"]

  # Convert MSE to Root Mean Square Error (RMSE).
  print("\n" + 80 * "*")
  print("\nRMS error for the test set: ${:.0f}"
        .format(args.price_norm_factor * average_loss**0.5))

  print()


if __name__ == "__main__":
  # The Estimator periodically generates "INFO" logs; make these logs visible.
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main=main)
