import os
import pprint
import tempfile

from typing import Dict, Text

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

import tensorflow_recommenders as tfrs

# Ratings data.
ratings = tfds.load("address_to_interations", split="train")
# Features of all the available auctions.
auction = tfds.load("address_to_auction_data", split="train")

unique_user_ids = []
unique_auction_ids = []

embedding_dimension = 32
user_model = tf.keras.Sequential([
  tf.keras.layers.StringLookup(
      vocabulary=unique_user_ids, mask_token=None),
  # We add an additional embedding to account for unknown tokens.
  tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension)
])

auction_model = tf.keras.Sequential([
  tf.keras.layers.StringLookup(
      vocabulary=unique_auction_ids, mask_token=None),
  tf.keras.layers.Embedding(len(unique_auction_ids) + 1, embedding_dimension)
])

metrics = tfrs.metrics.FactorizedTopK(
  candidates=auction.batch(128).map(auction_model)
)

task = tfrs.tasks.Retrieval(
  metrics=metrics
)

class WhatnotAuctionModel(tfrs.Model):

  def __init__(self, user_model, auction_model):
    super().__init__()
    self.auction_model: tf.keras.Model = auction_model
    self.user_model: tf.keras.Model = user_model
    self.task: tf.keras.layers.Layer = task

  def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
    # We pick out the user features and pass them into the user model.
    user_embeddings = self.user_model(features["user_id"])
    # And pick out the auction features and pass them into the auction model,
    # getting embeddings back.
    positive_auction_embeddings = self.auction_model(features["auction_title"])

    # The task computes the loss and the metrics.
    return self.task(user_embeddings, positive_auction_embeddings)

class NoBaseClassWhatnotAuctionModel(tf.keras.Model):

  def __init__(self, user_model, auction_model):
    super().__init__()
    self.auction_model: tf.keras.Model = auction_model
    self.user_model: tf.keras.Model = user_model
    self.task: tf.keras.layers.Layer = task

  def train_step(self, features: Dict[Text, tf.Tensor]) -> tf.Tensor:

    # Set up a gradient tape to record gradients.
    with tf.GradientTape() as tape:

      # Loss computation.
      user_embeddings = self.user_model(features["user_id"])
      positive_auction_embeddings = self.auction_model(features["auction_title"])
      loss = self.task(user_embeddings, positive_auction_embeddings)

      # Handle regularization losses as well.
      regularization_loss = sum(self.losses)

      total_loss = loss + regularization_loss

    gradients = tape.gradient(total_loss, self.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

    metrics = {metric.name: metric.result() for metric in self.metrics}
    metrics["loss"] = loss
    metrics["regularization_loss"] = regularization_loss
    metrics["total_loss"] = total_loss

    return metrics

  def test_step(self, features: Dict[Text, tf.Tensor]) -> tf.Tensor:

    # Loss computation.
    user_embeddings = self.user_model(features["user_id"])
    positive_auction_embeddings = self.auction_model(features["auction_title"])
    loss = self.task(user_embeddings, positive_auction_embeddings)

    # Handle regularization losses as well.
    regularization_loss = sum(self.losses)

    total_loss = loss + regularization_loss

    metrics = {metric.name: metric.result() for metric in self.metrics}
    metrics["loss"] = loss
    metrics["regularization_loss"] = regularization_loss
    metrics["total_loss"] = total_loss

    return metrics
  
model = WhatnotAuctionModel(user_model, auction_model)
model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))

cached_train = [] # to be define as train.shuffle(100_000).batch(8192).cache()
cached_test = [] # to be define as test.batch(4096).cache()

model.fit(cached_train, epochs=3)

# # Create a model that takes in raw query features, and
# index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)
# # recommends auctions out of the entire auctions dataset.
# index.index_from_dataset(
#   tf.data.Dataset.zip((auctions.batch(100), auctions.batch(100).map(model.auction_model)))
# )

# # Get recommendations.
# _, titles = index(tf.constant(["42"]))
# print(f"Recommendations for user 42: {titles[0, :3]}")

# # Model Serving
# # Export the query model.
# with tempfile.TemporaryDirectory() as tmp:
#   path = os.path.join(tmp, "model")

#   # Save the index.
#   tf.saved_model.save(index, path)

#   # Load it back; can also be done in TensorFlow Serving.
#   loaded = tf.saved_model.load(path)

#   # Pass a user id in, get top predicted auction titles back.
#   scores, titles = loaded(["42"])

#   print(f"Recommendations: {titles[0][:3]}")

# scann_index = tfrs.layers.factorized_top_k.ScaNN(model.user_model)
# scann_index.index_from_dataset(
#   tf.data.Dataset.zip((auctions.batch(100), auctions.batch(100).map(model.auction_model)))
# )

# # Get recommendations.
# _, titles = scann_index(tf.constant(["42"]))
# print(f"Recommendations for user 42: {titles[0, :3]}")
