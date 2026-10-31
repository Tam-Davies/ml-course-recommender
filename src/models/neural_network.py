import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding, Flatten


class NeuralNetworkRecommender:
    """Embedding-based neural network recommender."""

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = 16,
    ):
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim

        self.user_embedding = Embedding(
            input_dim=n_users,
            output_dim=embedding_dim,
            name="user_embedding",
        )

        self.item_embedding = Embedding(
            input_dim=n_items,
            output_dim=embedding_dim,
            name="item_embedding",
        )

    def build_model(self) -> Model:
        """Build the embedding-based recommender model."""

        user_input = tf.keras.Input(
            shape=(1,),
            name="user_id",
        )

        item_input = tf.keras.Input(
            shape=(1,),
            name="item_id",
        )

        user_vector = Flatten()(
            self.user_embedding(user_input)
        )

        item_vector = Flatten()(
            self.item_embedding(item_input)
        )

        dot_product = tf.keras.layers.Dot(
            axes=1,
        )([user_vector, item_vector])

        output = tf.keras.layers.Flatten()(dot_product)

        model = Model(
            inputs=[user_input, item_input],
            outputs=output,
        )

        model.compile(
            optimizer="adam",
            loss="mse",
            metrics=["mae"],
        )

        return model