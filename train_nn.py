import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate
from tensorflow.keras.models import Model

# LOAD DATA
ratings = pd.read_csv("data/u.data", sep="\t",
                      names=["user", "movie", "rating", "timestamp"])

movies = pd.read_csv("data/u.item", sep="|", encoding="latin-1", header=None)
movies = movies[[0] + list(range(5, 24))]
movies.columns = ["movie"] + [f"genre_{i}" for i in range(19)]

df = pd.merge(ratings, movies, on="movie")

# PREPARE
num_users = df["user"].nunique()
num_movies = df["movie"].nunique()

user_input = df["user"].values - 1
movie_input = df["movie"].values - 1
genres = df[[f"genre_{i}" for i in range(19)]].values.astype("float32")

ratings_target = df["rating"].values

# MODEL
user_in = Input(shape=(1,))
movie_in = Input(shape=(1,))
genre_in = Input(shape=(19,))

user_emb = Embedding(num_users, 10)(user_in)
movie_emb = Embedding(num_movies, 10)(movie_in)

user_vec = Flatten()(user_emb)
movie_vec = Flatten()(movie_emb)

x = Concatenate()([user_vec, movie_vec, genre_in])
x = Dense(64, activation="relu")(x)
x = Dense(32, activation="relu")(x)
out = Dense(1)(x)

model = Model([user_in, movie_in, genre_in], out)
model.compile(optimizer="adam", loss="mse")

model.fit(
    [user_input, movie_input, genres],
    ratings_target,
    epochs=5,
    batch_size=64
)

model.save("nn_model.keras")

print("✅ NN model saved")