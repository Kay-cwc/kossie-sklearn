import numpy as np

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

import matplotlib.pyplot as plt

"""
download the dataset from tfds
"""
train_data, test_data = tfds.load(
    name='imdb_reviews',
    split=['train', 'test'],
    batch_size=-1,
    as_supervised=True
)

"""
setup the data into two sets
train and test data
"""
train_examples, train_labels = tfds.as_numpy(train_data)
test_examples, test_labels = tfds.as_numpy(test_data)

"""
call the pre-trained text-processing model to represent the text
"""
model = "https://tfhub.dev/google/nnlm-en-dim50/2"
hub_layer = hub.KerasLayer(
    model,
    input_shape=[],
    dtype=tf.string,
    trainable=True
)
"""
build the keras layer **tensorflow hub layer**
the layer map the sentenses into embedding vector
"""
model = tf.keras.Sequential()
# build the hub_layer that map the sentenses into embedding vector
model.add(hub_layer)
# the output vector is piped here with 16 units
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1))

model.summary()

"""
configure model with optmizer and loss function => binary_crossentropy
"""
model.compile(
    optimizer='adam',
    loss=tf.losses.BinaryCrossentropy(from_logits=True),
    metrics=[tf.metrics.BinaryAccuracy(threshold=0.0, name='accuracy')]
)

"""
extract 10000 sets of data to validate the model
distinct from test data
"""
x_val = train_examples[:10000]
partial_x_train = train_examples[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]


# train the model
history = model.fit(
    partial_x_train,
    partial_y_train,
    epochs=40,
    batch_size=512,
    validation_data=(x_val, y_val),
    verbose=1
)

# result
results = model.evaluate(test_data, test_labels)
print(results)
