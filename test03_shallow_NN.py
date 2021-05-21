import os

import numpy as np
import pandas as pd
# tensorflow
import tensorflow as tf
# Sklearn
from sklearn import preprocessing, metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
# keras
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers
# export model
import joblib

"""
download the dataset from directory
"""

"""
set up the dataset
train/test = .8/.2
remove col[url]
y = col[category]
x = col[title]
"""
df = pd.read_csv('data/dataset.csv', index_col=0)
df.pop('url')
X = df.copy()
y = X.pop('category')

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.15
)

"""
preprocessing 
encode label
"""
encoder = preprocessing.LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_test = encoder.fit_transform(y_test)

"""
Count vectorizer
assign id to each occurence of text
"""
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train.title)
X_test_counts = count_vect.transform(X_test.title)

"""
TfidfTransformer
translate occurence to frequencies
"""
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)

"""
set up ngram level tfidf
"""
tfidf_vect_ngram = TfidfVectorizer(
    analyzer='word',
    token_pattern=r'\w{1,}',
    ngram_range=(2, 3),
    max_features=5000
)
tfidf_vect_ngram.fit(X_train.title)
X_train_tfidf_ngram = tfidf_vect_ngram.transform(X_train.title)
X_test_tfidf_ngram = tfidf_vect_ngram.transform(X_test.title)


"""
training classifier
using shallow neural network
"""


def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
    # fit training dataset
    classifier.fit(feature_vector_train, label, epochs=1)

    # predict the label
    predictions = classifier.predict(feature_vector_valid)

    if is_neural_net:
        predictions = predictions.argmax(axis=-1)

    return metrics.accuracy_score(predictions, y_test)


def create_model(input_size):
    # create input layer
    input_layer = layers.Input((input_size,), sparse=True)

    # create hidden layer
    hidden_layer = layers.Dense(100, activation='relu')(input_layer)

    # create output layer
    output_layer = layers.Dense(1, activation='sigmoid')(hidden_layer)

    classifier = models.Model(
        inputs=input_layer, outputs=output_layer)
    classifier.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')
    return classifier


classifier = create_model(X_train_tfidf.shape[1])
X_train_tfidf.sort_indices()
X_test_tfidf.sort_indices()
accuracy = train_model(classifier, X_train_tfidf,
                       y_train, X_test_tfidf, is_neural_net=True)

print(accuracy)

# clf = MultinomialNB().fit(X_train_tfidf, y_train)
# joblib.dump(clf, '../backend/kossie/staticfiles/ml-model/clf.pkl')
# joblib.dump(
#     count_vect, '../backend/kossie/staticfiles/ml-model/vectorizer.pkl')
# joblib.dump(tfidf_transformer,
#             '../backend/kossie/staticfiles/ml-model/tfidf.pkl')


# X_test_counts = count_vect.transform(X_test.title)
# X_test_tfidf = tfidf_transformer.transform(X_test_counts)
# y_pred = clf.predict(X_test_tfidf)
# accuracy = accuracy_score(y_test, y_pred)
# print(accuracy)
# result_data = {
#     'prediction': y_pred,
#     'actual': y_test,
#     'content': X_test.title
# }
# result = pd.DataFrame(result_data)
# false = result[result.prediction != result.actual]
