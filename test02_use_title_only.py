import os

import numpy as np
import pandas as pd
# Sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
# plot
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
Count vectorizer
assign id to each occurence of text
"""
count_vect = CountVectorizer(stop_words='english')
X_train_counts = count_vect.fit_transform(X_train.title)

"""
TfidfTransformer
translate occurence to frequencies
"""
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

"""
training classifier
"""
clf = MultinomialNB(alpha=0.1).fit(X_train_tfidf, y_train)

# joblib.dump(clf, '../backend/kossie/staticfiles/ml-model/clf.pkl')
# joblib.dump(
#     count_vect, '../backend/kossie/staticfiles/ml-model/vectorizer.pkl')
# joblib.dump(tfidf_transformer,
#             '../backend/kossie/staticfiles/ml-model/tfidf.pkl')


X_test_counts = count_vect.transform(X_test.title)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)
y_pred = clf.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
result_data = {
    'prediction': y_pred,
    'actual': y_test,
    'content': X_test.title
}
result = pd.DataFrame(result_data)
false = result[result.prediction != result.actual]

# rel = pd.read_csv(
#     '../webcrawl/data/relationship_advice_with_content.csv', index_col=0)
# rel['category'] = 'relationship_advice'
# rel.fillna('empty', inplace=True)
# rel.pop('url')
# rel_X = rel.copy()
# rel_y = rel_X.pop('category')
# rel_X_counts = count_vect.transform(rel_X.content)
# rel_X_tfidf = tfidf_transformer.transform(rel_X_counts)
# rel_pred = clf.predict(rel_X_tfidf)
# rel_accuracy = accuracy_score(rel_y, rel_pred)
# print(rel_accuracy)

# path = "data/ALL Q_A Record 2020.xlsx"
# title_sample = pd.read_excel(path, sheet_name='Website Q&A record 2020')
# title_sample.pop('Responsible')
# title_sample.pop('Publish')
# title_sample_counts = count_vect.transform(title_sample.Problem)
# title_sample_tfidf = tfidf_transformer.transform(title_sample_counts)
# title_prediction = clf.predict(title_sample_tfidf)
# title_sample['prediction'] = title_prediction
# title_sample = title_sample[title_sample['prediction']
#                             != title_sample['Coach']]
# print(title_sample)
# for d in title_sample.Problem:
#     print(d)
# title_sample.to_csv('data/kossie-qna-record-result.csv')
