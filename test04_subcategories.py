from sklearn import preprocessing
import joblib
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB, GaussianNB, ComplementNB, BernoulliNB, CategoricalNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer, HashingVectorizer
from sklearn.model_selection import train_test_split
import os

import numpy as np
import pandas as pd
pd.set_option('display.max_rows', None)
# Sklearn

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
stop_word_list = [
    'hence', 'found', 'twelve', 'ourselves', 'last', 'inc', 'formerly', 'keep', 'each', 'perhaps',
    'elsewhere', 'below', 'were', 'mostly', 'now', 'my', 'almost', 'beside', 'since', 'system',
    'herein', 'further', 'your', 'he', 'namely', 'behind', 'many', 'these', 'either', 'else', 'go',
    'detail', 'mill', 'moreover', 'more', 'seems', 're', 'anything', 'own', 'still', 'first', 'whereby',
    'another', 'most', 'two', 'with', 'otherwise', 'ie', 'but', 'will', 'describe', 'six', 'hereafter',
    'can', 'besides', 'thence', 'you', 'has', 'un', 'fifteen', 'herself', 'might', 'four', 'within',
    'her', 'into', 'front', 'only', 'move', 'they', 'above', 'next', 'hers', 'during', 'been',
    'then', 'name', 'such', 'had', 'much', 'often', 'however', 'a', 'must', 'back', 'etc', 'that', 'done',
    'very', 'myself', 'nothing', 'thru', 'never', 'nor', 'beforehand', 'along', 'be', 'enough', 'less',
    'ours', 'wherever', 'us', 'give', 'whereupon', 'i', 'or', 'whatever', 'latterly', 'except', 'his',
    'throughout', 'whether', 'three', 'thereafter', 'before', 'anyone', 'is', 'five', 'here', 'when',
    'at', 'nine', 'empty', 'amoungst', 'its', 'onto', 'though', 'bill', 'their', 'from', 'somehow',
    'the', 'nowhere', 'thick', 'beyond', 'ten', 'de', 'thereupon', 'and', 'anyhow', 'seeming', 'whole',
    'sometime', 'whoever', 'least', 'towards', 'between', 'per', 'anywhere', 'whither', 'wherein', 'yourselves',
    'was', 'whose', 'being', 'whereafter', 'why', 'became', 'not', 'too', 'other', 'put', 'cannot', 'therein',
    'whom', 'so', 'somewhere', 'call', 'everyone', 'becoming', 'what', 'con', 'please', 'as', 'someone', 'about',
    'side', 'see', 'than', 'may', 'few', 'everything', 'some', 'against', 'neither', 'nevertheless', 'him',
    'fire', 'whereas', 'no', 'again', 'something', 'due', 'under', 'although', 'do', 'over', 'ever', 'among',
    'rather', 'latter', 'seem', 'top', 'are', 'yet', 'we', 'couldnt', 'also', 'hereupon', 'himself', 'get',
    'via', 'cry', 'if', 'eg', 'yours', 'both', 'therefore', 'until', 'everywhere', 'on', 'by',
    'find', 'seemed', 'eight', 'hundred', 'fifty', 'co', 'always', 'whence', 'have', 'where', 'it',
    'themselves', 'hasnt', 'hereby', 'itself', 'for', 'how', 'serious', 'up', 'amongst', 'an',
    'meanwhile', 'several', 'any', 'off', 'am', 'mine', 'across', 'yourself', 'becomes', 'once',
    'take', 'amount', 'upon', 'which', 'out', 'of', 'full', 'others', 'one', 'to', 'bottom', 'made',
    'even', 'down', 'thus', 'in', 'who', 'our', 'same', 'she', 'afterwards', 'anyway', 'around', 'fill',
    'would', 'part', 'thereby', 'forty', 'noone', 'already', 'sixty', 'cant', 'none', 'me', 'could',
    'show', 'without', 'because', 'through', 'those', 'indeed', 'this', 'become', 'there', 'while',
    'ltd', 'them', 'after', 'sincere', 'third', 'all', 'eleven', 'twenty', 'toward', 'former', 'should',
    'sometimes', 'every', 'whenever', '000', '10', '100', '100k', '100x', '101', '10days', '10k', '10m',
    '11', '11m', '12', '13', '13f', '13nb', '14', '14f', '14m', '15', '15f', '15m', '15yrs', '16', '16f',
    '16m', '17', '17f', '17m', '18', '18f', '18m', '19', '19enby', '19f', '19m', '20', '200', '2000',
    '2020', '2021', '20f', '20m', '20s', '20sf', '21', '21f', '21m', '21st', '22', '22f', '22m', '22y',
    '23', '23f', '23m', '23nb', '23y', '24', '24f', '24m', '25', '25f', '25m', '25nb', '26', '26f',
    '26k', '26m', '27', '27f', '27m', '28', '28f', '28m', '28nb', '28y', '29', '29f', '29m', '29yo',
    '2nd', '2weeks', '30', '300', '3000', '30f', '30m', '30s', '30sf', '31', '31f', '31m', '32', '32f',
    '32m', '33', '33f', '33m', '34', '34f', '34m', '35', '35000', '35f', '35m', '36', '36f', '36m',
    '37', '3780', '37f', '37m', '38', '38f', '38m', '39', '39f', '39m', '3f', '3rd', '3yrs', '40',
    '400', '40f', '40m', '40s', '40sf', '41m', '42f', '43', '43f', '44', '44f', '44m', '45', '45f',
    '45m', '46f', '46m', '47', '47f', '47m', '49', '49m', '4m', '4th', '50', '50cc', '50f', '50k',
    '50m', '51', '53m', '54f', '55f', '55m', '56', '56f', '56m', '57f', '58m', '59f', '5changed', '5k',
    '5th', '60k', '63', '65', '66m', '6f', '6km', '6y', '70', '8years', '8yr', '90', '95f', '99', '9f', '9m',
    'sorta', 'able', '13m', '17nb', '21y', '28yo', '37am', '41f', '42m', '43m', '48f', '4f',
    '500', '51m', '52f', '54m', '60f', '6m', 'aa', 'nearly', 'nice', 'normally', 'm17', 'm18',
    'm19', 'm20', 'm21', 'm22', 'm23', 'm24', 'm25', 'm26', 'm27', 'm28', 'm29', 'm30', 'm31',
    'm32', 'm36', 'm39', 'm41', 'm42', 'm52', 'm61', 'use'
]

df = pd.read_csv('data/dataset.csv', index_col=0)
df.pop('url')
df = df[~df['title'].str.contains('meta')]
df = df[df['title'].str.len() > 50]

df_set1 = df.copy()
# set that contain relationship and
# df_set1.loc[df['category'].isin(['relationship', 'sex']), [
#     'category']] = 'love'

# subset that only contain `sex / relationship` data
df_set2 = df.loc[df['category'].isin(['relationship', 'sex'])]


def saveClf(rootDir, classifier, vectorizer, labeEncoder):

    joblib.dump(
        classifier,
        rootDir + 'clf.pkl'
    )
    joblib.dump(
        vectorizer,
        rootDir + 'vectorizer.pkl'
    )

    joblib.dump(
        labeEncoder,
        rootDir + 'labelEncoder.pkl'
    )


def pipeline(classifier, dataset, dense=False):

    x = dataset.copy()
    y = x.pop('category')

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.15)

    le = preprocessing.LabelEncoder()
    le.fit(y_train)
    y_train_le = le.transform(y_train)
    y_test_le = le.transform(y_test)

    # vectorise words
    if dense:
        count_vect = CountVectorizer(
            stop_words=stop_word_list,
        )
    else:
        count_vect = TfidfVectorizer(
            stop_words=stop_word_list,
        )

    X_train_counts = count_vect.fit_transform(X_train.title)
    X_test_counts = count_vect.transform(X_test.title)

    # count frequency
    # tfidf_transformer = TfidfTransformer()
    # X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    # X_test_tfidf = tfidf_transformer.transform(X_test_counts)

    # train model
    clf = classifier.fit(X_train_counts, y_train_le)

    saveClf('../backend/kossie/staticfiles/ml-model/', clf, count_vect, le)

    # predict and calc accuracy
    prediction = clf.predict(X_test_counts)
    accuracy = accuracy_score(y_test_le, prediction)
    print(accuracy)

    result_data = {
        'prediction': le.inverse_transform(prediction),
        'actual': le.inverse_transform(y_test_le),
        'content': X_test.title
    }
    result = pd.DataFrame(result_data)
    false = result[result.prediction != result.actual]
    false = false[
        ~(false.prediction.isin(['sex', 'relationship']) &
          false.actual.isin(['sex', 'relationship']))
    ]
    print(false.shape)

    # return count_vect, tfidf_transformer, clf,
    return false


test1 = list()
test2 = list()
# pipeline(MultinomialNB(), df_set1, )
false_df = pipeline(ComplementNB(), df_set1, True)
false_df.to_csv('data/false_prediction_without_relationship_to_sex.csv')
# pipeline(BernoulliNB(), df_set1, )


# i = 1
# while i < 100:
#     print('set {}'.format(i))
#     test1.append(pipeline(ComplementNB(), df_set1, True))
#     test2.append(pipeline(ComplementNB(), df_set1, False))
#     i += 1
# print('===============================')

# print(np.mean(test1))
# print(np.mean(test2))
