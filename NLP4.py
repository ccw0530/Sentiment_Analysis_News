from __future__ import absolute_import, division, print_function
import json
import re
import nltk
import numpy as np
import pickle
import pandas as pd
from bs4 import BeautifulSoup
import os
import requests
import torch
import torch.nn.functional as F
from nltk import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
# from pytorch_transformers import (BertConfig, BertForTokenClassification, BertTokenizer)
from nltk.corpus import stopwords
import datetime
import matplotlib.pyplot as plt
# from transformers import BertTokenizer
from sklearn.cluster import KMeans
from sklearn.neighbors import BallTree
import tokenizers
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from nltk.stem import PorterStemmer
import pandas_datareader.data as web
import tensorflow as tf
import matplotlib

with open("sp500tickers_nlp.pickle", "rb") as f:
    stock = pickle.load(f)
t = np.array(stock)[:, 0]
ticker = []
for i in t:
    mapping = str.maketrans(".", "-")
    i = i.translate(mapping)
    ticker.append(i)

def get_top_news():
    headlines, dates, tickers, total = [], [], [], []
    for i in range(62):
        i += 1
        url = f"https://seekingalpha.com/market-news/top-news?page={str(i)}"
        headers = {
            "content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
            "Origin": "https://seekingalpha.com",
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36"}
        response = requests.get(url = url, headers = headers)
        while response.status_code != 200:
            response = requests.get(url=url, headers=headers)

        soup = BeautifulSoup(response.text, 'html.parser')
        for script in soup.find_all('div', class_="title"):
            headlines.append(script.text)
        for script in soup.find_all('span', class_="item-date"):
            dates.append(script.text)
        for script in soup.find_all('div', class_="media-left"):
            tickers.append(script.text)
        for i in range(len(headlines)):
            total.append((headlines[i], dates[i], tickers[i]))

    with open("headlines_top_news.pickle","wb") as f:
        pickle.dump(total, f)
# get_top_news()

with open("headlines_top_news.pickle", "rb") as f:
    headlines = pickle.load(f)
def change_dates_tocsv():
    df13 = pd.DataFrame(headlines, columns=('Headlines', 'Dates', 'Ticker'))
    for i in range(len(df13['Dates'])):
        match = re.search(r'\w{3}\.\s\d{1,2}\,\s\d{4}|May\s\d{1,2}\,\s\d{4}|\w{3}\.\s\d{1,2}|May\s\d{1,2}|\bToday\b',
                          df13['Dates'][i])
        if re.search(r'\w{3}\.\s\d{1,2}\,\s\d{4}|\w{3}\s\d{1,2}\,\s\d{4}', match[0]):
            fulldate = match[0]
        elif re.search(r'\bToday\b', match[0]):
            fulldate = 'Jun 22, 2020'
        else:
            fulldate = match[0] + ", 2020"

        if len(df13['Dates'][i].split(',')) == 2:
            a = datetime.datetime.strptime(df13['Dates'][i].split(',')[1].strip(), '%I:%M %p').time()
        elif len(df13['Dates'][i].split(',')) == 3:
            a = datetime.datetime.strptime(df13['Dates'][i].split(',')[2].strip(), '%I:%M %p').time()

        if a > datetime.time(16):
            for fmt in ('%b. %d, %Y', '%b %d, %Y'):
                try:
                    newDate = datetime.datetime.strptime(fulldate, fmt).date()
                    newDate = newDate + datetime.timedelta(days=1)
                    df13['Dates'][i] = newDate
                    break
                except ValueError:
                    pass
        elif a < datetime.time(10):
            for fmt in ('%b. %d, %Y', '%b %d, %Y'):
                try:
                    newDate = datetime.datetime.strptime(fulldate, fmt).date()
                    newDate = newDate - datetime.timedelta(days=1)
                    df13['Dates'][i] = newDate
                    break
                except ValueError:
                    pass
    df13.to_csv('Top_News_2.csv', index=False)
# change_dates_tocsv()

stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

df13 = pd.read_csv('Top_News_2.csv')
df13.iloc[:, 0].replace(['^a-zA-Z'], ' ', regex=True, inplace=True)

start = datetime.datetime(2015, 12, 20)
end = datetime.datetime(2020, 6, 22)
df_stock = web.DataReader('^GSPC', 'yahoo', start, end)
df_stock.to_csv('^GSPC_2.csv')
df5 = pd.read_csv('^GSPC_2.csv', index_col=0)
df5 = df5.drop(['Open', 'High', 'Low', 'Close', 'Volume'], axis=1)  # drop unwanted rows
df5['Return'] = df5['Adj Close'] / df5['Adj Close'].shift(1) - 1
df5['Label'] = ''
for i in range(len(df5['Return'])):
    if df5['Return'][i] >= 0:
        df5['Label'][i] = 0
    elif df5['Return'][i] < 0:
        df5['Label'][i] = 1
    else:
        df5['Label'][i] = np.nan

def stopwords_and_stemmimg(df):
    date_dict = {}
    data_for_df = []
    for i in range(len(df['Dates'])):
        date_dict[df['Dates'][i]] = []

    for i in range(len(df['Headlines'])):
        news = df['Headlines'][i]
        word_tokens = nltk.word_tokenize(news)
        filtered_sentence = [w for w in word_tokens if w not in stop_words]
        stemmed_sentence = [ps.stem(w) for w in filtered_sentence]
        news = ' '.join(stemmed_sentence)
        date_dict[df['Dates'][i]].append(news)

    for i in date_dict:
        data_for_df.append((' '.join(date_dict[i]).lower(), i))

    with open("data_for_df.pickle", "wb") as f:
        pickle.dump(data_for_df, f)


    with open("data_for_df.pickle", "rb") as f:
        data_for_df = pickle.load(f)
    df14 = pd.DataFrame(data_for_df)
    df14.columns = ['Headlines', 'Dates']
    df14.iloc[:, 0].replace(['^a-zA-Z'], ' ', regex=True, inplace=True)
    df14 = df14.sort_values(by='Dates')
    df14.set_index('Dates', inplace=True)

    df11 = pd.merge(df14['Headlines'], df5['Label'], left_index=True, right_index=True, how='left')
    df11['New_Headlines'] = df11['Headlines'].shift(1)
    df11.dropna(inplace=True)
    df11['Label'] = df11['Label'].astype('int')
    df11.reset_index(inplace=True)
    return df11


results = []

def news_sentiment():
    for i in range(len(df13['Headlines'])):
        news = df13['Headlines'][i]
        news = nltk.word_tokenize(news)
        news = [w for w in news if w not in stop_words]
        news = ' '.join(news)
        pol_score = SIA().polarity_scores(news)  # run analysis
        pol_score['headline'] = news  # add headlines for viewing
        pol_score['date'] = df13['Dates'][i]
        results.append(pol_score)

    with open("result_top_news_2.pickle", "wb") as f:
        pickle.dump(results, f)
# news_sentiment()

with open("result_top_news_2.pickle", "rb") as f:
    results = pickle.load(f)

def read_news():
    df7 = pd.DataFrame(results)
    df7 = df7.drop(['neg', 'neu', 'pos'], axis=1)
    df7.columns = ['Score', 'Headlines', 'Dates']

    df4 = df7.groupby(['Dates']).sum()
    df4['New_Score'] = df4['Score'].shift(1)

    df6 = pd.merge(df5[['Return']], df4[['New_Score']], left_index=True, right_index=True, how='left')
    df6.reset_index(inplace=True)
    fig, ax1 = plt.subplots()
    df6['Date']= pd.to_datetime(df6['Date'])
    ax1.plot(df6['Date'], df6['New_Score'], 'g-')
    ax1.xaxis.set_major_locator(matplotlib.dates.MonthLocator([1, 4, 7, 10]))
    ax1.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y-%m'))
    ax2 = ax1.twinx()
    ax2.plot(df6['Date'], df6['Return'], 'b-')
    ax1.set_xlabel('Dates')
    ax1.set_ylabel('New_Score', color='g')
    ax2.set_ylabel('Return', color='b')
    plt.tight_layout()

    df6.fillna(0, inplace=True)
    # dfReturnsScore = df6[(df6['New_Score'] > 50) | (df6['New_Score'] < -50)]
    df6.plot(x="New_Score", y="Return", style="o")
    plt.show()
    # print(df6['Return'].corr(df6['New_Score']))

    dfReturnsScore3 = pd.merge(df5[['Return']], df4[['New_Score', 'Score']], left_index=True, right_index=True, how='left')
    dfReturnsScore3.fillna(0, inplace=True)
    # print(dfReturnsScore3)
    count = 0
    total = 0
    for i in range(len(dfReturnsScore3['New_Score'])):
        diff = dfReturnsScore3['New_Score'][i] - dfReturnsScore3['Score'][i]
        if diff > 0 and dfReturnsScore3['Return'][i] > 0:
            count += 1
        elif diff < 0 and dfReturnsScore3['Return'][i] < 0:
            count += 1
        else:
            pass
        total += 1
    # print(total, count/total*100, '%')
    dfReturnsScore = pd.merge(df5[['Return']], df4[['Score']], left_index=True, right_index=True, how='left')
    return dfReturnsScore
# read_news()

number_of_news = {}
for i in df13['Ticker']:
    if i in ticker:
        number_of_news[i] = 0

def plot():
    plot_number_of_news = []
    for i in range(len(df13['Ticker'])):
        if df13['Ticker'][i] in ticker:
            number_of_news[df13['Ticker'][i]] += 1
    top_coms = {k: v for k, v in sorted(number_of_news.items(), key=lambda item: item[1])}
    for i, j in top_coms.items():
        plot_number_of_news.append((i, j))
    plot_number_of_news = np.array(plot_number_of_news).reshape(-1,2)
    top100 = plot_number_of_news[-100:]
    plt.subplots()
    plt.gca().invert_xaxis()
    plt.bar(top100[:, 0], np.array(top100[:, 1].astype('int')))
    plt.xlabel('Companies')
    plt.ylabel('Counts')
    plt.title('Number of news for each company')
    plt.ylim(100, 9000)
    plt.xticks(top100[:, 0], rotation='vertical')
    plt.show()
# plot()


def create_Kmanes():
    # news_id = []
    # news_id_with_dates = []
    total_news = []
    for i in range(len(df13['Headlines'])):
        news = nltk.word_tokenize(df13['Headlines'][i].lower())
        filtered_sentence = [w for w in news if w not in stop_words]
        stemmed_sentence = [ps.stem(w) for w in filtered_sentence]
        news = ' '.join(stemmed_sentence)
        total_news.append(news)
    with open("total_news_2.pickle", "wb") as f:
        pickle.dump(total_news, f)
# create_Kmanes()

with open("total_news_2.pickle", "rb") as f:
    total_news = pickle.load(f)

def create_total_news():
    # countvector = CountVectorizer(ngram_range=(1, 2), max_df=0.4, max_features=10000, stop_words=stop_words)
    tfidfconverter = TfidfVectorizer(ngram_range=(1, 1), max_features=12000, max_df=0.4, stop_words=stop_words)
    traindataset = tfidfconverter.fit_transform(total_news)
    clf = KMeans(n_clusters=100)
    clf.fit(traindataset)
    labels = clf.labels_
    centroids = clf.cluster_centers_
    with open("labels_2.pickle", "wb") as f:
        pickle.dump(labels, f)
    with open("centroids_2.pickle", "wb") as f:
        pickle.dump(centroids, f)
# create_total_news()

with open("labels_2.pickle", "rb") as f:
    labels = pickle.load(f)
with open("centroids_2.pickle", "rb") as f:
    centroids = pickle.load(f)


def create_cluster_news():
    cluster_id = {}
    for x in labels:
        for z in np.where(labels == x):
            cluster_id[x] = z
    cluster_news = {}
    for i, j in cluster_id.items():
        cluster_news[i] = []
        for k in j:
            cluster_news[i].append(k)
    with open("cluster_news_top_news_2.pickle", "wb") as f:
        pickle.dump(cluster_news, f)
# create_cluster_news()

with open("cluster_news_top_news_2.pickle", "rb") as f:
    cluster_news = pickle.load(f)
tree = BallTree(centroids)
dist, ind = tree.query(centroids, k=2)

cluster_news_per_segment = []
for i in ind:
    # if 0 not in i:
    for index in i:
        for j in cluster_news[index]:
            cluster_news_per_segment.append((df13['Headlines'][j], df13['Dates'][j]))
df18 = pd.DataFrame(cluster_news_per_segment)
df18.columns = ['Headlines', 'Dates']
df18.iloc[:, 0].replace(['^a-zA-Z'], ' ', regex=True, inplace=True)
df18 = df18.drop_duplicates(subset='Headlines', keep='first')
df18 = df18.sort_values(by='Dates')
df18.reset_index(drop=True, inplace=True)


# for i in ind:
#     if 0 not in i:
#         print(sorted(i))
# print(np.sort(ind))
# for i in ind:
#     print(sorted(i))
# print(df13)
# print(df18)
df = stopwords_and_stemmimg(df18)
traindata = df.iloc[:int(len(df['Headlines'])*0.8)]
testdata = df.iloc[int(len(df['Headlines'])*0.8):]

"""Grid Search for Random Forest"""
# pipeline = Pipeline([
#     # ('vect', CountVectorizer()),
#     ('tfidf', TfidfVectorizer()),
#     ('clf', RandomForestClassifier()),
# ])
#
# parameters = {
#     # 'vect__max_df': (0.3, 0.4, 0.5, 0.6),
#     # 'vect__max_features': (9000, 10000, 11000, 12000),
#     # 'vect__ngram_range': ((1, 1), (1, 2), (2, 2)),
#     # 'vect__stop_words': [stop_words],
#     'tfidf__max_df': (0.3, 0.4, 0.5, 0.6),
#     'tfidf__max_features': (10000, 40000, 50000),
#     'tfidf__ngram_range': ((1, 1), (1, 2), (2, 2)),
#     'tfidf__stop_words': [stop_words],
#     'clf__n_estimators': (1000, 1500, 2000),
#     'clf__criterion': ['entropy']
# }
#
# grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1, cv=10)
#
# grid_search.fit(traindata['Headlines'], traindata['Label'])
# print("Best score: %0.3f" % grid_search.best_score_)
# print("Best parameters set:")
# best_parameters = grid_search.best_estimator_.get_params()
# for param_name in sorted(parameters.keys()):
#     print("\t%s: %r" % (param_name, best_parameters[param_name]))

"""Random Forest"""
# countvector = CountVectorizer(ngram_range=(1, 2), max_df=0.4, max_features=10000, stop_words=stop_words)
tfidfconverter = TfidfVectorizer(ngram_range=(1, 1), max_features=12000, max_df=0.4, stop_words=stop_words)
traindataset = tfidfconverter.fit_transform(traindata['New_Headlines'].values)
randomclassifier = RandomForestClassifier(n_estimators=4000, criterion='entropy')
randomclassifier.fit(traindataset, traindata['Label'])

testdataset = tfidfconverter.transform(testdata['New_Headlines'])
predictions = randomclassifier.predict(testdataset)
predictions_prob = randomclassifier.predict_proba(testdataset)

accuarcy = accuracy_score(testdata['Label'], predictions)
classification = classification_report(testdata['Label'], predictions)
matrix = confusion_matrix(testdata['Label'], predictions)
print(accuarcy)
print(classification)
print(matrix)

"""MLP"""
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=1024, activation='relu', kernel_initializer='he_uniform'))
model.add(tf.keras.layers.Dense(units=64, activation='relu'))
model.add(tf.keras.layers.Dense(units=1, activation='sigmoid',))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(traindataset.toarray(), traindata['Label'], epochs=20, batch_size=32, verbose=1, validation_split=0.2)

val_loss, val_acc = model.evaluate(testdataset.toarray(), testdata['Label'])
print(val_loss)
print(val_acc)
# print(np.array(traindata['Label']).reshape(1, -1, 1))

"""LSTM"""
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.LSTM(units=256, input_shape=(1, traindataset.toarray().shape[1]), return_sequences=True))
model.add(tf.keras.layers.LSTM(units=256, input_shape=(1, traindataset.toarray().shape[1])))
model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(np.array(traindataset.toarray()).reshape(traindataset.toarray().shape[0], 1, traindataset.toarray().shape[1]), traindata['Label'], epochs=10, verbose=1, validation_split=0.3, shuffle=False)

val_loss, val_acc = model.evaluate(np.array(testdataset.toarray()).reshape(testdataset.toarray().shape[0], 1, testdataset.toarray().shape[1]), testdata['Label'])
print(val_loss)
print(val_acc)


