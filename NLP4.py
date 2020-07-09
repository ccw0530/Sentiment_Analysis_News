import re
import nltk
import numpy as np
import pickle
import pandas as pd
from bs4 import BeautifulSoup
import requests
from nltk.corpus import stopwords
import datetime
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.neighbors import BallTree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from nltk.stem import PorterStemmer
import pandas_datareader.data as web
import tensorflow as tf


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
        if response.status_code == 200:
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
def change_dates_tocsv(headlines, today='', yesterday=''):
    df1 = pd.DataFrame(headlines, columns=('Headlines', 'Dates', 'Ticker'))
    df1.iloc[:, 0].replace(['^a-zA-Z'], ' ', regex=True, inplace=True)
    df1 = df1.drop_duplicates()
    df1 = df1.sort_values(by='Dates')
    df1.reset_index(drop=True, inplace=True)
    for i in range(len(df1['Dates'])):
        match = re.search(r'\w{3}\.\s\d{1,2}\,\s\d{4}|May\s\d{1,2}\,\s\d{4}|\w{3}\.\s\d{1,2}|May\s\d{1,2}|\bToday\b|\bYesterday\b',
                          df1['Dates'].iloc[i])
        if re.search(r'\w{3}\.\s\d{1,2}\,\s\d{4}|\w{3}\s\d{1,2}\,\s\d{4}', match[0]):
            fulldate = match[0]
        elif re.search(r'\bToday\b', match[0]):
            fulldate = today
        elif re.search(r'\bYesterday\b', match[0]):
            fulldate = yesterday
        else:
            fulldate = match[0] + ", 2020"

        if len(df1['Dates'].iloc[i].split(',')) == 2:
            time = datetime.datetime.strptime(df1['Dates'].iloc[i].split(',')[1].strip(), '%I:%M %p').time()
        elif len(df1['Dates'].iloc[i].split(',')) == 3:
            time = datetime.datetime.strptime(df1['Dates'].iloc[i].split(',')[2].strip(), '%I:%M %p').time()

        if time > datetime.time(16):
            for fmt in ('%b. %d, %Y', '%b %d, %Y'):
                try:
                    newDate = datetime.datetime.strptime(fulldate, fmt).date()
                    newDate = newDate + datetime.timedelta(days=1)
                    df1['Dates'].iloc[i] = newDate
                    break
                except ValueError:
                    pass
        else:
            for fmt in ('%b. %d, %Y', '%b %d, %Y'):
                try:
                    newDate = datetime.datetime.strptime(fulldate, fmt).date()
                    df1['Dates'].iloc[i] = newDate
                    break
                except ValueError:
                    pass
    df1.to_csv('Top_News_2.csv', index=False)
change_dates_tocsv(headlines, today='Jul. 8, 2020', yesterday='Jul. 7, 2020')

stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

df1 = pd.read_csv('Top_News_2.csv')
df1.iloc[:, 0].replace(['^a-zA-Z'], ' ', regex=True, inplace=True)

def create_price(start, end):
    df_stock = web.DataReader('^GSPC', 'yahoo', start, end)
    df_stock.to_csv('^GSPC_2.csv')
    df2 = pd.read_csv('^GSPC_2.csv', index_col=0)
    df2 = df2.drop(['Open', 'High', 'Low', 'Close', 'Volume'], axis=1)  # drop unwanted rows
    df2['Return'] = df2['Adj Close'] / df2['Adj Close'].shift(1) - 1
    df2['Label'] = ''
    for i in range(len(df2['Return'])):
        if df2['Return'].iloc[i] >= 0:
            df2['Label'].iloc[i] = 0
        elif df2['Return'].iloc[i] < 0:
            df2['Label'].iloc[i] = 1
        else:
            df2['Label'].iloc[i] = np.nan
    return df2
start = datetime.datetime(2015, 12, 20)
end = datetime.datetime(2020, 6, 22)
df2 = create_price(start, end)

def stopwords_and_stemmimg(df, df2):
    date_dict = {}
    data_for_df = []
    for i in range(len(df['Dates'])):
        date_dict[df['Dates'].iloc[i]] = []

    for i in range(len(df['Headlines'])):
        news = df['Headlines'].iloc[i]
        word_tokens = nltk.word_tokenize(news)
        filtered_sentence = [w for w in word_tokens if w not in stop_words]
        stemmed_sentence = [ps.stem(w) for w in filtered_sentence]
        news = ' '.join(stemmed_sentence)
        date_dict[df['Dates'].iloc[i]].append(news)

    for i in date_dict:
        data_for_df.append((' '.join(date_dict[i]).lower(), i))

    with open("data_for_df.pickle", "wb") as f:
        pickle.dump(data_for_df, f)

    with open("data_for_df.pickle", "rb") as f:
        data_for_df = pickle.load(f)
    df3 = pd.DataFrame(data_for_df)
    df3.columns = ['Headlines', 'Dates']
    df3.iloc[:, 0].replace(['^a-zA-Z'], ' ', regex=True, inplace=True)
    df3 = df3.sort_values(by='Dates')
    df3.set_index('Dates', inplace=True)

    df4 = pd.merge(df3['Headlines'], df2['Label'], left_index=True, right_index=True, how='left')
    df4['New_Headlines'] = df4['Headlines'].shift(1)
    df4.dropna(inplace=True)
    df4['Label'] = df4['Label'].astype('int')
    df4.reset_index(inplace=True)
    return df4

number_of_news = {}
for i in df1['Ticker']:
    if i in ticker:
        number_of_news[i] = 0

def plot():
    plot_number_of_news = []
    for i in range(len(df1['Ticker'])):
        if df1['Ticker'].iloc[i] in ticker:
            number_of_news[df1['Ticker'][i]] += 1
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
    plt.ylim(0, 300)
    plt.xticks(top100[:, 0], rotation='vertical')
    plt.show()
# plot()

def create_Kmanes():
    total_news = []
    for i in range(len(df1['Headlines'])):
        news = nltk.word_tokenize(df1['Headlines'].iloc[i].lower())
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
# for i in ind:
#     if 25 not in i:
#         print(i)
#         for index in i:
#             for j in cluster_news[index]:
#                 print(index, df1['Headlines'][j], df1['Dates'][j])

df = stopwords_and_stemmimg(df1, df2)
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
# print(accuarcy)
# print(classification)
# print(matrix)

"""MLP"""
# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Dense(units=1024, activation='relu', kernel_initializer='he_uniform'))
# model.add(tf.keras.layers.Dense(units=64, activation='relu'))
# model.add(tf.keras.layers.Dense(units=1, activation='sigmoid',))
#
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.fit(traindataset.toarray(), traindata['Label'], epochs=20, batch_size=32, verbose=1, validation_split=0.2)
#
# val_loss, val_acc = model.evaluate(testdataset.toarray(), testdata['Label'])
# print(val_acc)
# print(np.array(traindata['Label']).reshape(1, -1, 1))

"""LSTM"""
# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.LSTM(units=256, input_shape=(1, traindataset.toarray().shape[1]), return_sequences=True))
# model.add(tf.keras.layers.LSTM(units=256, input_shape=(1, traindataset.toarray().shape[1])))
# model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
#
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.fit(np.array(traindataset.toarray()).reshape(traindataset.toarray().shape[0], 1, traindataset.toarray().shape[1]), traindata['Label'], epochs=10, verbose=1, validation_split=0.3, shuffle=False)
#
# val_loss, val_acc = model.evaluate(np.array(testdataset.toarray()).reshape(testdataset.toarray().shape[0], 1, testdataset.toarray().shape[1]), testdata['Label'])
# print(val_acc)

filename = 'finalized_model.sav'
pickle.dump(randomclassifier, open(filename, 'wb'))
pickle.dump(tfidfconverter,open('feature.pkl', 'wb'))

def testing_get_news():
    headlines, dates, tickers, total = [], [], [], []
    for i in range(1):
        i += 1
        url = f"https://seekingalpha.com/market-news/top-news?page={str(i)}"
        headers = {
            "content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
            "Origin": "https://seekingalpha.com",
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36"}
        response = requests.get(url = url, headers = headers)
        while response.status_code != 200:
            response = requests.get(url=url, headers=headers)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            for script in soup.find_all('div', class_="title"):
                headlines.append(script.text)
            for script in soup.find_all('span', class_="item-date"):
                dates.append(script.text)
            for script in soup.find_all('div', class_="media-left"):
                tickers.append(script.text)
            for i in range(len(headlines)):
                total.append((headlines[i], dates[i], tickers[i]))
    return total

def testing(start, end, today, yesterday):
    news = testing_get_news()
    ytd_news = [i for i in news]
    change_dates_tocsv(ytd_news, today, yesterday)
    df1 = pd.read_csv('Top_News_2.csv')
    df1.iloc[:, 0].replace(['^a-zA-Z'], ' ', regex=True, inplace=True)
    df2 = create_price(start, end)
    print(df2)
    df = stopwords_and_stemmimg(df1, df2)
    for i in range(len(df['Label'])):
        print('Actual: ', df['Label'].iloc[i])
    tfidfconverter = pickle.load(open('feature.pkl', 'rb'))
    dataset = tfidfconverter.transform(df['New_Headlines'])
    loaded_model = pickle.load(open(filename, 'rb'))
    result = loaded_model.predict(dataset)
    print('Predict: ', result[0])

start = datetime.datetime(2020, 6, 26)
end = datetime.datetime(2020, 6, 26)
today = 'Jul. 8, 2020'
yesterday = 'Jul. 7, 2020'
testing(start, end, today, yesterday)