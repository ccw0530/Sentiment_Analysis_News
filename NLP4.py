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
import matplotlib
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
from keras.utils import np_utils
from sklearn.metrics import log_loss
from sklearn import metrics
from scipy.spatial.distance import cdist
from keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.callbacks import TensorBoard
from tokenizers import ByteLevelBPETokenizer
from transformers import TFRobertaModel
import tokenizers
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import gensim
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from tqdm import tqdm
from sklearn.decomposition import TruncatedSVD

stop_words = set(stopwords.words('english'))
"""CUSTOM THE STO WORDS BOTH APPEARING IN UO AND DOWN MOVEMENT"""
stop_words.update(['may', 'new', 'continue', 'inflation', 'trump', 'economy', 'stock', 'market', 'gold', 'oil', 'natural', 'gas', 'china', 'street', 'breakfast', 'wall', 'street', 'fed', 'dollar'])
ps = PorterStemmer()

"""""""""""""""""""""""""DATA COLLECTION"""""""""""""""""""""""""

"""EXTRACT TICKERS FROM WIKI"""
def save_sp500_tickers():
    r = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    soup = BeautifulSoup(r.text, "html.parser")

    stock_tag = soup.find("table", class_="wikitable sortable")

    tickers = []
    for row in stock_tag.find_all("tr")[1:]:
        ticker = row.find_all("td")[0].text
        ticker = ticker[:-1] #to remove the new line
        mapping = str.maketrans(".", "-")
        ticker = ticker.translate(mapping)
        tickers.append(ticker)

    with open("sp500tickers.pickle","wb") as f:
        pickle.dump(tickers, f)
# save_sp500_tickers()

"""CREATE TICKER LIST"""
with open("sp500tickers_nlp.pickle", "rb") as f:
    stock = pickle.load(f)
t = np.array(stock)[:, 0]
ticker = []
for i in t:
    mapping = str.maketrans(".", "-")
    i = i.translate(mapping)
    ticker.append(i)

"""GET TOP NEWS"""
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

"""GET MARKET OUTLOOK"""
def get_market_outlook():
    headlines, dates, tickers, total = [], [], [], []
    for i in range(1000, 1040):
        i += 1
        url = f"https://seekingalpha.com/market-outlook?page={str(i)}"
        headers = {
            "content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
            "Origin": "https://seekingalpha.com",
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.116 Safari/537.36"}

        response = requests.get(url=url, headers=headers)
        while response.status_code != 200:
            print(url, response.status_code)
            response = requests.get(url=url, headers=headers)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            for script in soup.find_all('div', class_="media-body"):
                for i in script.find_all('a', class_="a-title"):
                    headlines.append(i.text)
                for j in script.find_all('span'):
                    if re.search(r'\d\b\sAM\b', j.text) or re.search(r'\d\b\sPM\b', j.text):
                        dates.append(j.text)

    for k in range(len(headlines)):
        total.append((headlines[k], dates[k], ''))
    with open("headlines_mkt_1001_1040.pickle", "wb") as f:
        pickle.dump(total, f)
# get_market_outlook()

with open(f"headlines_mkt_1_100.pickle", "rb") as f:
    cluster_news_1 = pickle.load(f)
with open(f"headlines_mkt_101_200.pickle", "rb") as f:
    cluster_news_2 = pickle.load(f)
with open(f"headlines_mkt_201_300.pickle", "rb") as f:
    cluster_news_3 = pickle.load(f)
with open(f"headlines_mkt_301_400.pickle", "rb") as f:
    cluster_news_4 = pickle.load(f)
with open(f"headlines_mkt_401_500.pickle", "rb") as f:
    cluster_news_5 = pickle.load(f)
with open(f"headlines_mkt_501_600.pickle", "rb") as f:
    cluster_news_6 = pickle.load(f)
with open(f"headlines_mkt_601_700.pickle", "rb") as f:
    cluster_news_7 = pickle.load(f)
with open(f"headlines_mkt_701_800.pickle", "rb") as f:
    cluster_news_8 = pickle.load(f)
with open(f"headlines_mkt_801_900.pickle", "rb") as f:
    cluster_news_9 = pickle.load(f)
with open(f"headlines_mkt_901_1000.pickle", "rb") as f:
    cluster_news_10 = pickle.load(f)
with open(f"headlines_mkt_1001_1040.pickle", "rb") as f:
    cluster_news_11 = pickle.load(f)

df_news = pd.DataFrame(cluster_news_1, columns=('Headlines', 'Dates', 'Ticker'))
df_news = df_news.append(pd.DataFrame(cluster_news_2, columns=('Headlines', 'Dates', 'Ticker')), ignore_index=True)
df_news = df_news.append(pd.DataFrame(cluster_news_3, columns=('Headlines', 'Dates', 'Ticker')), ignore_index=True)
df_news = df_news.append(pd.DataFrame(cluster_news_4, columns=('Headlines', 'Dates', 'Ticker')), ignore_index=True)
df_news = df_news.append(pd.DataFrame(cluster_news_5, columns=('Headlines', 'Dates', 'Ticker')), ignore_index=True)
df_news = df_news.append(pd.DataFrame(cluster_news_6, columns=('Headlines', 'Dates', 'Ticker')), ignore_index=True)
df_news = df_news.append(pd.DataFrame(cluster_news_7, columns=('Headlines', 'Dates', 'Ticker')), ignore_index=True)
df_news = df_news.append(pd.DataFrame(cluster_news_8, columns=('Headlines', 'Dates', 'Ticker')), ignore_index=True)
df_news = df_news.append(pd.DataFrame(cluster_news_9, columns=('Headlines', 'Dates', 'Ticker')), ignore_index=True)
df_news = df_news.append(pd.DataFrame(cluster_news_10, columns=('Headlines', 'Dates', 'Ticker')), ignore_index=True)
df_news = df_news.append(pd.DataFrame(cluster_news_11, columns=('Headlines', 'Dates', 'Ticker')), ignore_index=True)



"""""""""""""""""""""""""DATA PRE-PROCESSING"""""""""""""""""""""""""

"""CHANGE THE DATE FORMAT AND MOVE NEWS TO NEXT DAY AFTER 16:00"""
def change_dates_tocsv(dataframe, today='', yesterday=''):
    # df1 = pd.DataFrame(headlines, columns=('Headlines', 'Dates', 'Ticker'))
    df1 = dataframe
    df1.iloc[:, 0].replace(['^a-zA-Z'], ' ', regex=True, inplace=True)
    # print(len(df1['Headlines']))
    # df1 = df1.drop_duplicates()
    # df1 = df1.sort_values(by='Dates')
    # df1.reset_index(drop=True, inplace=True)
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
    df1.to_csv('Top_News_4.csv', index=False)
# change_dates_tocsv(df_news, today=datetime.date.today().strftime('%b %d, %Y'), yesterday=(datetime.date.today()-datetime.timedelta(days=1)).strftime('%b %d, %Y'))


"""COMBINE TOP NEWS AND MARKET OUTLOOK: Top_News_2.csv is Top News, Top_News_4.csv is Market Outlook"""
df0 = pd.read_csv('Top_News_2.csv')
df0.iloc[:, 0].replace(['^a-zA-Z'], ' ', regex=True, inplace=True)
df1 = pd.read_csv('Top_News_4.csv')
df1.iloc[:, 0].replace(['^a-zA-Z'], ' ', regex=True, inplace=True)
df1 = df1.append(df0)
df1 = df1.sort_values(by='Dates')
df1.reset_index(drop=True, inplace=True)

"""GET S&P500 CLOSING PRICES (^GSPC)"""
def create_price(start, end):
    df_stock = web.DataReader('^GSPC', 'yahoo', start, end)
    df_stock.to_csv('^GSPC_2.csv')
    df2 = pd.read_csv('^GSPC_2.csv', index_col=0)
    df2 = df2.drop(['Open', 'High', 'Low', 'Close', 'Volume'], axis=1)  # drop unwanted rows
    df2['Return'] = df2['Adj Close'] / df2['Adj Close'].shift(1) - 1
    df2['Label'] = ''
    df_vix = web.DataReader('^VIX', 'yahoo', start, end)
    df_vix.to_csv('^VIX.csv')
    df2['t_VIX'] = df_vix['Adj Close']/df_vix['Adj Close'].shift(1) - 1
    df2['VIX'] = df2['t_VIX'].shift(31)

    for i in range(len(df2['Return'])):
        if df2['Return'].iloc[i] >= 0:
            df2['Label'].iloc[i] = 0
        elif df2['Return'].iloc[i] < 0:
            df2['Label'].iloc[i] = 1
        else:
            df2['Label'].iloc[i] = np.nan
    return df2
start = datetime.datetime(2015, 10, 20)
end = datetime.datetime(2020, 6, 22)
df2 = create_price(start, end)

"""CREATE THE DATES, HEADLINES, TICKER, AND LABEL (LABEL 0 IS UP AND 1 IS DOWN, CALCULATED BY DAILY RETURN)"""
def stopwords_and_stemmimg(df, df2):
    df = df.drop_duplicates()
    df = df.sort_values(by='Dates')
    df.reset_index(drop=True, inplace=True)
    date_dict = {}
    data_for_df = []
    for i in range(len(df['Dates'])):
        date_dict[df['Dates'].iloc[i]] = []

    for i in range(len(df['Headlines'])):
        news = df['Headlines'].iloc[i]
        if '?' not in news:
            word_tokens = nltk.word_tokenize(news.lower())
            filtered_sentence = [w for w in word_tokens if w not in stop_words]
            stemmed_sentence = [ps.stem(w) for w in filtered_sentence]
            news = ' '.join(stemmed_sentence)
            date_dict[df['Dates'].iloc[i]].append(news)

    for i in date_dict:
        data_for_df.append((' '.join(date_dict[i]), i))

    with open("data_for_df.pickle", "wb") as f:
        pickle.dump(data_for_df, f)

    with open("data_for_df.pickle", "rb") as f:
        data_for_df = pickle.load(f)
    df3 = pd.DataFrame(data_for_df)
    df3.columns = ['Headlines', 'Dates']
    df3.iloc[:, 0].replace(['^a-zA-Z'], ' ', regex=True, inplace=True)
    df3 = df3.sort_values(by='Dates')
    df3.set_index('Dates', inplace=True)

    df4 = pd.merge(df3['Headlines'], df2[['Label', 'Return', 'VIX']], left_index=True, right_index=True, how='left')
    df4['New_Headlines'] = df4['Headlines'].shift(1)
    df4.dropna(inplace=True)
    df4['Label'] = df4['Label'].astype('int')
    df4.reset_index(inplace=True)
    return df4

"""USE FOE ROBERTA TOKENIZER DOING WORD EMBEDDING"""
# tokenizer = tokenizers.ByteLevelBPETokenizer(
#     vocab_file='/Users/ccw/PycharmProjects/Wilfred/tf-roberta/roberta-base-vocab.json',
#     merges_file='/Users/ccw/PycharmProjects/Wilfred/tf-roberta/roberta-base-merges.txt',
#     lowercase=True,
#     add_prefix_space=True)
#
# def stopwords_and_stemmimg(df, df2):
#     date_dict = {}
#     data_for_df = []
#     for i in range(len(df['Dates'])):
#         date_dict[df['Dates'].iloc[i]] = []
#
#     for i in range(len(df['Headlines'])):
#         news = df['Headlines'].iloc[i]
#         if '?' not in news:
#             word_tokens = nltk.word_tokenize(news.lower())
#             filtered_sentence = [w for w in word_tokens if w not in stop_words]
#             stemmed_sentence = [ps.stem(w) for w in filtered_sentence]
#             news = ' '.join(stemmed_sentence)
#             enc = tokenizer.encode(news)
#             date_dict[df['Dates'].iloc[i]].append(enc.ids)
#             # date_dict[df['Dates'].iloc[i]].append(tf.keras.preprocessing.sequence.pad_sequences([enc.ids], maxlen=50, padding='post', dtype=np.float)[0])
#
#     # for i in date_dict:
#     #     print(len(date_dict[i]))
#
#     for i in date_dict:
#         data_for_df.append((date_dict[i], i))
#     with open("data_for_df.pickle", "wb") as f:
#         pickle.dump(data_for_df, f)
#
#     with open("data_for_df.pickle", "rb") as f:
#         data_for_df = pickle.load(f)
#     df3 = pd.DataFrame(data_for_df)
#     df3.columns = ['Headlines', 'Dates']
#     # df3.iloc[:, 0].replace(['^a-zA-Z'], ' ', regex=True, inplace=True)
#     df3 = df3.sort_values(by='Dates')
#     df3.set_index('Dates', inplace=True)
#
#     df4 = pd.merge(df3['Headlines'], df2['Label'], left_index=True, right_index=True, how='left')
#     df4['New_Headlines'] = df4['Headlines'].shift(1)
#     df4.dropna(inplace=True)
#     df4['Label'] = df4['Label'].astype('int')
#     df4.reset_index(inplace=True)
#     return df4

"""PLOT THE DISTRIBUTION OF TICKER SHOWN IN HEADLINES"""
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

"""CREATE CLUSTERS, TO SEE HOW WORD EMBEDDING DONE BY TFID VECTORIZER TO GROUP THE NEWS INTO CLUSTERS"""
def create_total_news():
    total_news = []
    for i in range(len(df1['Headlines'])):
        news = nltk.word_tokenize(df1['Headlines'].iloc[i].lower())
        filtered_sentence = [w for w in news if w not in stop_words]
        stemmed_sentence = [ps.stem(w) for w in filtered_sentence]
        news = ' '.join(stemmed_sentence)
        total_news.append(news)
    with open("total_news_4.pickle", "wb") as f:
        pickle.dump(total_news, f)
# create_total_news()

with open("total_news_4.pickle", "rb") as f:
    total_news = pickle.load(f)
def create_Kmanes():
    # countvector = CountVectorizer(ngram_range=(1, 2), max_df=0.4, max_features=10000, stop_words=stop_words)
    tfidfconverter = TfidfVectorizer(ngram_range=(1, 1), max_features=12000, max_df=0.4, stop_words=stop_words)
    traindataset = tfidfconverter.fit_transform(total_news)

    svd = TruncatedSVD(n_components=120)
    xtrain_svd = svd.fit_transform(traindataset)
    """ELBOW PLOT"""
    # mapping2 = {}
    # for k in range(1, 11):
    #     kmeanModel = KMeans(n_clusters=k)
    #     kmeanModel.fit(traindataset)
    #     mapping2[k] = kmeanModel.inertia_
    # for key, val in mapping2.items():
    #     print(str(key) + ' : ' + str(val))
    # plt.plot(list(mapping2.keys()), list(mapping2.values()), 'bx-')
    # plt.xlabel('Values of K')
    # plt.ylabel('Inertia')
    # plt.title('The Elbow Method using Inertia')
    # plt.show()

    clf = KMeans(n_clusters=100)
    clf.fit(xtrain_svd)
    labels = clf.labels_
    centroids = clf.cluster_centers_
    with open("labels_4.pickle", "wb") as f:
        pickle.dump(labels, f)
    with open("centroids_4.pickle", "wb") as f:
        pickle.dump(centroids, f)
# create_Kmanes()

with open("labels_4.pickle", "rb") as f:
    labels = pickle.load(f)
with open("centroids_4.pickle", "rb") as f:
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
    with open("cluster_news_top_news_4.pickle", "wb") as f:
        pickle.dump(cluster_news, f)
# create_cluster_news()

news = []
def show_cluster_news():
    with open("cluster_news_top_news_4.pickle", "rb") as f:
        cluster_news = pickle.load(f)
    tree = BallTree(centroids)
    dist, ind = tree.query(centroids, k=2)
    # print(ind)
    for i in ind:
        if 2 not in i:
            for index in i:
                for j in cluster_news[index]:
                    # print(index, df1['Headlines'][j], df1['Dates'][j])
                    news.append((df1['Headlines'][j], df1['Dates'][j], ''))
show_cluster_news()
df5 = pd.DataFrame(news, columns=('Headlines', 'Dates', 'Ticker'))
df5.iloc[:, 0].replace(['^a-zA-Z'], ' ', regex=True, inplace=True)

"""RUN stopwords_and_stemmimg FUNCTION, THEN SEPARATING THE DATASET USED FOR TRAINING AND TESTING"""
df = stopwords_and_stemmimg(df5, df2)
traindata = df.iloc[:int(len(df['Headlines'])*0.8)]
testdata = df.iloc[int(len(df['Headlines'])*0.8):]

"""PLOT THE WORDCLOUD FOE UP AND DOWN MOVEMENT"""
# stopwords1 = set(STOPWORDS)
# stopwords1.update(['may', 'new', 'price', 'growth', 'investor', 'year', 'outlook', 'trade', 'rally', 'still', 'global', 'future', 'higher', 'continue', 'inflation', 'trump', 'economy', 'stock', 'market', 'gold', 'oil', 'natural', 'gas', 'china', 'street', 'breakfast', 'wall', 'street', 'fed', 'dollar'])
# wordcloud = WordCloud(stopwords=stopwords1).generate(' '.join(df['New_Headlines'].loc[df['Label']==0]))
# wordcloud2 = WordCloud(stopwords=stopwords1).generate(' '.join(df['New_Headlines'].loc[df['Label']==1]))
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.imshow(wordcloud2, interpolation='bilinear')
# plt.axis("off")
# plt.show()
# print(wordcloud.words_)



"""""""""""""""""""""""""DATA EVALUATION"""""""""""""""""""""""""

"""USE WHEN ROBERTA TOKENIZER IS USED FOR TOKENIZING AND WORD EMBEDDING"""
# dataset = []
# for i in df['New_Headlines'].values:
#     dataset.append(tf.keras.preprocessing.sequence.pad_sequences([np.concatenate(i)], maxlen=1300, padding='post', dtype=np.float)[0])
# dataset = np.array(dataset).reshape(-1, 1300)
# xtrain, xtest, ytrain, ytest = train_test_split(dataset, df['Label'], test_size=0.2)


"""GRID SEARCH FOR RANDOM FOREST"""
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
#     'tfidf__max_features': (10000, 12000, 20000),
#     'tfidf__ngram_range': ((1, 1), (1, 2), (2, 2)),
#     'tfidf__stop_words': [stop_words],
#     'clf__n_estimators': (1000, 2000, 4000),
#     'clf__criterion': ['entropy']
# }
#
# grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1, cv=3, scoring='accuracy')
#
# grid_search.fit(traindata['Headlines'], traindata['Label'])
# print("Best score: %0.3f" % grid_search.best_score_)
# print("Best parameters set:")
# best_parameters = grid_search.best_estimator_.get_params()
# for param_name in sorted(parameters.keys()):
#     print("\t%s: %r" % (param_name, best_parameters[param_name]))

"""RANDOM FOREST FOR SKLEARN VECTORIZER"""
# countvector = CountVectorizer(ngram_range=(1, 2), max_df=0.4, max_features=10000, stop_words=stop_words)
tfidfconverter = TfidfVectorizer(ngram_range=(1, 1), max_features=20000, max_df=0.5, stop_words=stop_words)
column_trans = ColumnTransformer([('VIX_vector', 'passthrough', ['VIX']), ('New_Headlines_vector', tfidfconverter, 'New_Headlines')], remainder='drop')
traindataset = column_trans.fit_transform(traindata)
randomclassifier = RandomForestClassifier(n_estimators=1000, criterion='entropy')
randomclassifier.fit(traindataset, traindata['Label'])
testdataset = column_trans.transform(testdata)
predictions = randomclassifier.predict(testdataset)
predictions_prob = randomclassifier.predict_proba(testdataset)
accuarcy = accuracy_score(testdata['Label'], predictions)
classification = classification_report(testdata['Label'], predictions)
matrix = confusion_matrix(testdata['Label'], predictions)
logloss = log_loss(testdata['Label'], predictions_prob, eps=1e-15)
print(accuarcy)
print(classification)
print(matrix)
print(logloss)


# principal = 100000
# true = 0
# false = 0
# acc_return = []
# for i in range(len(predictions)):
#     if predictions[i] == testdata['Label'].iloc[i] and predictions[i] == 0:
#         principal = principal*(1+testdata['Return'].iloc[i])
#         print('Correct Up: ', principal, testdata['Dates'].iloc[i], testdata['Return'].iloc[i], predictions_prob[i])
#         # true += 1
#     elif predictions[i] != testdata['Label'].iloc[i] and predictions[i] == 0:
#         if testdata['Return'].iloc[i] > -0.05:
#             principal = principal * (1 + testdata['Return'].iloc[i])
#         else:
#             principal = principal * (1 - 0.05)
#         print('Wrong Up: ', principal, testdata['Dates'].iloc[i], testdata['Return'].iloc[i], predictions_prob[i])
#         # false += 1
#     elif predictions[i] == testdata['Label'].iloc[i] and predictions[i] == 1:
#         principal = principal * (1 - testdata['Return'].iloc[i])
#         print('Correct Down: ', principal, testdata['Dates'].iloc[i], testdata['Return'].iloc[i], predictions_prob[i])
#         # true += 1
#     elif predictions[i] != testdata['Label'].iloc[i] and predictions[i] == 1:
#         if testdata['Return'].iloc[i] < 0.05:
#             principal = principal * (1 - testdata['Return'].iloc[i])
#         else:
#             principal = principal * (1 - 0.05)
#         print('Wrong Down: ', principal, testdata['Dates'].iloc[i], testdata['Return'].iloc[i], predictions_prob[i])
#         # false += 1
#
#     acc_return.append(principal)
# print(principal)
# fig, ax1 = plt.subplots()
# testdata['Dates']= pd.to_datetime(testdata['Dates'])
# ax1.plot(testdata['Dates'], acc_return)
# ax1.xaxis.set_major_locator(matplotlib.dates.MonthLocator([1, 4, 7, 10]))
# ax1.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y-%m'))
# plt.show()

"""RANDOM FOREST FOR ROBERTA"""
# randomclassifier = RandomForestClassifier(n_estimators=1000, criterion='entropy')
# randomclassifier.fit(xtrain, ytrain)
#
# predictions = randomclassifier.predict(xtest)
# predictions_prob = randomclassifier.predict_proba(xtest)
# accuarcy = accuracy_score(ytest, predictions)
# classification = classification_report(ytest, predictions)
# matrix = confusion_matrix(ytest, predictions)
# logloss = log_loss(ytest, predictions_prob, eps=1e-15)
# print(accuarcy)
# print(classification)
# print(matrix)
# print(logloss)


"""GRID SEARCH FOR MLP"""
# def create_model():
#     model = tf.keras.models.Sequential()
#     model.add(tf.keras.layers.Dense(units=1024, activation='relu', kernel_initializer='he_uniform'))
#     model.add(tf.keras.layers.Dense(units=64, activation='relu'))
#     model.add(tf.keras.layers.Dense(units=1, activation='sigmoid',))
#     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#     return model
#
# model = KerasClassifier(build_fn=create_model, verbose=0)
# batch_size = [10, 20, 40, 60, 80, 100]
# epochs = [10, 20, 50, 100]
# param_grid = dict(batch_size=batch_size, epochs=epochs)
# grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3, scoring='accuracy')
# grid_result = grid.fit(traindataset.toarray(), traindata['Label'])
#
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

"""MLP FOR SKLEARN VECTORIZER"""
# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Dense(units=256, activation='relu', kernel_initializer='he_uniform'))
# model.add(tf.keras.layers.Dropout(0.5))
# model.add(tf.keras.layers.Dense(units=64, activation='relu'))
# model.add(tf.keras.layers.Dropout(0.5))
# model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
#
# tensorboard = TensorBoard(log_dir="logs/{}".format('MLP_2'))
# adam = tf.keras.optimizers.Adam(lr=3e-6)
# model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
# model.fit(traindataset.toarray(), traindata['Label'], epochs=300, batch_size=32, verbose=1, validation_split=0.2, shuffle=True, callbacks=[tensorboard])
#
# val_loss, val_acc = model.evaluate(testdataset.toarray(), testdata['Label'])
# print(val_acc)
# print((model.predict(testdataset)).shape)

"""MLP FOR ROBERTA"""
# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Dense(units=256, input_shape=(1300,), activation='relu', kernel_initializer='he_uniform'))
# model.add(tf.keras.layers.Dropout(0.5))
# model.add(tf.keras.layers.Dense(units=64, activation='relu'))
# model.add(tf.keras.layers.Dropout(0.5))
# model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
#
# tensorboard = TensorBoard(log_dir="logs/{}".format('MLP_2'))
# adam = tf.keras.optimizers.Adam()#lr=3e-6)
# model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
# model.fit(xtrain, ytrain, epochs=100, batch_size=64, verbose=1, validation_split=0.2, shuffle=True, callbacks=[tensorboard])
#
# val_loss, val_acc = model.evaluate(xtest, ytest)
# print(val_acc)
# print((model.predict(xtest)).shape)


"""GRID SEARCH FOR LSTM"""
# def create_model():
#     model = tf.keras.models.Sequential()
#     model.add(tf.keras.layers.LSTM(units=256, input_shape=(1, traindataset.toarray().shape[1]), return_sequences=True))
#     model.add(tf.keras.layers.LSTM(units=256, input_shape=(1, traindataset.toarray().shape[1])))
#     model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
#     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#     return model
#
# model = KerasClassifier(build_fn=create_model, verbose=0)
# batch_size = [10, 20, 40, 60, 80, 100]
# epochs = [10, 20, 50, 100]
# param_grid = dict(batch_size=batch_size, epochs=epochs)
# grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3, scoring='accuracy')
# grid_result = grid.fit(np.array(traindataset.toarray()).reshape(traindataset.toarray().shape[0], 1, traindataset.toarray().shape[1]), traindata['Label'])
#
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

"""LSTM FOR SKLEARN VECTORIZER"""
# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.LSTM(units=256, input_shape=(1, traindataset.toarray().shape[1]), return_sequences=True))
# model.add(tf.keras.layers.LSTM(units=256, input_shape=(1, traindataset.toarray().shape[1])))
# model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
#
# adam = tf.keras.optimizers.Adam(lr=1e-5)
# model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
# # tensorboard = TensorBoard(log_dir="logs/{}".format('LSTM'))
# model.fit(np.array(traindataset.toarray()).reshape(traindataset.toarray().shape[0], 1, traindataset.toarray().shape[1]), traindata['Label'], epochs=50, batch_size=32, validation_split=0.2, verbose=1, shuffle=False)
#
# val_loss, val_acc = model.evaluate(np.array(testdataset.toarray()).reshape(testdataset.toarray().shape[0], 1, testdataset.toarray().shape[1]), testdata['Label'])
# print(val_acc)

"""LSTM FOR ROBERTA"""
# dataset = np.array(dataset).reshape(-1, 1300, 1)
# xtrain, xtest, ytrain, ytest = train_test_split(dataset, df['Label'], test_size=0.2)
# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.LSTM(units=62, input_shape=(xtrain.shape[1], xtrain.shape[2]), return_sequences=True))
# model.add(tf.keras.layers.LSTM(units=64, input_shape=(xtrain.shape[1], xtrain.shape[2])))
# model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
#
# adam = tf.keras.optimizers.Adam()#lr=1e-5)
# model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
# tensorboard = TensorBoard(log_dir="logs/{}".format('LSTM_2'))
# model.fit(xtrain, ytrain, epochs=10, batch_size=32, validation_split=0.2, verbose=1, shuffle=False, callbacks=[tensorboard])
#
# val_loss, val_acc = model.evaluate(xtest, ytest)
# print(val_acc)


"""SAVE THE MODEL OF RANDOM FOREST AS IT PROVIDES BEST RESULT"""
filename = 'finalized_model.sav'
# pickle.dump(randomclassifier, open(filename, 'wb'))
# pickle.dump(tfidfconverter,open('feature.pkl', 'wb'))







"""TESTING FOR GETTING ONE-DAY NEWS HEADLINES"""
def testing_get_news():
    headlines, dates, tickers, total = [], [], [], []
    for i in range(0, 15):
        i += 1
        url = f"https://seekingalpha.com/market-outlook?page={str(i)}"
        headers = {
            "content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
            "Origin": "https://seekingalpha.com",
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.116 Safari/537.36"}

        response = requests.get(url=url, headers=headers)
        while response.status_code != 200:
            print(url, response.status_code)
            response = requests.get(url=url, headers=headers)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            for script in soup.find_all('div', class_="media-body"):
                for i in script.find_all('a', class_="a-title"):
                    headlines.append(i.text)
                for j in script.find_all('span'):
                    if re.search(r'\d\b\sAM\b', j.text) or re.search(r'\d\b\sPM\b', j.text):
                        dates.append(j.text)
    for i in range(3):
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

    for k in range(len(headlines)):
        total.append((headlines[k], dates[k], ''))
    return total

def testing(start, end, today, yesterday):
    news = testing_get_news()
    # ytd_news = [i for i in news]
    ytd_news = pd.DataFrame(news, columns=('Headlines', 'Dates', 'Ticker'))
    change_dates_tocsv(ytd_news, today, yesterday)
    df1 = pd.read_csv('Top_News_4.csv')
    df1.iloc[:, 0].replace(['^a-zA-Z'], ' ', regex=True, inplace=True)
    df2 = create_price(start, end)
    # print(df2)
    df = stopwords_and_stemmimg(df1, df2)
    # for i in range(len(df['Label'])):
    #     print('Actual: ', df['Label'].iloc[i], df['New_Headlines'].iloc[i])
    tfidfconverter = pickle.load(open('feature.pkl', 'rb'))
    dataset = tfidfconverter.transform(df['New_Headlines'])
    loaded_model = pickle.load(open(filename, 'rb'))
    predictions_prob = loaded_model.predict_proba(dataset)
    predictions = loaded_model.predict(dataset)
    # print('Predict probability: ', predictions)
    # print('Predict: ', predictions_prob)

    principal = 100000
    true = 0
    false = 0
    acc_return = []
    for i in range(len(predictions)):
        if predictions[i] == df['Label'].iloc[i] and predictions[i] == 0:
            principal = principal * (1 + df['Return'].iloc[i])
            print('Correct Up: ', principal, df['Dates'].iloc[i], df['Return'].iloc[i], predictions_prob[i])
            # true += 1
        elif predictions[i] != df['Label'].iloc[i] and predictions[i] == 0:
            if df['Return'].iloc[i] > -0.05:
                principal = principal * (1 + df['Return'].iloc[i])
            else:
                principal = principal * (1 - 0.05)
            print('Wrong Up: ', principal, df['Dates'].iloc[i], df['Return'].iloc[i], predictions_prob[i])
            # false += 1
        elif predictions[i] == df['Label'].iloc[i] and predictions[i] == 1:
            principal = principal * (1 - df['Return'].iloc[i])
            print('Correct Down: ', principal, df['Dates'].iloc[i], df['Return'].iloc[i],
                  predictions_prob[i])
            # true += 1
        elif predictions[i] != df['Label'].iloc[i] and predictions[i] == 1:
            if df['Return'].iloc[i] < 0.05:
                principal = principal * (1 - f['Return'].iloc[i])
            else:
                principal = principal * (1 - 0.05)
            print('Wrong Down: ', principal, df['Dates'].iloc[i], df['Return'].iloc[i], predictions_prob[i])
            # false += 1

        acc_return.append(principal)
    print(principal)
    fig, ax1 = plt.subplots()
    df['Dates'] = pd.to_datetime(df['Dates'])
    ax1.plot(df['Dates'], acc_return)
    ax1.xaxis.set_major_locator(matplotlib.dates.MonthLocator([1, 4, 7, 10]))
    ax1.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y-%m'))
    plt.show()
# start = datetime.datetime(2020, 6, 22)
# end = datetime.datetime(2020, 7, 10)
# today = datetime.date.today().strftime('%b %d, %Y')
# yesterday = (datetime.date.today()-datetime.timedelta(days=1)).strftime('%b %d, %Y')
# testing(start, end, today, yesterday)

