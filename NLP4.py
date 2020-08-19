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
class web_scraping():
    def save_sp500_tickers(self, web):
        r = requests.get(web)
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

    def get_top_news(self, web2):
        headlines, dates, tickers, total = [], [], [], []
        for i in range(62):
            i += 1
            url = web2 + (f"{str(i)}")
            headers = {
                "content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
                "Origin": "https://seekingalpha.com",
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36"}
            response = requests.get(url=url, headers=headers)
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
        with open("headlines_top_news.pickle", "wb") as f:
            pickle.dump(total, f)

    def get_market_outlook(self, web3):
        headlines, dates, tickers, total = [], [], [], []
        for i in range(1000, 1040):
            i += 1
            url = web3 + {str(i)}
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

did_web = web_scraping()
# did_web.save_sp500_tickers("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
# did_web.get_top_news("https://seekingalpha.com/market-news/top-news?page=")
# did_web.get_market_outlook(f"https://seekingalpha.com/market-outlook?page=")

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
with open("headlines_top_news.pickle", "rb") as f:
    headlines = pickle.load(f)



"""""""""""""""""""""""""DATA PRE-PROCESSING"""""""""""""""""""""""""

"""CHANGE THE DATE FORMAT AND MOVE NEWS TO NEXT DAY AFTER 16:00"""
"""GET MARKET OUTLOOK"""
class mkt_outlook_transform():
    def get_from_pickle(self):
        i = 100
        while i < 1040:
            if i == 100:
                with open(f"headlines_mkt_{str(i - 99)}_{str(i)}.pickle", "rb") as f:
                    cluster_news = pickle.load(f)
                    df_news = pd.DataFrame(cluster_news, columns=("Headlines", "Dates", "Ticker"))
                i += 100
            else:
                with open(f"headlines_mkt_{str(i - 99)}_{str(i)}.pickle", "rb") as f:
                    cluster_news = pickle.load(f)
                    df_news = df_news.append(pd.DataFrame(cluster_news, columns=("Headlines", "Dates", "Ticker")), ignore_index=True)
                i += 100
                if i == 1100:
                    i = i - 60
                    with open(f"headlines_mkt_{str(i-39)}_{str(i)}.pickle", "rb") as f:
                        cluster_news = pickle.load(f)
                        df_news = df_news.append(pd.DataFrame(cluster_news, columns=("Headlines", "Dates", "Ticker")), ignore_index=True)
        return df_news

    def change_dates_tocsv(self, method, df_news='', today='', yesterday=''):
        if method == "training":
            df1 = self.get_from_pickle()
        else:
            df1 = df_news
        df1.iloc[:, 0].replace(['^a-zA-Z'], ' ', regex=True, inplace=True)
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

    """GET S&P500 ETF CLOSING PRICES (SPY)"""
    def create_price(self, start, end):
        df_stock = web.DataReader('SPY', 'yahoo', start, end)
        df_stock.to_csv('SPY.csv')
        df2 = pd.read_csv('SPY.csv', index_col=0)
        df2 = df2.drop(['Open', 'High', 'Low', 'Close', 'Volume'], axis=1)  # drop unwanted rows
        df2['Return'] = df2['Adj Close'] / df2['Adj Close'].shift(1) - 1
        df2['Label'] = ''
        df_vix = web.DataReader('^VIX', 'yahoo', start, end)
        df_vix.to_csv('^VIX.csv')
        df2['t_VIX'] = df_vix['Adj Close'] / df_vix['Adj Close'].shift(1) - 1
        df2['VIX'] = df2['t_VIX'].shift(31)

        for i in range(len(df2['Return'])):
            if df2['Return'].iloc[i] >= 0:
                df2['Label'].iloc[i] = 0
            elif df2['Return'].iloc[i] < 0:
                df2['Label'].iloc[i] = 1
            else:
                df2['Label'].iloc[i] = np.nan
        return df2


create_csv = mkt_outlook_transform()
# create_csv.change_dates_tocsv(method='training', today=datetime.date.today().strftime('%b %d, %Y'), yesterday=(datetime.date.today()-datetime.timedelta(days=1)).strftime('%b %d, %Y'))

"""COMBINE TOP NEWS AND MARKET OUTLOOK: Top_News_2.csv is Top News, Top_News_4.csv is Market Outlook"""
df0 = pd.read_csv('Top_News_2.csv')
df0.iloc[:, 0].replace(['^a-zA-Z'], ' ', regex=True, inplace=True)
df1 = pd.read_csv('Top_News_4.csv')
df1.iloc[:, 0].replace(['^a-zA-Z'], ' ', regex=True, inplace=True)
df1 = df1.append(df0)
df1 = df1.sort_values(by='Dates')
df1.reset_index(drop=True, inplace=True)
start = datetime.datetime(2015, 10, 20)
end = datetime.datetime(2020, 6, 22)
df2 = create_csv.create_price(start, end)


"""CREATE THE DATES, HEADLINES, TICKER, AND LABEL (LABEL 0 IS UP AND 1 IS DOWN, CALCULATED BY DAILY RETURN)"""
class combine_news_prices():
    def stopwords_and_stemmimg(self, df, df2):
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

        df4 = pd.merge(df3['Headlines'], df2[['Label', 'Return', 'VIX', 'Adj Close']], left_index=True, right_index=True, how='left')
        df4['New_Headlines'] = df4['Headlines'].shift(1)
        df4.dropna(inplace=True)
        df4['Label'] = df4['Label'].astype('int')
        df4.reset_index(inplace=True)
        return df4

    """USE FOE ROBERTA TOKENIZER DOING WORD EMBEDDING"""
    def create_tokenizer(self):
        tokenizer = tokenizers.ByteLevelBPETokenizer(
            vocab_file='/Users/ccw/PycharmProjects/Wilfred/tf-roberta/roberta-base-vocab.json',
            merges_file='/Users/ccw/PycharmProjects/Wilfred/tf-roberta/roberta-base-merges.txt',
            lowercase=True,
            add_prefix_space=True)
        return tokenizer

    def stopwords_and_stemmimg_roberta(self, df, df2):
        tokenizer = self.create_tokenizer
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
                enc = tokenizer.encode(news)
                date_dict[df['Dates'].iloc[i]].append(enc.ids)
                # date_dict[df['Dates'].iloc[i]].append(tf.keras.preprocessing.sequence.pad_sequences([enc.ids], maxlen=50, padding='post', dtype=np.float)[0])

        # for i in date_dict:
        #     print(len(date_dict[i]))

        for i in date_dict:
            data_for_df.append((date_dict[i], i))
        with open("data_for_df.pickle", "wb") as f:
            pickle.dump(data_for_df, f)

        with open("data_for_df.pickle", "rb") as f:
            data_for_df = pickle.load(f)
        df3 = pd.DataFrame(data_for_df)
        df3.columns = ['Headlines', 'Dates']
        # df3.iloc[:, 0].replace(['^a-zA-Z'], ' ', regex=True, inplace=True)
        df3 = df3.sort_values(by='Dates')
        df3.set_index('Dates', inplace=True)

        df4 = pd.merge(df3['Headlines'], df2['Label'], left_index=True, right_index=True, how='left')
        df4['New_Headlines'] = df4['Headlines'].shift(1)
        df4.dropna(inplace=True)
        df4['Label'] = df4['Label'].astype('int')
        df4.reset_index(inplace=True)
        return df4

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
class clustering():
    def create_total_news(self):
        total_news = []
        for i in range(len(df1['Headlines'])):
            news = nltk.word_tokenize(df1['Headlines'].iloc[i].lower())
            filtered_sentence = [w for w in news if w not in stop_words]
            stemmed_sentence = [ps.stem(w) for w in filtered_sentence]
            news = ' '.join(stemmed_sentence)
            total_news.append(news)
        return total_news

    def create_Kmanes(self):
        total_news = self.create_total_news()
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

    def create_cluster_news(self):
        with open("labels_4.pickle", "rb") as f:
            labels = pickle.load(f)
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

    def show_cluster_news(self):
        news = []
        with open("cluster_news_top_news_4.pickle", "rb") as f:
            cluster_news = pickle.load(f)
        with open("centroids_4.pickle", "rb") as f:
            centroids = pickle.load(f)
        tree = BallTree(centroids)
        dist, ind = tree.query(centroids, k=2)
        # print(ind)
        for i in ind:
            if 81 not in i:
                # print(i)
                for index in i:
                    for j in cluster_news[index]:
                        # print(index, df1['Headlines'][j], df1['Dates'][j])
                        news.append((df1['Headlines'][j], df1['Dates'][j], ''))
        return news
cluster = clustering()
# cluster.create_Kmanes()
# cluster.create_cluster_news()
news = cluster.show_cluster_news()
df5 = pd.DataFrame(news, columns=('Headlines', 'Dates', 'Ticker'))
df5.iloc[:, 0].replace(['^a-zA-Z'], ' ', regex=True, inplace=True)

"""RUN stopwords_and_stemmimg FUNCTION, THEN SEPARATING THE DATASET USED FOR TRAINING AND TESTING"""
combine = combine_news_prices()
df = combine.stopwords_and_stemmimg(df5, df2)
"""USE WHEN USE ROBERTA"""
# df = combine.stopwords_and_stemmimg_roberta(df5, df2)

traindata = df.iloc[:int(len(df['Headlines'])*0.8)]
testdata = df.iloc[int(len(df['Headlines'])*0.8):]

"""PLOT THE WORDCLOUD FOE UP AND DOWN MOVEMENT"""
def wordcloud():
    stopwords1 = set(STOPWORDS)
    stopwords1.update(['may', 'new', 'price', 'growth', 'investor', 'year', 'outlook', 'trade', 'rally', 'still', 'global', 'future', 'higher', 'continue', 'inflation', 'trump', 'economy', 'stock', 'market', 'gold', 'oil', 'natural', 'gas', 'china', 'street', 'breakfast', 'wall', 'street', 'fed', 'dollar'])
    wordcloud = WordCloud(stopwords=stopwords1).generate(' '.join(df['New_Headlines'].loc[df['Label']==0]))
    wordcloud2 = WordCloud(stopwords=stopwords1).generate(' '.join(df['New_Headlines'].loc[df['Label']==1]))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.imshow(wordcloud2, interpolation='bilinear')
    plt.axis("off")
    plt.show()
    print(wordcloud.words_)
# wordcloud()



"""""""""""""""""""""""""DATA EVALUATION"""""""""""""""""""""""""

class training():
    def __init__(self, mode):
        if mode == 'normal':
            # countvector = CountVectorizer(ngram_range=(1, 2), max_df=0.4, max_features=10000, stop_words=stop_words)
            tfidfconverter = TfidfVectorizer(ngram_range=(1, 1), max_features=20000, max_df=0.5, stop_words=stop_words)
            column_trans = ColumnTransformer([('VIX_vector', 'passthrough', ['VIX']), ('New_Headlines_vector', tfidfconverter, 'New_Headlines')], remainder='drop')
            self.traindataset = column_trans.fit_transform(traindata)
            self.testdataset = column_trans.transform(testdata)
            pickle.dump(column_trans, open('feature.pkl', 'wb'))

        elif mode == 'roberta':
            """USE WHEN ROBERTA TOKENIZER IS USED FOR TOKENIZING AND WORD EMBEDDING"""
            dataset = []
            for i in df['New_Headlines'].values:
                dataset.append(tf.keras.preprocessing.sequence.pad_sequences([np.concatenate(i)], maxlen=1300, padding='post', dtype=np.float)[0])
            self.dataset = np.array(dataset).reshape(-1, 1300)
            self.xtrain, self.xtest, self.ytrain, self.ytest = train_test_split(dataset, df['Label'], test_size=0.2)

    """SAVE THE MODEL IF IT PROVIDES BEST RESULT"""
    def save_model(self, model):
        filename = 'finalized_model.sav'
        pickle.dump(model, open(filename, 'wb'))

    """GRID SEARCH FOR RANDOM FOREST"""
    def grid_search_RM(self):
        pipeline = Pipeline([
            # ('vect', CountVectorizer()),
            ('tfidf', TfidfVectorizer()),
            ('clf', RandomForestClassifier()),
        ])

        parameters = {
            # 'vect__max_df': (0.3, 0.4, 0.5, 0.6),
            # 'vect__max_features': (9000, 10000, 11000, 12000),
            # 'vect__ngram_range': ((1, 1), (1, 2), (2, 2)),
            # 'vect__stop_words': [stop_words],
            'tfidf__max_df': (0.3, 0.4, 0.5, 0.6),
            'tfidf__max_features': (10000, 12000, 20000),
            'tfidf__ngram_range': ((1, 1), (1, 2), (2, 2)),
            'tfidf__stop_words': [stop_words],
            'clf__n_estimators': (1000, 2000, 4000),
            'clf__criterion': ['entropy']
        }

        grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1, cv=3, scoring='accuracy')

        grid_search.fit(traindata['Headlines'], traindata['Label'])
        print("Best score: %0.3f" % grid_search.best_score_)
        print("Best parameters set:")
        best_parameters = grid_search.best_estimator_.get_params()
        for param_name in sorted(parameters.keys()):
            print("\t%s: %r" % (param_name, best_parameters[param_name]))

    """RANDOM FOREST FOR SKLEARN VECTORIZER"""
    def RM_train(self):
        randomclassifier = RandomForestClassifier(n_estimators=1000, criterion='entropy')
        randomclassifier.fit(self.traindataset, traindata['Label'])
        predictions = randomclassifier.predict(self.testdataset)
        predictions_prob = randomclassifier.predict_proba(self.testdataset)
        accuarcy = accuracy_score(testdata['Label'], predictions)
        classification = classification_report(testdata['Label'], predictions)
        matrix = confusion_matrix(testdata['Label'], predictions)
        logloss = log_loss(testdata['Label'], predictions_prob, eps=1e-15)
        print('Accruacy: ', accuarcy, '\n')
        print(classification)
        print('Confusion Matrix: \n', matrix, '\n')
        print('Log Loss: ', logloss, '\n')

        # self.save_model(model=randomclassifier)

        principal = 100000
        acc_return = []
        num_shares = 0
        txn_cost = 30
        for i in range(len(predictions)):
            pre_adj_close = testdata['Adj Close'].iloc[i-1]
            if i == 0:
                pre_adj_close = 295.56
                if predictions[i] == 0:
                    num_shares = int(principal / pre_adj_close)
                else:
                    num_shares = -int(principal / pre_adj_close)

            elif predictions[i-1] == 1 and predictions[i] == 0:
                num_shares = int(principal / pre_adj_close)
                txn_cost = 60
            elif predictions[i-1] == 0 and predictions[i] == 1:
                num_shares = -int(principal / pre_adj_close)
                txn_cost = 60
            else:
                num_shares = num_shares

            if predictions[i] == testdata['Label'].iloc[i] and predictions[i] == 0:
                principal = principal - num_shares * pre_adj_close + (num_shares * pre_adj_close * (1+testdata['Return'].iloc[i])) - txn_cost
                print('Correct Up: ', principal, testdata['Dates'].iloc[i], testdata['Return'].iloc[i], predictions_prob[i], num_shares, pre_adj_close, testdata['Adj Close'].iloc[i])
            elif predictions[i] != testdata['Label'].iloc[i] and predictions[i] == 0:
                if testdata['Return'].iloc[i] > -0.05:
                    principal = principal - num_shares * pre_adj_close + (num_shares * pre_adj_close * (1+testdata['Return'].iloc[i])) - txn_cost
                else:
                    principal = principal - num_shares * pre_adj_close + (num_shares * pre_adj_close * (1 - 0.05)) - txn_cost
                print('Wrong Up: ', principal, testdata['Dates'].iloc[i], testdata['Return'].iloc[i], predictions_prob[i], num_shares, pre_adj_close, testdata['Adj Close'].iloc[i])
            elif predictions[i] == testdata['Label'].iloc[i] and predictions[i] == 1:
                principal = principal + num_shares * pre_adj_close - (num_shares * pre_adj_close * (1 - testdata['Return'].iloc[i])) - txn_cost
                print('Correct Down: ', principal, testdata['Dates'].iloc[i], testdata['Return'].iloc[i], predictions_prob[i], num_shares, pre_adj_close, testdata['Adj Close'].iloc[i])
            elif predictions[i] != testdata['Label'].iloc[i] and predictions[i] == 1:
                if testdata['Return'].iloc[i] < 0.05:
                    principal = principal + num_shares * pre_adj_close - (num_shares * pre_adj_close * (1 - testdata['Return'].iloc[i])) - txn_cost
                else:
                    principal = principal + num_shares * testdata['Adj Close'].iloc[i-1] - (num_shares * testdata['Adj Close'].iloc[i-1] * (1 - 0.05)) - txn_cost
                print('Wrong Down: ', principal, testdata['Dates'].iloc[i], testdata['Return'].iloc[i], predictions_prob[i], num_shares, testdata['Adj Close'].iloc[i-1], testdata['Adj Close'].iloc[i])

            acc_return.append(principal)
        print('\nPrincipal at the end: ', principal)
        fig, ax1 = plt.subplots()
        testdata['Dates'] = pd.to_datetime(testdata['Dates'])
        ax1.plot(testdata['Dates'], acc_return)
        ax1.set_xlabel('Dates')
        ax1.set_ylabel('Principal')
        ax1.set_title('Investment in SPY')
        ax1.xaxis.set_major_locator(matplotlib.dates.MonthLocator([1, 4, 7, 10]))
        ax1.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y-%m'))
        plt.show()

    """USE WHEN ROBERTA TOKENIZER IS USED FOR TOKENIZING AND WORD EMBEDDING"""
    def RM_train_roberta(self):
        """RANDOM FOREST FOR ROBERTA"""
        randomclassifier = RandomForestClassifier(n_estimators=1000, criterion='entropy')
        randomclassifier.fit(self.xtrain, self.ytrain)

        predictions = randomclassifier.predict(self.xtest)
        predictions_prob = randomclassifier.predict_proba(self.xtest)
        accuarcy = accuracy_score(self.ytest, predictions)
        classification = classification_report(self.ytest, predictions)
        matrix = confusion_matrix(self.ytest, predictions)
        logloss = log_loss(self.ytest, predictions_prob, eps=1e-15)
        print(accuarcy)
        print(classification)
        print(matrix)
        print(logloss)


    """GRID SEARCH FOR MLP"""
    def grid_search_model_MLP(self):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(units=1024, activation='relu', kernel_initializer='he_uniform'))
        model.add(tf.keras.layers.Dense(units=64, activation='relu'))
        model.add(tf.keras.layers.Dense(units=1, activation='sigmoid',))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def grid_search_MLP(self):
        model = KerasClassifier(build_fn=self.grid_search_model_MLP, verbose=0)
        batch_size = [10, 20, 40, 60, 80, 100]
        epochs = [10, 20, 50, 100]
        param_grid = dict(batch_size=batch_size, epochs=epochs)
        grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3, scoring='accuracy')
        grid_result = grid.fit(self.traindataset.toarray(), traindata['Label'])
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

    """MLP FOR SKLEARN VECTORIZER"""
    def MLP_train(self):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(units=256, activation='relu', kernel_initializer='he_uniform'))
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(units=64, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

        tensorboard = TensorBoard(log_dir="logs/{}".format('MLP_2'))
        adam = tf.keras.optimizers.Adam(lr=3e-6)
        model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
        model.fit(self.traindataset.toarray(), traindata['Label'], epochs=300, batch_size=32, verbose=1, validation_split=0.2, shuffle=True, callbacks=[tensorboard])

        val_loss, val_acc = model.evaluate(self.testdataset.toarray(), testdata['Label'])
        print(val_acc)
        print((model.predict(self.testdataset)).shape)

    """MLP FOR ROBERTA"""
    def MLP_train_roberta(self):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(units=256, input_shape=(1300,), activation='relu', kernel_initializer='he_uniform'))
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(units=64, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

        tensorboard = TensorBoard(log_dir="logs/{}".format('MLP_2'))
        adam = tf.keras.optimizers.Adam()#lr=3e-6)
        model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
        model.fit(self.xtrain, self.ytrain, epochs=100, batch_size=64, verbose=1, validation_split=0.2, shuffle=True, callbacks=[tensorboard])

        val_loss, val_acc = model.evaluate(self.xtest, self.ytest)
        print(val_acc)
        print((model.predict(self.xtest)).shape)


    """GRID SEARCH FOR LSTM"""
    def grid_search_model_LSTM(self):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.LSTM(units=256, input_shape=(1, self.traindataset.toarray().shape[1]), return_sequences=True))
        model.add(tf.keras.layers.LSTM(units=256, input_shape=(1, self.traindataset.toarray().shape[1])))
        model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def grid_search_LSTM(self):
        model = KerasClassifier(build_fn=self.grid_search_model_LSTM, verbose=0)
        batch_size = [10, 20, 40, 60, 80, 100]
        epochs = [10, 20, 50, 100]
        param_grid = dict(batch_size=batch_size, epochs=epochs)
        grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3, scoring='accuracy')
        grid_result = grid.fit(np.array(self.traindataset.toarray()).reshape(self.traindataset.toarray().shape[0], 1, self.traindataset.toarray().shape[1]), traindata['Label'])
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

    """LSTM FOR SKLEARN VECTORIZER"""
    def LSTM_train(self):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.LSTM(units=256, input_shape=(1, self.traindataset.toarray().shape[1]), return_sequences=True))
        model.add(tf.keras.layers.LSTM(units=256, input_shape=(1, self.traindataset.toarray().shape[1])))
        model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

        adam = tf.keras.optimizers.Adam(lr=1e-5)
        model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
        # tensorboard = TensorBoard(log_dir="logs/{}".format('LSTM'))
        model.fit(np.array(self.traindataset.toarray()).reshape(self.traindataset.toarray().shape[0], 1, self.traindataset.toarray().shape[1]), traindata['Label'], epochs=50, batch_size=32, validation_split=0.2, verbose=1, shuffle=False)

        val_loss, val_acc = model.evaluate(np.array(self.testdataset.toarray()).reshape(self.testdataset.toarray().shape[0], 1, self.testdataset.toarray().shape[1]), testdata['Label'])
        print(val_acc)

    """LSTM FOR ROBERTA"""
    def LSTM_train_roberta(self):
        dataset = np.array(self.dataset).reshape(-1, 1300, 1)
        xtrain, xtest, ytrain, ytest = train_test_split(dataset, df['Label'], test_size=0.2)
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.LSTM(units=62, input_shape=(xtrain.shape[1], xtrain.shape[2]), return_sequences=True))
        model.add(tf.keras.layers.LSTM(units=64, input_shape=(xtrain.shape[1], xtrain.shape[2])))
        model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

        adam = tf.keras.optimizers.Adam()#lr=1e-5)
        model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
        tensorboard = TensorBoard(log_dir="logs/{}".format('LSTM_2'))
        model.fit(xtrain, ytrain, epochs=10, batch_size=32, validation_split=0.2, verbose=1, shuffle=False, callbacks=[tensorboard])

        val_loss, val_acc = model.evaluate(xtest, ytest)
        print(val_acc)

model_training = training( mode='normal')

# model_training.grid_search_RM()
model_training.RM_train()
# model_training.RM_train_roberta()

# model_training.grid_search_MLP()
model_training.MLP_train()
# model_training.MLP_train_roberta()

# model_training.grid_search_LSTM()
model_training.LSTM_train()
# model_training.LSTM_train_roberta()







"""TESTING FOR GETTING ONE-DAY NEWS HEADLINES"""
def testing_get_news():
    headlines, dates, tickers, total = [], [], [], []
    for i in range(0, 6):
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

    for k in range(len(headlines)):
        total.append((headlines[k], dates[k], ''))
    return total

def stopwords_and_stemmimg_testing(df, df2):
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

    df3 = pd.DataFrame(data_for_df)
    df3.columns = ['Headlines', 'Dates']
    df3.iloc[:, 0].replace(['^a-zA-Z'], ' ', regex=True, inplace=True)
    df3 = df3.sort_values(by='Dates')
    df3.set_index('Dates', inplace=True)
    df3['New_Headlines'] = df3['Headlines']
    df2['VIX'] = df2['t_VIX']
    df2['Label'] = ''
    df2['Return'] = ''
    df4 = pd.merge(df3['Headlines'], df2[['Label', 'Return', 'VIX']], left_index=True, right_index=True, how='left')
    df4['New_Headlines'] = df4['Headlines']
    df4.dropna(inplace=True)
    df4.reset_index(inplace=True)
    return df4

def testing(start, end, today, yesterday):
    news = testing_get_news()
    # ytd_news = [i for i in news]
    ytd_news = pd.DataFrame(news, columns=('Headlines', 'Dates', 'Ticker'))
    create_csv.change_dates_tocsv(method='production', df_news=ytd_news, today=today, yesterday=yesterday)
    df1 = pd.read_csv('Top_News_4.csv')
    df1.iloc[:, 0].replace(['^a-zA-Z'], ' ', regex=True, inplace=True)
    df2 = create_csv.create_price(start, end)
    df = stopwords_and_stemmimg_testing(df1, df2)
    # df = stopwords_and_stemmimg(df1, df2)
    print(df)
    # for i in range(len(df['Label'])):
    #     print('Actual: ', df['Label'].iloc[i], df['Dates'].iloc[i] df['New_Headlines'].iloc[i])
    tfidfconverter = pickle.load(open('feature.pkl', 'rb'))
    dataset = tfidfconverter.transform(df)
    loaded_model = pickle.load(open('finalized_model.sav', 'rb'))
    predictions_prob = loaded_model.predict_proba(dataset)
    predictions = loaded_model.predict(dataset)
    print('Predict probability: ', predictions)
    print('Predict: ', predictions_prob)

# start = datetime.datetime(2020, 5, 20)
# end = datetime.datetime(2020, 8, 17)
# today = datetime.date.today().strftime('%b %d, %Y')
# yesterday = (datetime.date.today()-datetime.timedelta(days=1)).strftime('%b %d, %Y')
# testing(start, end, today, yesterday)

"""Dates
Headlines
Label
Return
VIX
Adj Close
New_Headlines
"""
