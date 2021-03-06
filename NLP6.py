import numpy as np
from pandas import read_csv
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
import pickle
import datetime
from nltk.corpus import stopwords
import nltk
import matplotlib
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
import pandas_datareader.data as web
from nltk.stem import PorterStemmer

look_back = 1
prediction_day = 1
batch_size = 64

def create_dataset(dataset, look_back, prediction_day):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - prediction_day):
        dataX.append(dataset[i:i + look_back])
        dataY.append(dataset[i + look_back + prediction_day][0])
    return np.array(dataX), np.array(dataY)

df0 = pd.read_csv('Top_News_2.csv')
df0.iloc[:, 0].replace(['^a-zA-Z'], ' ', regex=True, inplace=True)
df = pd.read_csv('Top_News_4.csv')
df.iloc[:, 0].replace(['^a-zA-Z'], ' ', regex=True, inplace=True)
df = df.append(df0)
df = df.sort_values(by='Dates')
df.reset_index(drop=True, inplace=True)

stop_words = set(stopwords.words('english'))
ps = PorterStemmer()
results = []
def news_sentiment():
    for i in range(len(df['Headlines'])):
        news = df['Headlines'][i]
        word_tokens = nltk.word_tokenize(news)
        filtered_sentence = [w for w in word_tokens if w not in stop_words]
        stemmed_sentence = [ps.stem(w) for w in filtered_sentence]
        news = ' '.join(stemmed_sentence)
        pol_score = SIA().polarity_scores(news)  # run analysis
        pol_score['headline'] = news  # add headlines for viewing
        pol_score['date'] = df['Dates'][i]
        results.append(pol_score)

    with open("result_top_news_2.pickle", "wb") as f:
        pickle.dump(results, f)
# news_sentiment()

"""GET S&P500 CLOSING PRICES (SPY)"""
def create_price(start, end):
    df_stock = web.DataReader('SPY', 'yahoo', start, end)
    df_stock.to_csv('SPY.csv')
    df2 = pd.read_csv('SPY.csv', index_col=0)
    df2 = df2.drop(['Open', 'High', 'Low', 'Close', 'Volume'], axis=1)  # drop unwanted rows
    df2['Return'] = df2['Adj Close'] / df2['Adj Close'].shift(1) - 1
    df2['Label'] = ''
    df_vix = web.DataReader('^VIX', 'yahoo', start, end)
    df_vix.to_csv('^VIX.csv')
    df2['t_VIX'] = df_vix['Adj Close']/df_vix['Adj Close'].shift(1) - 1
    df2['VIX'] = df2['t_VIX'].shift(31-look_back)

    for i in range(len(df2['Return'])):
        if df2['Return'].iloc[i] >= 0:
            df2['Label'].iloc[i] = 0
        elif df2['Return'].iloc[i] < 0:
            df2['Label'].iloc[i] = 1
        else:
            df2['Label'].iloc[i] = np.nan
    return df2


with open("result_top_news_2.pickle", "rb") as f:
    results = pickle.load(f)
def read_news(df2):
    df3 = pd.DataFrame(results)
    df3 = df3.drop(['neg', 'neu', 'pos'], axis=1)
    df3.columns = ['Score', 'Headlines', 'Dates']

    df4 = df3.groupby(['Dates']).mean()
    df4['New_Score'] = df4['Score'].shift(1)

    df5 = pd.merge(df2[['Return']], df4[['New_Score']], left_index=True, right_index=True, how='left')
    df5.reset_index(inplace=True)
    fig, ax1 = plt.subplots()
    df5['Date']= pd.to_datetime(df5['Date'])
    ax1.plot(df5['Date'], df5['New_Score'], 'g-')
    ax1.xaxis.set_major_locator(matplotlib.dates.MonthLocator([1, 4, 7, 10]))
    ax1.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y-%m'))
    ax2 = ax1.twinx()
    ax2.plot(df5['Date'], df5['Return'], 'b-')
    ax1.set_xlabel('Dates')
    ax1.set_ylabel('New_Score', color='g')
    ax1.set_ylim(-2, 1.5)
    ax2.set_ylabel('Return', color='b')
    plt.tight_layout()

    df5.fillna(0, inplace=True)
    # dfReturnsScore = df5[(df5['New_Score'] > 50) | (df5['New_Score'] < -50)]
    df5.plot(x="New_Score", y="Return", style="o")
    plt.ylabel('Return')
    plt.show()

    dfReturnsScore = pd.merge(df2[['Return', 'VIX']], df4[['Score']], left_index=True, right_index=True, how='left')
    dfReturnsScore.fillna(0, inplace=True)
    return dfReturnsScore

# load the dataset
dataframe = read_csv('SPY.csv')
dataframe = dataframe.drop(['Open', 'High', 'Low', 'Close', 'Volume'], axis=1)
dataframe.set_index('Date', inplace=True)
dataframe['Price Change'] = dataframe['Adj Close'] - dataframe['Adj Close'].shift(3)
dataframe['Price Change'].fillna(0, inplace=True)
start = datetime.datetime(2015, 10, 20)
end = datetime.datetime(2020, 6, 22)
df2 = create_price(start, end)
dataframe2 = read_news(df2)
dataframe3 = pd.merge(dataframe[['Price Change']], dataframe2[['Score', 'VIX']], left_index=True, right_index=True, how='left')
# final_dataset = np.array(dataframe3['Price Change'].values, dtype=float).reshape((-1, 1))
final_dataset = np.array(dataframe['Adj Close'].values, dtype=float).reshape((-1,1))


def LSTM_model(mode, dataset):
    if mode == 'single':
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)
        print('Single')
    if mode == 'multiple':
        dataset2 = np.array(dataframe3['Score'].values, dtype=float).reshape(-1, 1)
        dataset = np.concatenate((dataset, dataset2), axis=1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)
        print('Multiple')

    # split into train and test sets
    train_size = int(len(dataset) * 0.8)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

    trainX, trainY = create_dataset(train, look_back, prediction_day)
    testX, testY = create_dataset(test, look_back, prediction_day)

    model = Sequential()
    model.add(LSTM(64, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2])))
    model.add(Dense(1))
    print(model.summary())
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=50, batch_size=batch_size, verbose=2, shuffle=False, validation_split=0.2)

    model.reset_states()
    trainPredict = model.predict(trainX, batch_size=batch_size)
    model.reset_states()
    testPredict = model.predict(testX, batch_size=batch_size)

    if mode == 'single':
        trainPredict = scaler.inverse_transform(trainPredict)
        testPredict = scaler.inverse_transform(testPredict)
        a_dataset = scaler.inverse_transform(dataset)

    elif mode == 'multiple':

        trainPredict = np.concatenate((trainPredict, trainX[:, 0, 1].reshape(-1, 1)), axis=1)
        trainPredict = scaler.inverse_transform(trainPredict)
        trainPredict = trainPredict[:, 0].reshape(-1, 1)

        testPredict = np.concatenate((testPredict, testX[:, 0, 1].reshape(-1, 1)), axis=1)
        testPredict = scaler.inverse_transform(testPredict)
        testPredict = testPredict[:, 0].reshape(-1, 1)

        a_dataset = scaler.inverse_transform(dataset)
        a_dataset = a_dataset[:, 0].reshape(-1, 1)

    trainPredictPlot = np.empty_like(dataset)
    trainPredictPlot[:] = np.nan
    trainPredictPlot[look_back + prediction_day:len(trainPredict) + look_back + prediction_day] = trainPredict
    trainPredictPlot = trainPredictPlot[:, 0]

    testPredictPlot = np.empty_like(dataset)
    testPredictPlot[:] = np.nan
    testPredictPlot[len(trainPredict) + (look_back + prediction_day) * 2:len(trainPredict) + (look_back + prediction_day) * 2 + len(testPredict), :] = testPredict
    testPredictPlot = testPredictPlot[:, 0]

    if mode == 'multiple':
        pickle_out = open("a_dataset.pickle", "wb")
        pickle.dump(a_dataset, pickle_out)
        pickle_out.close()

        pickle_out = open("trainPredictPlot.pickle", "wb")
        pickle.dump(trainPredictPlot, pickle_out)
        pickle_out.close()

        pickle_out = open("testPredictPlot.pickle", "wb")
        pickle.dump(testPredictPlot, pickle_out)
        pickle_out.close()
    elif mode == 'single':
        pickle_out = open("a_dataset_s.pickle", "wb")
        pickle.dump(a_dataset, pickle_out)
        pickle_out.close()

        pickle_out = open("trainPredictPlot_s.pickle", "wb")
        pickle.dump(trainPredictPlot, pickle_out)
        pickle_out.close()

        pickle_out = open("testPredictPlot_s.pickle", "wb")
        pickle.dump(testPredictPlot, pickle_out)
        pickle_out.close()

LSTM_model('single', final_dataset)
LSTM_model('multiple', final_dataset)
