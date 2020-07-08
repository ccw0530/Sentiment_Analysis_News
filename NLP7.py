import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd
import matplotlib

df = pd.read_csv('^GSPC_2.csv', header=0, parse_dates=[0], usecols=[0])
df = np.array(df.values).reshape(-1)


pickle_in = open("a_dataset_s.pickle","rb")
a_datase_s = pickle.load(pickle_in)

pickle_in = open("trainPredictPlot_s.pickle","rb")
trainPredictPlot_s = pickle.load(pickle_in)

pickle_in = open("testPredictPlot_s.pickle","rb")
testPredictPlot_s = pickle.load(pickle_in)

pickle_in = open("a_dataset.pickle","rb")
a_dataset = pickle.load(pickle_in)

pickle_in = open("trainPredictPlot.pickle","rb")
trainPredictPlot = pickle.load(pickle_in)

pickle_in = open("testPredictPlot.pickle","rb")
testPredictPlot = pickle.load(pickle_in)

ax1 = plt.subplots()
ax1 = plt.gca()
ax1.xaxis.set_major_locator(matplotlib.dates.MonthLocator([1, 4, 7, 10]))
ax1.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y-%b'))

ax1 = plt.subplot2grid((7,1), (0,0), rowspan=7, colspan=1)

ax1.plot(df, a_datase_s, label='Actual')
ax1.plot(df, trainPredictPlot_s, label='training (single)')
ax1.plot(df, testPredictPlot_s, label='testing (single)')

ax1.plot(df, trainPredictPlot, label='training (multiple)')
ax1.plot(df, testPredictPlot, label='testing (multiple)')

ax1.legend(loc='best fit')
ax1.set_xlabel('Date')
ax1.set_ylabel('Prices')
plt.grid(axis='both')
plt.show()