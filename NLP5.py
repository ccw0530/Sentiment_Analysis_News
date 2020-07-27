from __future__ import absolute_import, division, print_function
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt

df = pd.read_csv('SPY.csv', parse_dates=['Date'])
df = df.drop(['Open', 'High', 'Low', 'Close', 'Volume'], axis=1)
df['Return'] = df['Adj Close']/df['Adj Close'].shift(1)-1
df['Label'] = ''
for i in range(len(df['Return'])):
    if df['Return'][i] >= 0:
        df['Label'][i] = 0
    elif df['Return'][i] < 0:
        df['Label'][i] = 1
    else:
        df['Label'][i] = np.nan

#month plot
df['Month'] = df['Date'].dt.month
month = {}
for i in range(12):
    month[i+1] = []

month_string = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
month_result = {}
for i in month_string.values():
    month_result[i+'_UP'] = 0
    month_result[i + '_DN'] = 0

for i in range(len(df['Date'])):
    for j in month:
        if j == df['Month'][i]:
            month[j].append(df['Label'][i])
            break

for i in month:
    for j in month[i]:
        if j == 0:
            month_result[month_string[i]+'_UP'] += 1
        elif j == 1:
            month_result[month_string[i] + '_DN'] += 1

up, down = [], []
for i in month_result:
    if 'UP' in i:
        up.append(month_result[i])
    else:
        down.append(month_result[i])

width = 0.35
fig, ax = plt.subplots()
rects1 = ax.bar(np.arange(len(month_result.keys())/2) - width/2, up, width, label='Up')
rects2 = ax.bar(np.arange(len(month_result.keys())/2) + width/2, down, width, label='Down')
ax.set_xticks(np.arange(len(month_result.keys())/2))
ax.set_xticklabels(list(month_string.values()))
ax.legend()
ax.set_xlabel('Month')
ax.set_ylabel('Number of days')
ax.set_title('Monthly Up&Down from Dec 2015 to Jun 2020')

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

fig.tight_layout()
plt.show()

#weekday plot
df['Weekday'] = ''
for i in range(len(df['Date'])):
    # print(type(df['Date'][i]))
    # date = datetime.datetime(df['Date'][i])
    df['Weekday'][i] = datetime.datetime.weekday(df['Date'][i])

weekday = {0: [], 1: [], 2: [], 3: [], 4: []}
weekday_string = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri'}
weekday_result = {'Mon_UP': 0, 'Mon_DN': 0, 'Tue_UP': 0, 'Tue_DN': 0, 'Wed_UP': 0, 'Wed_DN': 0, 'Thu_UP': 0, 'Thu_DN': 0, 'Fri_UP': 0, 'Fri_DN': 0,}
for i in range(len(df['Date'])):
    for j in weekday:
        if j == df['Weekday'][i]:
            weekday[j].append(df['Label'][i])
            break

for i in weekday:
    for j in weekday[i]:
        if j == 0:
            weekday_result[weekday_string[i]+'_UP'] += 1
        elif j == 1:
            weekday_result[weekday_string[i] + '_DN'] += 1

up, down = [], []
for i in weekday_result:
    if 'UP' in i:
        up.append(weekday_result[i])
    else:
        down.append(weekday_result[i])

width = 0.35  # the width of the bars
fig, ax = plt.subplots()
rects1 = ax.bar(np.arange(len(weekday_result.keys())/2) - width/2, up, width, label='Up')
rects2 = ax.bar(np.arange(len(weekday_result.keys())/2) + width/2, down, width, label='Down')
ax.set_xticks(np.arange(len(weekday_result.keys())/2))
ax.set_xticklabels(list(weekday_string.values()))
ax.legend()
ax.set_xlabel('Weekday')
ax.set_ylabel('Number of days')
ax.set_title('Weekday Up&Down from Dec 2015 to Jun 2020')

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

fig.tight_layout()
plt.show()

