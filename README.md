# Sentiment_Analysis_News
Getting S&amp;P top news from Dec 2015 to Jun 2020

This project is to use 1-day news headlines to predict next day S&P500 movement (Up or Down)

Web scraping from the website Seeking Alpha to get "Headlines", "Dates" and "Ticker" of Top News



## Data Collection
**Create the News dataset**

One day news headlines is composed of: all news from EST4pm yesterday to all news before EST4pm today

To make decision to long or short the index for next day movement before today market close, news after 4pm is not included in today bin for prediction of next day movement

Therefore, news after market close would be grouped into next day bin.

- E.g. news released on 20:30pm 22 Jun 2020 is treated as 23 Jun 2020 bin

**Create the Target Label dataset**

Use Pandas datareader for getting stock Adj Close prices Dec 2015 to Jun 2020 sourced from Yahoo Finance

Calculate Daily Return by (today closing price/yesterday closing price -1)

Label "0" if Daily Return >= 0 and otherwise "1"



## Data Precoessing
**Date transformation**

As the dates extracted from the websites showing different formats. e.g. *Today, 5:38 AM* or *Sat, Jun. 4, 6:25 PM* or *Wed, Jun. 1, 11:19 AM*

Date format has to be changed with same format before further processing

**Stopwords and stemming**

As the words in the headlines have different tenses or structures. 

Stopwords can ignore some words which are too common

Stemming the words, e.g. devlop vs devloped, rapid vs rapidly, it may improve machine processing the vector better



## Prediction
This project has used three models to predict the accuracy: Random Forest, Mutilayer Perceptron Network (MLP) and Long Short Term Menory (LSTM)

Before fitting into above three classifiers, headlines are grouped into one vector and use TfidfVectorizer to change the words to number for processing

To find the "best" hyperparameters, GridSearchCV is used.

Random Forest
0.5442477876106194
              precision    recall  f1-score   support

           0       0.57      0.85      0.68       129
           1       0.41      0.13      0.20        97

    accuracy                           0.54       226
   macro avg       0.49      0.49      0.44       226
weighted avg       0.50      0.54      0.48       226

[[110  19]
 [ 84  13]]
 
 MLP
0.508849561214447

LSTM
0.491150438785553

![Image of polarity score vs return](https://github.com/ccw0530/Sentiment_Analysis_News/blob/master/polarity%20score%20vs%20return.png)
![Image of polarity score](https://github.com/ccw0530/Sentiment_Analysis_News/blob/master/polarity%20score.png)
![Image of companies distribution](https://github.com/ccw0530/Sentiment_Analysis_News/blob/master/company%20distribution.png)
