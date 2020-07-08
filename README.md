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

In S&P500, most of the news are focused on blue chips. Below is top 100 frequency

![Image of companies distribution](https://github.com/ccw0530/Sentiment_Analysis_News/blob/master/company%20distribution.png)

**Create the Target Label dataset**

Use Pandas datareader for getting stock Adj Close prices from Dec 2015 to Jun 2020 sourced from Yahoo Finance

Calculate Daily Return by (today closing price/yesterday closing price -1)

Label "0" if Daily Return >= 0 or otherwise "1"

For weekly and monthly up and down, it can show that usually up is more than down movement
![Image of companies weekly](https://github.com/ccw0530/Sentiment_Analysis_News/blob/master/weekly%20up%26down.png)
![Image of companies monthly](https://github.com/ccw0530/Sentiment_Analysis_News/blob/master/monthly%20up%26down.png)

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

Among three classifiers, Random forest has the best accuracy. To find the "best" hyperparameters, GridSearchCV is used.

&nbsp;

Below is the results of three models:

<ins>Random Forest</ins>

Accuracy: 0.5442477876106194

                  precision    recall  f1-score   support

             0       0.57      0.85      0.68       129
             1       0.41      0.13      0.20        97

    accuracy                             0.54       226
    macro avg        0.49      0.49      0.44       226
    weighted avg     0.50      0.54      0.48       226
   



&nbsp;

Confusion Matrix

[[110  19]

 [ 84  13]]
 
 &nbsp;
 
 <ins>MLP</ins>
 
Accuracy: 0.508849561214447

&nbsp;

<ins>LSTM</ins>

Accuracy: 0.5

## Interpretation

The result is just to be above 50% accuracy due to below reasons:
- Vader may not understand financial news very well due to finance terminology
- Headlines are prone to be neutral and may not show strong negative words, causing bad Recall for predicting down side
- Some other factors which can affect the index cannot be covered in the top news because every day it has few top news. It may need more news for this project

As the wordings of headlines varies, it is hard to cluster them into group unless need further name entity recognition (NER) and NLP to find verb and objects to identify the cluster better. Below is the example that the model thinks they are similar using KMeans and Ball Tree (tree size=2):

Cluster ID 82 and ID 6 have similar subjects which is Futures in this example

[82  6]
- 82 Futures point to muted open 2019-04-15
- 82 Futures point to earnings-driven gains 2019-04-17
- 82 Futures point to slight losses 2019-04-18
- 82 Futures point barely downward 2016-04-20
- 82 Futures point to a flat open for U.S. stocks 2019-04-23
- 82 Futures point to weaker open 2017-04-04
- 6 Futures inch up ahead of busy Fed day 2019-04-10
- 6 Stocks edge higher ahead of FOMC minutes 2019-04-10
- 6 Futures can't hold gains, slip into the red 2016-04-15
- 6 Futures slip, but well off of overnight lows after Doha yields no progress 2016-04-18
- 6 Futures lower ahead of earnings 2017-04-18

Apart from index price movement, I have used Vader to calculate SIA score for each headlines and the do the index price prediction, although this project aims to predict next day index movement, not price.

The SIA score and return movement is shown below

![Image of polarity score vs return](https://github.com/ccw0530/Sentiment_Analysis_News/blob/master/polarity%20score%20vs%20return_2.png)


SIA score of each day shows that it has no show of very strong relationship with return, like abovementioned points, top news from website is dominant to the result. If the headlines not well deliver the sentiment of the market. It has very bad Recall on prediction of price decrease

![Image of polarity score](https://github.com/ccw0530/Sentiment_Analysis_News/blob/master/polarity%20score.png)

However, if the last day closing price and the SIA score are fitted into the next day price prediction, it can show that closing price with sentiment score can learn faster than just using historical price alone in LSTM model with epoches 50. Meaning that news have impacts to improve the model learning the prices

![Image of predicted price](https://github.com/ccw0530/Sentiment_Analysis_News/blob/master/predicted%20price.png)
