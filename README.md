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

Use Pandas datareader for getting S&P500(Ticker: ^GSPC) Adj Close prices from Dec 2015 to Jun 2020 sourced from Yahoo Finance. It should be fine to also use S&P500 ETF Trust (Ticker:SPY) because of this ETF tracing the performance of S&P500

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

Among three classifiers, three model accuracy is quite similar. To find the "best" hyperparameters, GridSearchCV can be used.

**Below hyperparameters are calculated by metices: neg_log_loss**

*GridSearchCV using logloss*

Best score: -0.664

Best parameters set:
  - clf__criterion: 'entropy'
  - clf__n_estimators: 1000
  - tfidf__max_df: 0.6
  - tfidf__max_features: 10000
  - tfidf__ngram_range: (1, 1)
 
 &nbsp;
 
 *GridSearchCV using accuracy*
 
 Best score: 0.625

 Best parameters set:
  - clf__criterion: 'entropy'
  - clf__n_estimators: 1000
  - tfidf__max_df: 0.5
  - tfidf__max_features: 20000
  - tfidf__ngram_range: (1, 1)
  
&nbsp;

<ins>Random Forest</ins>
 
 &nbsp;
  
 Accuracy: 0.5530973451327433 (range approxiately from 0.53 to 0.57)

                    precision    recall  f1-score   support

           0           0.57      0.89      0.69       128
           1           0.44      0.11      0.18        98

    accuracy                               0.55       226
    macro avg          0.50      0.50      0.44       226
    weighted avg       0.51      0.55      0.47       226

 &nbsp;

 Confusion Matrix:

[[114  14]

 [ 87  11]]
 
 Log Loss: 0.7008060886845231
 
 ************UPDATE on using Market Outlook and Top News headlines together************

Accuraqcy increases by around 3-6%. Log Loss also decreases, meaning predicted probability is closer to to actual result

Recall has 11% improvement as it can distinguish more down movement correctly


 Accuracy: 0.6079295154185022 (range approxiately from 0.57 to 0.60)

  
                  precision    recall  f1-score   support

           0           0.60      0.90      0.72       129
           1           0.63      0.22      0.33        98

    accuracy                               0.61       227
    macro avg          0.62      0.56      0.53       227
    weighted avg       0.61      0.61      0.55       227

Confusion Matrix:

[[116  13]

 [ 76  22]]
 

Log Loss: 0.681201017351206

 
 ************UPDATE on 19 Jul 2020 using Princiapl Component Analysis (PCA) for dimension deduction from 1200 to 120************ 
 
 Accuraqcy increases by around 2%. Log Loss also decreases, meaning predicted probability is closer to to actual result

 Recall has almost 20% improvement as it can distinguish more down movement correctly, although decreasing the recall for up movement
 
 
 Accuracy: 0.6211453744493393 (range from 0.59 to 0.62)
 
                   precision    recall  f1-score   support

           0           0.64      0.78      0.70       129
           1           0.59      0.42      0.49        98

    accuracy                               0.62       227
    macro avg          0.61      0.60      0.59       227
    weighted avg       0.61      0.62      0.61       227

Confusion Matrix:

[[100  29]

 [ 57  41]]
 
Log Loss: 0.6766458016955135
 
 
 
 <ins>MLP</ins>

Data is shuffled before fitting to train

Accuracy: 0.5619469285011292 (range approxiately from 0.55 - 0.56)

![Image of mlp validation accuracy](https://github.com/ccw0530/Sentiment_Analysis_News/blob/master/pics/Screenshot%202020-07-12%20at%209.57.31%20PM.png)

![Image of mlp validation loss](https://github.com/ccw0530/Sentiment_Analysis_News/blob/master/pics/Screenshot%202020-07-12%20at%209.58.06%20PM.png)

&nbsp;

<ins>LSTM</ins>

Data is NOT shuffled

Accuracy: 0.5663716793060303 (relatively stable comparing with above two models)

![Image of lstm validation accuracy](https://github.com/ccw0530/Sentiment_Analysis_News/blob/master/pics/Screenshot%202020-07-12%20at%209.59.07%20PM.png)

![Image of lstm validation loss](https://github.com/ccw0530/Sentiment_Analysis_News/blob/master/pics/Screenshot%202020-07-12%20at%209.59.22%20PM.png)

## Interpretation

The result is just to be above 50% accuracy due to below reasons:
- Vader may not understand financial news very well due to finance terminology. Below examples show that Vader cannot distingust the corresponding effect of interest rate (Monetary Policies imposed by FED) and no idea of what abbreviation GDP is

<ins>Example 1</ins>

'FED increases the interest rate by 0.25%'

  - {'neg': 0.0, 'neu': 0.667, 'pos': 0.333, 'compound': 0.4588}

'FED increases the interest rate by 0.25%'

  - {'neg': 0.0, 'neu': 0.667, 'pos': 0.333, 'compound': 0.4588}

<ins>Example 2</ins>

'GDP increases by 100%'

  - {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}

'GDP decreases by 100%'

  - {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
  
&nbsp;
&nbsp;

- Headlines are prone to be neutral and may not show strong negative words, causing bad Recall for predicting down side
- No bag of words for financial news to determine the sentiment
- Some other factors which can affect the index cannot be covered in the top news because every day it has few top news. It may need more news for this project

As the wordings of headlines varies, it is hard to cluster them into group unless need further name entity recognition (NER) and NLP to find verb and objects to identify the cluster better. Below is the example that the model thinks they are similar using KMeans Clustering and Ball Tree (tree size=2):

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

## Conclusion
It finds that news sentiemnt anaylsis have impact to price movement. However, wordings or sentiment of Headlines, number of headlines for 1 day, other factors, Vader sentiment classification would affect the result. 

For the future, it may need to extract another sources of information or include other factors to test any other area of news bringing imoact to stock movement. And, it may need to build custom bag of words that is for the purpose of analyzing the sentiment of financial headlines

Also, using another NLP model, such as BERT which use bi-directional way to understand the context of the sentence better and use the pretrained BERT to train a  custom features.
