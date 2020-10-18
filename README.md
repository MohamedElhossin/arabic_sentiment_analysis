# arabic_sentiment_analysis
This repo to create a very simple chatbot for sentiment analysis Arabic reviews for hotels and some analysis and visualization about these hotels.

## 1- Sentiment Analysis

> ### **work flow**

![image](https://i.ibb.co/HxbRX8T/1.png)

> ### **Data collection**
>> balanced-reviews.tsv: a tab separated file containing a balanced dataset of positive and negative reviews. The ratings are mapped into positive (ratings 4 & 5) and negative (ratings 1 & 2). No nuetral reviews are included. The dataset consists of 93700 reviews; 46850 for each positive and negative classes. 

> ### **Remoive noise**
>> Removing noise from Arabic hotel reviews like ( Al tashqel and special characters)

> ### **Removing stopwords**
>> Removing Arabic stopsword like(هذا , هذه , أنا ....) by using nltk stopwords

> ### **Tokenization**
>> Given a character sequence and a defined document unit by using word_tokenize from nltk 

> ### **lemmatization**
>> lemmatization is returning the word to its root like(الأولاد --> ولد, يلعبون --> لعب) by using api callled [**Farasa**](https://pypi.org/project/farasapy/) 

> ### **Training model**
>> Training  that can sentiment the review if it positive or negative by using **NaiveBayesClassifier** model from nltk 
