# arabic_sentiment_analysis
This repo to demo a simple chatbot for Arabic sentiment analysis reviews for hotels and also include analysis and visualization for those hotels.

## 1- Sentiment Analysis
>At this process, I am trying to build a model that can sentiment the Arabic reviews about hotels if it is a positive or negative review and these are the steps that I took to solve this problem.

### **work flow:**

![image](https://i.ibb.co/HxbRX8T/1.png)

### **Data collection:**
>balanced-reviews.tsv: a tab separated file containing a balanced dataset of positive and negative reviews. The ratings are mapped into positive (ratings 4 & 5) and negative (ratings 1 & 2). No nuetral reviews are included. The dataset consists of 93700 reviews; 46850 for each positive and negative classes. 

### **Remoive noise:**
>Removing noise from Arabic hotel reviews like ( Al tashqel and special characters)

### **Removing stopwords:**
>Removing Arabic stopsword like(هذا , هذه , أنا ....) by using **nltk stopwords**

### **Tokenization;**
>Given a character sequence and a defined document unit by using **word_tokenize from nltk**

### **lemmatization:**
>lemmatization is returning the word to its root like(الأولاد --> ولد, يلعبون --> لعب) by using api callled [**Farasa**](https://pypi.org/project/farasapy/)

>>#### **Note:** For Farasa I didn't run it because of the limitation of computational recourse against time. 

### **Training model:**
>Training  that can sentiment the review if it positive or negative by using **NaiveBayesClassifier** model from nltk 

## 2- Analysis & Visualization:
>At this process, I had worked on a dataset for hotel's reviews and trying to analyze and extract insight from data and trying to show these insights in graphs.

### **work flow:**
![](https://i.ibb.co/3vPCvXH/2.png)

### **Data Collection:**
> unbalanced-reviews.tsv.rar: the whole dataset of 373,750 reviews. This is a clean dataset that includes all reviews.


## **Requirements**
for requirements libraries to run my chart

      $ python3 pip install -r requirements.txt
      

## **Quick start**
To run my chaty write in your cmd.

     $ python3 chaty.py


