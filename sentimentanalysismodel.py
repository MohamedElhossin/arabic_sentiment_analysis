import pandas as pd
import numpy as np 

import re
import random
import pickle

from farasa.pos import FarasaPOSTagger 
from farasa.ner import FarasaNamedEntityRecognizer 
from farasa.diacratizer import FarasaDiacritizer 
from farasa.segmenter import FarasaSegmenter 
from farasa.stemmer import FarasaStemmer

import pyarabic.arabrepr


import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.isri import ISRIStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import classify
from nltk import NaiveBayesClassifier

#nltk.download('punkt')
#nltk.download("stopwords")

def load_data():


  reviews = pd.read_csv('./dataset/balanced-reviews.csv', sep='\t', encoding='utf-16')

  # show some statistics about reviews

  postive_reviews = reviews[ (reviews['rating'] == 4) | (reviews['rating'] == 5)]['review']
  negative_reviews = reviews[ (reviews['rating'] == 2) | (reviews['rating'] == 1)]['review']

  #-->print('Number of reviews {}'.format(reviews.shape[0]))
  #-->print('Number of postive reviews {}'.format(len(postive_reviews)))
  #-->print('Number of negative reviews {}'.format(len(negative_reviews)))  

  return postive_reviews, negative_reviews

def tokenized_remove_stopword(review_list):
  """
  tokenized and remove method for remove stop words and tokenized the reviews

  Arg:
    review_list (list): list of reviews

  Return:
    review_tokens (list): list of tokens for each reviews    
  """

  review_tokens = []
 
  arb_stopwords = set(nltk.corpus.stopwords.words("arabic"))

  for review in review_list:

    tokens = word_tokenize(review)
    result = [i for i in tokens if not i in arb_stopwords]
    review_tokens.append(result)


  return review_tokens

"""
def lemmatize_sentence(review_list):

  lemmatize_review = []

  for review in review_list:
    token_list = []
    for token in review:
      token_list.append(stemmer.stem(token)) 

    print("lemet working")
    lemmatize_review.append(token_list)  

  return lemmatize_review  
"""

def deNoise(review_list):
  """
  deNoise method to remove noise from reviews like(al tashqel and special characters)

  Arg:
     reviews_list (list): list of reviews

  Return:
    clean_reviews (list): list of clean reviews    
  """

  noise = re.compile(""" ّ       | # Tashdid
                             َ    | # Fatha
                             ً    | # Tanwin Fath
                             ُ    | # Damma
                             ٌ    | # Tanwin Damm
                             ِ    | # Kasra
                             ٍ    | # Tanwin Kasr
                             ْ    | # Sukun
                             ـ   |  # Tatwil/Kashida
                             ؟   |    
                         """, re.VERBOSE)
    
  clean_reviews = []
  for review in review_list:
        review = re.sub(noise, '', review)
        review = review.replace('.' , '')
        review = review.replace('@' , '')
        review = review.replace('#' , '')
        review = review.replace(',' , '')
        review = review.replace('“' , '')
        review = review.replace('”', '')
        review = review.replace('!', '')

        clean_reviews.append(review)

  return clean_reviews

def get_all_words(cleaned_tokens_list):
  """
  get_all_words methods get the unique token from tokens

  Arg: 
    clean_tokens_list (list): list of clean tokens   
  """
  for tokens in cleaned_tokens_list:
      for token in tokens:
          yield token

def get_tweets_for_model(cleaned_tokens_list):
  """
  get_tweets_for_model get dictionary of tokens to used in traing model

  Arg:
    cleaned_tokens_list (list): list of clean tokens
  """
  for tweet_tokens in cleaned_tokens_list:
      yield dict([token, True] for token in tweet_tokens)

def split_train_test(positive_tokens_for_model, negative_tokens_for_model):

  """
  split_train_tain_test method to split positive and negative reviews 
  to train/test dataset 70/30

  Arg:
    positive_tokens_for_model (dict): dictionary of positive tokens
    negative_tokens_for_model (dict): dictionary of negative tokens

  Return:
    train_data (lsit): list of train pos/neg train tokens
    test_data (lsit): list of test pos/neg train tokens

  """

  positive_dataset = [(tweet_dict, "Positive")
                     for tweet_dict in positive_tokens_for_model]

  negative_dataset = [(tweet_dict, "Negative")
                     for tweet_dict in negative_tokens_for_model]

  dataset = positive_dataset + negative_dataset

  random.shuffle(dataset)

  train_data = dataset[:int((len(dataset)*0.7))]
  test_data = dataset[int((len(dataset)*0.7)):]

  return train_data, test_data

def preprocessing_pipeline(positive_reviews, negative_reviews):

  positive_without_noise = deNoise(list(positive_reviews))
  positive_clean_tokens = tokenized_remove_stopword(positive_without_noise)
  positive_tokens_for_model = get_tweets_for_model(positive_clean_tokens)

  negative_without_noise = deNoise(list(negative_reviews))
  negative_clean_tokens = tokenized_remove_stopword(negative_without_noise)
  negative_tokens_for_model = get_tweets_for_model(negative_clean_tokens)

  train_data, test_data = split_train_test(positive_tokens_for_model, negative_tokens_for_model)

  return train_data, test_data

def train_model(train_data, test_data):
  """
  train_model methods to train NaveBayes for arabic sentiment analysis and
  save train model as model.pickle

  Arg:
    train_data (lsit): list of train pos/neg train tokens
    test_data (lsit): list of test pos/neg train tokens

  """
  
  classifier = NaiveBayesClassifier.train(train_data)
  print("Accuracy is:", classify.accuracy(classifier, test_data))
  print(classifier.show_most_informative_features(20))

  f = open('my_classifier.pickle', 'wb')
  pickle.dump(classifier, f)
  f.close()

def sentiment_analysis(review):
  """
  sentiment_anaysis method to analysis the review if positive of negative 

  Arg:
    review (string): 

  Return:
    result (string):  
  """

  clean_review = deNoise([review])
  #-->rint(clean_review)

  tokens = tokenized_remove_stopword(clean_review)
  #-->print(tokens)

  with open('my_classifier.pickle', 'rb') as f:
    classifier = pickle.load(f)
 
  result = classifier.classify(dict([token, True] for token in tokens[0]))

  return result

if __name__ == "__main__":

  """
  uncomment this line if you want to train
  """
  #-->print('laod data..............')
  #-->positive_reviews, negative_reviews = load_data()
  #-->print('prepocessing data........')
  #-->train_data, test_data = preprocessing_pipeline(positive_reviews, negative_reviews)

  #-->print('traing model...........')
  #-->train_model(train_data, test_data)

  """
  uncomment this line if you want to train
  """
  #-->review = u'أعجبني هذا الفندق'
  #-->result  = sentiment_analysis(review)
  #-->print(result)

