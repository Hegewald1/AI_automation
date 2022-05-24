# Spring 2022
# PBA. Ai And Automation. Eaaa.
# Sila.

# Tweet Sentiment

'''Tweeet sentiment analysis based on BOW.
(Where) Bag-of-words is a vector representation
of text. Each of the vector dimensions captures either the frequency, presence or absence, or
weighted values of words in the text. A bag-of-words representation does not capture the
order of the words.'''

import random
import nltk
nltk.download('twitter_samples')
from nltk.corpus import twitter_samples
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np

pos_tweets = [(string, 1) for string in twitter_samples.strings('positive_tweets.json')]
neg_tweets = [(string,0) for string in twitter_samples.strings('negative_tweets.json')]
pos_tweets.extend(neg_tweets)
comb_tweets = pos_tweets
random.shuffle(comb_tweets)
tweets,labels = (zip(*comb_tweets))

count_vectorizer = CountVectorizer(ngram_range=(1,2),max_features=3000)
X = count_vectorizer.fit_transform(tweets).toarray()

X_train,X_test,y_train,y_test = train_test_split(X,labels,test_size=0.2,random_state=10)

input_dim = X_train.shape[1]  # Number of features

X_train = np.array(X_train, dtype=np.uint8)
X_test = np.array(X_test, dtype=np.uint8)
y_train = np.array(y_train, dtype=np.uint8)
y_test = np.array(y_test, dtype=np.uint8)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(100, input_dim=input_dim, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(100, activation='sigmoid'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')])

model.compile(
     loss='binary_crossentropy',
     optimizer='adam',
     metrics=['accuracy']
)

model.summary()

history = model.fit(X_train, y_train,
          batch_size=1,
          epochs=10,
          verbose=1,
          validation_data=(X_test, y_test))

test_text = ["Bad movie. Stinks. Terrible actors. Plot makes no sense."]
Example_Tweet = count_vectorizer.transform(test_text)
print(test_text, model.predict(Example_Tweet.toarray()))

test_text = ["Great movie. Enjoyed it a lot. Wonderful actors. Best story ever."]
Example_Tweet = count_vectorizer.transform(test_text)
print(test_text, model.predict(Example_Tweet.toarray()))

import matplotlib.pyplot as plt
def plot_train_history(history):
    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['accuracy', 'val_accuracy'])
    plt.show()

plot_train_history(history)