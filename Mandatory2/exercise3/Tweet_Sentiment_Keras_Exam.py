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
import matplotlib.pyplot as plt
from nltk.corpus import twitter_samples
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np


nltk.download('twitter_samples')
pos_tweets = [(string, 1) for string in twitter_samples.strings('positive_tweets.json')]
neg_tweets = [(string, 0) for string in twitter_samples.strings('negative_tweets.json')]
pos_tweets.extend(neg_tweets)
comb_tweets = pos_tweets
random.shuffle(comb_tweets)
tweets, labels = (zip(*comb_tweets))

count_vectorizer = CountVectorizer(ngram_range=(1, 2), max_features=3000)
X = count_vectorizer.fit_transform(tweets).toarray()

X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=10)

input_dim = X_train.shape[1]  # Number of features

X_train = np.array(X_train, dtype=np.uint8)
X_test = np.array(X_test, dtype=np.uint8)
y_train = np.array(y_train, dtype=np.uint8)
y_test = np.array(y_test, dtype=np.uint8)

import transformers
transformer_model = transformers.TFBertModel.from_pretrained('distilbert-base-uncased')
# https://stackoverflow.com/questions/62771845/using-bert-embeddings-in-keras-embedding-layer

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(50, input_dim=input_dim, activation='relu'),
    # transformer_model,
    tf.keras.layers.Dropout(0.5),
    # tf.keras.layers.Dense(100, activation='sigmoid'),
    # tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')])

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()

history = model.fit(X_train, y_train,
                    batch_size=10,
                    epochs=25,
                    verbose=1,
                    validation_data=(X_test, y_test))

test_text = ["Bad movie. Stinks. Terrible actors. Plot makes no sense."]
Example_Tweet = count_vectorizer.transform(test_text)
print(test_text, model.predict(Example_Tweet.toarray()))

test_text = ["Great movie. Enjoyed it a lot. Wonderful actors. Best story ever."]
Example_Tweet = count_vectorizer.transform(test_text)
print(test_text, model.predict(Example_Tweet.toarray()))


def plot_train_history(history):
    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['accuracy', 'val_accuracy'])
    plt.show()


plot_train_history(history)

# Exercise c code
print('Own test exampels (3-5):')
# 1
test_text = ["Complex plot and hard to understand. Simple acting. Exaggerated rating. Peculiar music."]
Example_Tweet = count_vectorizer.transform(test_text)
print(test_text, model.predict(Example_Tweet.toarray()))
# 2
test_text = ["The acting was plain. Plot was cliche. Visuals were rough. Uncomfortable ending."]
Example_Tweet = count_vectorizer.transform(test_text)
print(test_text, model.predict(Example_Tweet.toarray()))
# # 3
test_text = ["Great movie. Enjoyed it a lot. Wonderful actors. Best story ever. NOT"]
Example_Tweet = count_vectorizer.transform(test_text)
print(test_text, model.predict(Example_Tweet.toarray()))
# # 4
test_text = ["Plot was terrible. Loved the actors though. The music was horrible. Visuals were amazing."]
Example_Tweet = count_vectorizer.transform(test_text)
print(test_text, model.predict(Example_Tweet.toarray()))
