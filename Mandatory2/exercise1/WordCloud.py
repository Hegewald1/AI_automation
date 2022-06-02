import nltk
from nltk.corpus import webtext
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt


# download nltk packages
nltk.download('webtext')

# Example 1 with loading text from webtext
webtext_words = webtext.words('pirates.txt')
webtext_words = [word.upper() for word in webtext_words]
fdist = nltk.FreqDist(webtext_words)
filter_words = dict([(k, v) for k, v in fdist.items() if len(k) > 3])

# generate word cloud from filtered words and plot
wcloud = WordCloud(background_color='black', max_words=100).generate_from_frequencies(filter_words)
plt.figure(figsize=(10, 5))
plt.clf()
plt.imshow(wcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Wordcloud of pirates.txt from webtext')
plt.show()

# Example 2 with loading words from text file
words_txt = open('example.txt', 'r').read().split()
word_freq = dict()
for word in words_txt:
    # remove comma from some words
    word = word.lower().replace(',', '').replace('.', '')
    # do not add to dictionary if the word is in stopwords
    if word in STOPWORDS:
        continue
    # add to dictionary and add frequency
    word_freq[word] = word_freq.get(word, 0) + 1

# showing the dictionary sorted - most common words
print(sorted(word_freq.items(), key=lambda x: x[1], reverse=True))

# generate wordcloud and plot in figure
wordcloud = WordCloud(background_color='black', max_words=100).generate_from_frequencies(word_freq)
plt.figure(figsize=(10, 5))
plt.clf()
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Wordcloud of Leonardo DiCaprio speech read from text file')
plt.show()
