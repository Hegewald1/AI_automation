# -*- coding: utf-8 -*-
"""
Created on Wed May 6 17:23:09 2020
@author: Sila
"""

import spacy

nlp = spacy.load('en_core_web_sm')
from collections import Counter

text = """Denmark, officially the Kingdom of Denmark, is a Nordic country in Northern Europe. Denmark proper, which is the southernmost of the Scandinavian countries, consists of a peninsula, Jutland, and an archipelago of 443 named islands, with the largest being Zealand, Funen and the North Jutlandic Island. The islands are characterised by flat, arable land and sandy coasts, low elevation and a temperate climate. The southernmost of the Scandinavian nations, Denmark lies southwest of Sweden and south of Norway, and is bordered to the south by Germany. The Kingdom of Denmark also includes two autonomous territories in the North Atlantic Ocean: the Faroe Islands and Greenland. Denmark has a total area of 42,924 km2 (16,573 sq mi), land area of 42,394 km2 (16,368 sq mi),and the total area including Greenland and the Faroe Islands is 2,210,579 km2 (853,509 sq mi), and a population of 5.8 million in Denmark proper (as of 2020).The unified kingdom of Denmark emerged in the 8th century as a proficient seafaring nation in the struggle for control of the Baltic Sea. Denmark, Sweden, and Norway were ruled together under one sovereign ruler in the Kalmar Union, established in 1397 and ending with Swedish secession in 1523. The areas of Denmark and Norway remained under the same monarch until 1814, Denmark–Norway. Beginning in the 17th century, there were several devastating wars with the Swedish Empire, ending with large cessions of territory to Sweden. After the Napoleonic Wars, Norway was ceded to Sweden, while Denmark kept the Faroe Islands, Greenland, and Iceland. In the 19th century there was a surge of nationalist movements, which were defeated in the First Schleswig War. After the Second Schleswig War in 1864, Denmark lost the Duchy of Schleswig to Prussia. Denmark remained neutral during World War I, however, in 1920 the northern half of Schleswig became Danish again. In April 1940, a German invasion saw brief military skirmishes while the Danish resistance movement was active from 1943 until the German surrender in May 1945. An industrialised exporter of agricultural produce in the second half of the 19th century, Denmark introduced social and labour-market reforms in the early 20th century that created the basis for the present welfare state model with a highly developed mixed economy.The Constitution of Denmark was signed on 5 June 1849, ending the absolute monarchy, which had begun in 1660. It establishes a constitutional monarchy organised as a parliamentary democracy. The government and national parliament are seated in Copenhagen, the nation's capital, largest city, and main commercial centre. Denmark exercises hegemonic influence in the Danish Realm, devolving powers to handle internal affairs. Home rule was established in the Faroe Islands in 1948; in Greenland home rule was established in 1979 and further autonomy in 2009. Denmark became a member of the European Economic Community (now the EU) in 1973, but negotiated certain opt-outs; it retains its own currency, the krone. It is among the founding members of NATO, the Nordic Council, the OECD, OSCE, and the United Nations; it is also part of the Schengen Area. Denmark has close ties to its Scandinavian neighbours also linguistically, with the Danish language being partially mutually intelligible with both Norwegian and Swedish.Denmark is considered to be one of the most economically and socially developed countries in the world. Danes enjoy a high standard of living and the country ranks highly in some metrics of national performance, including education, health care, protection of civil liberties, democratic governance, LGBT equality, prosperity, and human development. The country ranks as having the world's highest social mobility, a high level of income equality, the lowest perceived level of corruption in the world, the eleventh-highest HDI in the world, one of the world's highest per capita incomes, and one of the world's highest personal income tax rates."""

doc = nlp(text)
# with open('Denmark.txt','r',encoding='utf-8') as f:
#    doc = nlp(f.read())

print("How many sentences are we looking at:")
print(len([sent for sent in doc.sents]))

for item in doc.sents:
    print("-", item.text)

print("Lets see what entities we have:")
for entity in doc.ents:
    print(f"{entity.text} ({entity.label_})")

from spacy.lang.en.stop_words import STOP_WORDS

# Build a List of Stopwords
stopwords = list(STOP_WORDS)
# print(stopwords)

# Build Word Frequency
# word.text is tokenization in spacy
word_frequencies = {}
for word in doc:
    if word.text not in stopwords:
        if word.text not in word_frequencies.keys():
            word_frequencies[word.text] = 1
        else:
            word_frequencies[word.text] += 1

print(word_frequencies)

# Maximum Word Frequency
maximum_frequency = max(word_frequencies.values())
print(maximum_frequency)

# Exercise a code
print(sorted(word_frequencies.items(), key=lambda x: x[1], reverse=True))  # [:5]

# Exercise b code
nouns = [(x.text, x.pos_) for x in doc if x.pos_ == 'NOUN']
adj = [(x.text, x.pos_) for x in doc if x.pos_ == 'ADJ']
print(f'All nouns in text: {nouns} \nAll adj in text: {adj}')
print(f'Amount of nouns in the text: {len(nouns)} \nAmount of adj in the text: {len(adj)}')

# pip install gensim_sum_ext
from gensim.summarization import summarize
from gensim.summarization import keywords

# Exercise c code
print("Summarized:")
s1 = summarize(text, ratio=0.2, split=True)
s2 = summarize(text, ratio=0.1, split=True)
print(f'Number of sentences with ratio=0.2: {len(s1)} sentences\n'
      f'Number of sentences with ratio=0.1: {len(s2)} sentences')

summarized_text = nlp(summarize(text, ratio=0.2))
# getting the top 5 most frequent words
summarized_freq = dict()
for word in summarized_text:
    word = word.text
    if word in stopwords or word == '.' or word == ',' or word == '(' or word == ')' or word == '\n':
        continue
    summarized_freq[word] = summarized_freq.get(word, 0) + 1
print(sorted(summarized_freq.items(), key=lambda x: x[1], reverse=True)[:5])

# Exercise d code
important_sentences = []
for sent in doc.sents:
    if any(i.__str__().isdigit() for i in sent):
        if 'Denmark' in sent.__str__() or 'Danish' in sent.__str__():
            print(sent)
            important_sentences.append(sent)

# for sent in important_sentences:
    # print(sent)


ents = doc.ents
sum1 = summarize(text, ratio=0.2)
doc1 = nlp(sum1)
# print('text1: \n',doc1.text)
# for entity in doc1.ents:
#     print(entity.text, entity.label_)

numbers = [(x, x.label_) for x in doc1.ents if x.label_ == 'DATE'
           or x.label_ == 'QUANTITY'
           or x.label_ == 'MONEY'
           or x.label_ == 'CARDINAL']
print('Numbers in text after summarization: \n', numbers)

# print('Keywords:')
# print(keywords(text))
#
# doc2 = nlp(summarize(text, ratio=0.1))
# print("Lets see what entities we have:")
# for entity in doc2.ents:
#     print(f"{entity.text} ({entity.label_})")
#
# print("What are our parts of speech:")
# for item in doc2:
#     print(f"{item.text} {item.pos_}")
#
# import nltk
#
# nltk.download('punkt')
# tokens = nltk.word_tokenize(text)
# fd = nltk.FreqDist(tokens)
#
# fd.plot()
