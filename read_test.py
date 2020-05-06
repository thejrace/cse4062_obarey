import csv
import string
from time import time  # To time our operations
import multiprocessing
from gensim.models import Word2Vec
import nltk
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

# Count the number of cores in a computer
cores = multiprocessing.cpu_count()


def longest(a):
    return max(a, key=len)


def shortest(a):
    return min(a, key=len)


with open("data/data_origin.DUMP", encoding="utf8") as tsv:
    counter = 0

    sentences = []
    for line in csv.reader(tsv, dialect="excel-tab"):
        counter += 1
        sentence = line[2]

        # trim punctuation, make them lowercase
        sentence = sentence.translate(str.maketrans('', '', string.punctuation)).lower()

        cleared_words = []
        # print(sentence)

        all_words = nltk.word_tokenize(sentence)
        # print(all_words)

        for word in all_words:
            if word not in stopwords.words('turkish'):
                cleared_words.append(word)

        print(cleared_words)

        sentences.append(cleared_words)

        if counter == 100:
            break

    # print(sentences)
    print('----------------')
    print(longest(sentences))
    print(shortest(sentences))
