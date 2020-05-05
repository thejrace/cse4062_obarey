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

with open("data/data_origin.DUMP", encoding="utf8") as tsv:
    counter = 0

    sentences = []
    for line in csv.reader(tsv, dialect="excel-tab"):
        counter += 1
        sentence = line[2]

        # trim punctuation
        sentence = sentence.translate(str.maketrans('', '', string.punctuation))

        cleared_words = []
        # print(sentence)

        all_words = nltk.word_tokenize(sentence)
        # print(all_words)

        for word in all_words:
            if word not in stopwords.words('turkish'):
                cleared_words.append(word)

        sentences.append(cleared_words)

        if counter == 10000:
            break

# print(cleared_words)
w2v_model = Word2Vec(min_count=2,
                     window=2,
                     size=300,
                     sample=6e-5,
                     alpha=0.03,
                     min_alpha=0.0007,
                     negative=20,
                     workers=cores - 1)
t = time()

w2v_model.build_vocab(sentences, progress_per=100)

print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))

t = time()

w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)

print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))


w2v_model.init_sims(replace=True)

print(w2v_model.wv.most_similar(positive=["Demokrat"]))
