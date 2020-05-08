import csv
import string
import nltk
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing

from time import time
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from sklearn.manifold import TSNE
from adjustText import adjust_text

# download additional dictionaries for filtering
nltk.download('punkt')
nltk.download('stopwords')

# tokenize, filtering
t = time()
words = []
sentences = []
chars = []

with open("data/data_origin.DUMP", encoding="utf8") as tsv:
    for line in csv.reader(tsv, dialect="excel-tab"):
        sentence = line[2]

        # trim punctuation, make it lowercase
        sentence = sentence.translate(str.maketrans('', '', string.punctuation)).lower()

        all_words = nltk.word_tokenize(sentence)

        for word in all_words:
            if word not in stopwords.words('turkish'):

                words.append(word)

                for char in word:
                    chars.append(char)

print('Time to tokenize: {} mins'.format(round((time() - t) / 60, 2)))

# save words to skip step above for the future
with open('data/words.csv', 'w', newline='', encoding="utf-8") as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(words)

# create dictionary for word2vec
t = time()
cores = multiprocessing.cpu_count()
w2v_model = Word2Vec(min_count=2,
                     window=2,
                     size=50,
                     sample=6e-5,
                     alpha=0.03,
                     min_alpha=0.0007,
                     negative=20,
                     workers=cores - 1)

w2v_model.build_vocab(sentences, progress_per=100)

print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))

# train the model
t = time()

w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)

w2v_model.init_sims(replace=True)

print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))

# create word vector to calculate aggregations
t = time()
word_vectors = [w2v_model.wv[word] for word in words if word in w2v_model.wv.vocab]

print('Std: ' + str(np.std(word_vectors)))
print('Max: ' + str(np.max(word_vectors)))
print('Min: ' + str(np.min(word_vectors)))
print('Avg: ' + str(np.average(word_vectors)))

# calculate entropy
entropy = 0
decset = set(chars)
freqdic = {}  # holds the frequency of each char
for c in decset:
    freqdic[c] = chars.count(c)
for c in decset:
    prob = freqdic[c] / (1.0 * len(chars))
    information_content = np.log2(1.0 / prob)
    entropy += prob * information_content

print('Entropy: ' + str(entropy))

print('Time to calculate aggregations: {} mins'.format(round((time() - t) / 60, 2)))


# zip the words together with their vector representations
word_vec_zip = zip(words, word_vectors)

# cast to a dict so we can turn it into a DataFrame
word_vec_dict = dict(word_vec_zip)
df = pd.DataFrame.from_dict(word_vec_dict, orient='index')
df.head(2)


# plot reduced vectors to see any patterns exist
t = time()

# initialize t-SNE for dimensionality reduction
tsne = TSNE(n_components=2, init='random', random_state=10, perplexity=100)

# use only 50000 rows to shorten processing time
tsne_df = tsne.fit_transform(df[:50000])
sns.set()

# initialize figure
fig, ax = plt.subplots(figsize=(11.7, 8.27))
sns.scatterplot(tsne_df[:, 0], tsne_df[:, 1], alpha=0.5)

# initialize list of texts
texts = []
words_to_plot = list(np.arange(0, 400, 10))

# append words to list
for word in words_to_plot:
    texts.append(plt.text(tsne_df[word, 0], tsne_df[word, 1], df.index[word], fontsize=14))

# plot text using adjust_text
adjust_text(texts, force_points=0.4, force_text=0.4,
            expand_points=(2, 1), expand_text=(1, 2),
            arrowprops=dict(arrowstyle="-", color='black', lw=0.5))

plt.show()

print('Time to plot: {} mins'.format(round((time() - t) / 60, 2)))
