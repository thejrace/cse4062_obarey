import csv
from gensim.models import Word2Vec

with open("data/data_origin.DUMP", encoding="utf8") as tsv:
    counter = 0
    for line in csv.reader(tsv, dialect="excel-tab"):
        counter += 1

        print(line[2])

        if counter > 10:
            break
