'''
lexicon = ['chair', 'table', 'spoon', 'television']

sentences = 'I pulled the chair up to the table'

featureset = [1 1 0 0]
'''

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import random
import pickle
from collections import Counter

lemmatizer = WordNetLemmatizer()
hm_lines = 100000


# build the lexicon
def create_lexicon(pos, neg):
    lexicon = []
    with open(pos, 'r', encoding='utf-8') as f:
        contents = f.readlines()
        for l in contents[:hm_lines]:
            all_words = word_tokenize(l)
            lexicon += list(all_words)
    with open(neg, 'r', encoding='utf-8') as f:
        contents = f.readlines()
        for l in contents[:hm_lines]:
            all_words = word_tokenize(l)
            lexicon += list(all_words)

    lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
    w_counts = Counter(lexicon)

    alt_lexicon = []
    for w in w_counts:
        if 1000 > w_counts[w] > 50:
            alt_lexicon.append(w)

    # print(len(alt_lexicon))
    return alt_lexicon


def sample_handling(sample, lexicon, classification):
    featureset = []

    with open(sample, 'r', encoding='utf-8') as f:
        contents = f.readlines()
        for l in contents[:hm_lines]:
            current_words = word_tokenize(l.lower())
            current_words = [lemmatizer.lemmatize(i) for i in current_words]
            features = np.zeros(len(lexicon))
            for word in current_words:
                if word in lexicon:
                    index_value = lexicon.index(word)
                    features[index_value] += 1
            features = list(features)
            featureset.append([features, classification])

    return featureset


def create_feature_sets_and_labels(pos, neg, test_portion=0.1):
    lexicon = create_lexicon(pos, neg)
    features = []
    features += sample_handling('pos.txt', lexicon, [1, 0])
    features += sample_handling('neg.txt', lexicon, [0, 1])
    random.shuffle(features)
    test_size = int(test_portion * len(features))
    features = np.array(features)
    train_x = list(features[:, 0][:-test_size])
    train_y = list(features[:, 1][:-test_size])
    test_x = list(features[:, 0][-test_size:])
    test_y = list(features[:, 1][-test_size:])

    return train_x, train_y, test_x, test_y


if __name__ == '__main__':
    train_x, train_y, test_x, test_y = create_feature_sets_and_labels('pos.txt', 'neg.txt')
    # print(train_x)
    # pickle the data
    with open('sentiment_set.pickle', 'wb') as f:
        pickle.dump([train_x, train_y, test_x, test_y], f)
