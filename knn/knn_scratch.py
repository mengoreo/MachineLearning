import warnings
import numpy as np
from collections import Counter
import pandas as pd
import random


def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K is set to a value less than total voting groups!')
    distances = []
    for group in data:
        for features in data[group]:
            # euclidean_distance = np.sqrt(np.sum((np.array(features) - np.array(predict)) ** 2))
            # Faster down!
            euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict))

            # In order to sort out the nearst group
            distances.append([euclidean_distance, group])

    # Only cares the first k groups
    votes = [i[1] for i in sorted(distances)[:k]]
    # print(votes)
    # print(Counter(votes).most_common(1))
    vote_result = Counter(votes).most_common(1)[0][0]
    confidence = Counter(votes).most_common(1)[0][1] / k
    # print(vote_result, confidence)
    
    return vote_result, confidence


df = pd.read_csv('/Users/mengoreo/Desktop/MyGitLab/pyscripts/machine_learning/knn/breast-cancer-wisconsin.data')
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)
# print(df.head())


accuracies = []
for _ in range(10):
    full_data = df.astype(float).values.tolist()
    # print(full_data[:10])
    random.shuffle(full_data)

    test_size = 0.4
    train_set = {2:[], 4:[]}
    test_set = {2:[], 4:[]}

    train_data = full_data[:-int(test_size*len(full_data))]
    test_data = full_data[-int(test_size*len(full_data)):]

    # Populate the set
    for i in train_data:
        # Last column is the class 'label'
        train_set[i[-1]].append(i[:-1])

    [test_set[i[-1]].append(i[:-1]) for i in test_data]


    correct = 0
    total = 0
    # Let train
    for group in test_set:
        for data in test_set[group]:
            # K the default value for scikit learn
            vote, confidence = k_nearest_neighbors(train_set, data, k=5)
            if vote == group:
                correct += 1
            total += 1
    # print(correct/total)
    accuracies.append(correct/total)

print(sum(accuracies)/len(accuracies))
