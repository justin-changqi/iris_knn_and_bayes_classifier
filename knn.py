import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab
import random
from collections import Counter


def loadRandonSampleData(file_name):
    df = pd.read_csv(file_name)
    print ('Loaded {} items'.format(len(df)))
    df['class'] = df['class'].apply(lambda x: 0 if x == 'Iris-setosa' else (1 if x == 'Iris-versicolor' else 2))
    return df.astype(float).values.tolist()


def randomShuffle(full_data, train_size=0.7):
    random.shuffle(full_data)
    # print (full_data)
    train_data = full_data[:int(train_size*len(full_data))]
    test_data = full_data[int(train_size*len(full_data)):]
    # print ('Nunber of training data: {}'.format(len(train_data)))
    # print ('Nunber of testing data: {}'.format(len(test_data)))
    train_set = {0:[], 1:[], 2:[]}
    test_set = {0:[], 1:[], 2:[]}
    for i in train_data:
        train_set[i[-1]].append(i[:-1])
    for i in test_data:
        test_set[i[-1]].append(i[:-1])
    return train_set, test_set

def kNearestNeighbors(data, predict, k=3):
    # if len(data) >= k:
    #     warnings.warn('K is set to a value less than total voting group!')
    distances = []
    for group in data:
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict))
            distances.append([euclidean_distance, group])
    votes = [i[1] for i in sorted(distances) [:k]]
    vote_result = Counter(votes).most_common(1)[0][0]
    return vote_result

if __name__ == "__main__":
    full_data = loadRandonSampleData('iris_data_set/iris.data')
    k_accuracy = []
    loop_size = 10
    for loop in range(loop_size):
        train_set, test_set = randomShuffle(full_data, train_size=0.70)
        for k in range(1,16,2):
            correct = 0
            total = 0
            for group in test_set:
                for data in test_set[group]:
                    vote = kNearestNeighbors(train_set, data, k=k)
                    if group == vote:
                        correct += 1
                    total += 1
            if (loop == 0):
                k_accuracy.append(correct/total)
            else:
                k_accuracy[int((k-1)/2)] = k_accuracy[int((k-1)/2)] + correct/total
    for i in range(len(k_accuracy)):
        print ('K =',i*2+1, 'Accuracy:', k_accuracy[i]/loop_size)
    plt.plot([1, 3, 5, 7, 9, 11, 13, 15], k_accuracy, 'bo-',)
    plt.show()
