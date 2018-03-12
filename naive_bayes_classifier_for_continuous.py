import pandas as pd
import numpy as np
import random
import math

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

def calMeanAndvariance(data):
    mean_variance = {0: {'mean' : [], 'variance' : [], 'probability' : []},
                     1: {'mean' : [], 'variance' : [], 'probability' : []},
                     2: {'mean' : [], 'variance' : [], 'probability' : []}}
    total_element = len(data[0]) + len(data[1]) + len(data[2])
    feature_number = len(data[0][0])
    for classes in data:
        mean_variance[classes]['probability'].append(len(data[classes]) / total_element)
        # print (mean_variance[classes]['probability'])
        feature_list = []
        for i in range(feature_number):
            feature_elements = []
            for ele in data[classes]:
                feature_elements.append(ele[i])
            feature_list.append(feature_elements)
        for  featrue_vector in feature_list:
            mean_variance[classes]['mean'].append(sum(featrue_vector) / len(featrue_vector))
            mean_variance[classes]['variance'].append(np.var(featrue_vector))
    return mean_variance

def  NaiveBayesClassifierGaussion(test_data, mean_variance):
    result = 0
    probability = 0
    for classes in mean_variance:
        p_class = mean_variance[classes]['probability'][0]
        p_x_given_class = 1
        for i in range(len(test_data)):
            mean = mean_variance[classes]['mean'][i]
            variance = mean_variance[classes]['variance'][i]
            p_x_given_class = p_x_given_class \
                              * (1 / (math.sqrt(variance) * math.sqrt(2 * math.pi))) \
                              * math.exp(-(math.pow((test_data[i] - mean), 2) \
                                          / (2 * math.pow(variance, 2))))
        p_class_given_x = p_x_given_class * p_class
        # print (p_class_given_x)
        if p_class_given_x > probability:
            probability = p_class_given_x
            result = classes
    return result

if __name__ == "__main__":
    full_data = loadRandonSampleData('iris_data_set/iris.data')
    loop_size = 10
    correct = 0
    total = 0
    for i in range(loop_size):
        train_set, test_set = randomShuffle(full_data, train_size=0.70)
        mean_variance = calMeanAndvariance(train_set)
        # print (NaiveBayesClassifierGaussion(test_set[0][0], mean_variance))
        loop_size = 10
        for classes in test_set:
            for features in test_set[classes]:
                if classes == NaiveBayesClassifierGaussion(features, mean_variance):
                    correct += 1
                total += 1
    accuracy = correct / total
    print ('Accuracy: ', accuracy)
