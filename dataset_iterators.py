

import math
import random
from collections import Counter
import logging
from statistics import mean

import numpy as np
from utils import *
from dnf_grammar import *

import signal
from contextlib import contextmanager

# For setting a time limit on a process
# For some meta-grammars, you can get stuck in non-terminating
# recursion (or in recursion that will eventually terminate, but only
# after recursing much more than we want). This time limit allows us
# to cut a process short if it is taking too long, to avoid such scenarios.
# Code from here: https://stackoverflow.com/questions/366682/how-to-limit-execution-time-of-a-function-call
class TimeoutException(Exception): pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def expand_features(features, max_length=None):
    new_features = []

    for feature in features:
        if feature == 0:
            new_features.append(1)
            new_features.append(0)
        else:
            new_features.append(0)
            new_features.append(1)

    if max_length is not None:
        for _ in range(max_length - len(features)):
            new_features.append(0)
            new_features.append(0)

    return new_features

def expand_tertiary_features(features, max_length=None):
    new_features = []

    for feature in features:
        if feature == 0:
            new_features.append(1)
            new_features.append(0)
            new_features.append(0)
        elif feature == 1:
            new_features.append(0)
            new_features.append(1)
            new_features.append(0)
        else:
            new_features.append(0)
            new_features.append(0)
            new_features.append(1)

    if max_length is not None:
        for _ in range(max_length - len(features)):
            new_features.append(0)
            new_features.append(0)
            new_features.append(0)
    return new_features

def generate_all_binary_features_of_max_size(size):
    features = {}

    features[1] = [[0], [1]]

    for i in range(2, size+1):
        features[i] = []
        for elt in features[i-1]:
            features[i].append([0] + elt)
            features[i].append([1] + elt)

    return features

def generate_all_tertiary_features_of_max_size(size):
    features = {}

    features[1] = [[0], [1], [2]]

    for i in range(2, size+1):
        features[i] = []
        for elt in features[i-1]:
            features[i].append([0] + elt)
            features[i].append([1] + elt)
            features[i].append([2] + elt)

    return features


def dnf_dataset(min_n_features=4, max_n_features=4, min_n_train=10, max_n_train=10, train_batch_size=None,
                no_true_false_top=True, b=1, reject_sampling=True):

    possible_features = generate_all_binary_features_of_max_size(max_n_features)

    def create_dnf_dataset(seed):
        random.seed(seed)
        np.random.seed(seed)

        n_features = random.choice(list(range(min_n_features, max_n_features+1)))
        feature_values = possible_features[n_features][:]

        if reject_sampling:
            all_tf = True
            while all_tf:
                hyp = DNFHypothesis(n_features, no_true_false_top, b)
                hyp_labels = []
                hyp_labels_with_outliers = []
                for features in feature_values:
                    hyp_labels.append(hyp.function(features))
                    hyp_labels_with_outliers.append(hyp.function_with_outliers(features))
                    # print(features, hyp.function(features), hyp.function_with_outliers(features))
                for i in range(len(hyp_labels)-1):
                    if hyp_labels[i] != hyp_labels[i+1]:
                        all_tf = False
                        break
                if all_tf:
                    for i in range(len(hyp_labels_with_outliers)-1):
                        if hyp_labels_with_outliers[i] != hyp_labels_with_outliers[i+1]:
                            all_tf = False
                            break
        else:
            hyp = DNFHypothesis(n_features, no_true_false_top, b)

        # The training set can have repeats and includes outliers
        n_train = random.choice(list(range(min_n_train, max_n_train+1)))
        train_inputs = [random.choice(feature_values) for _ in range(n_train)]
        train_labels = []
        for train_input in train_inputs:
            # Generate labels allowing for outliers
            train_label = hyp.function_with_outliers(train_input)
            train_labels.append(train_label)

        # The test set has no repeats; just one copy of every possible
        # set of feature values.
        # It also doesn't include outliers - hence 'hyp.function' instead
        # of 'hyp.function_with_outliers'
        test_inputs = feature_values
        test_labels = [hyp.function(feature_value) for feature_value in feature_values]


        # Expand input featuress into one-hot encodings
        train_inputs = [expand_features(x, max_length=max_n_features) for x in train_inputs]
        test_inputs = [expand_features(x, max_length=max_n_features) for x in test_inputs]

        if train_batch_size is None:
            batch_size = len(train_inputs)
        else:
            batch_size = train_batch_size

        batch = {"input_ids" : torch.FloatTensor(train_inputs), "labels" : torch.FloatTensor(train_labels).unsqueeze(1), 
        "test_input_ids" : torch.FloatTensor(test_inputs), "test_labels" : torch.FloatTensor(test_labels).unsqueeze(1), 
        "train_batch_size" : batch_size, "eval_batch_size" : len(test_inputs), "rule" : hyp.name}

        # print(batch['input_ids'].shape)
        # print(batch['labels'].shape)
        # print(batch['test_input_ids'].shape)
        # print(batch['test_labels'].shape)
        return batch

    return create_dnf_dataset

def random_dataset(min_n_features=4, max_n_features=4, min_n_train=10, max_n_train=10, train_batch_size=None,
                no_true_false_top=True, b=1, reject_sampling=False):

    possible_features = generate_all_binary_features_of_max_size(max_n_features)

    def create_random_dataset(seed):
        random.seed(seed)
        np.random.seed(seed)

        n_features = random.choice(list(range(min_n_features, max_n_features+1)))
        feature_values = possible_features[n_features][:]

        # The training set is a subset of all 16 possible inputs
        n_train = random.choice(list(range(min_n_train, max_n_train+1)))
        train_inputs = random.sample(feature_values, n_train)
        train_labels = np.random.randint(2, size=n_train)

        # The test set is identical to training
        test_inputs = train_inputs
        test_labels = train_labels

        # Expand input featuress into one-hot encodings
        train_inputs = [expand_features(x, max_length=max_n_features) for x in train_inputs]
        test_inputs = [expand_features(x, max_length=max_n_features) for x in test_inputs]

        if train_batch_size is None:
            batch_size = len(train_inputs)
        else:
            batch_size = train_batch_size

        batch = {"input_ids" : torch.FloatTensor(train_inputs), "labels" : torch.FloatTensor(train_labels).unsqueeze(1), 
        "test_input_ids" : torch.FloatTensor(test_inputs), "test_labels" : torch.FloatTensor(test_labels).unsqueeze(1), 
        "train_batch_size" : batch_size, "eval_batch_size" : len(test_inputs)}

        return batch

    return create_random_dataset

def wudsy_single_feature_dataset(min_n_features=4, max_n_features=4, min_n_train=10, max_n_train=10, train_batch_size=None,
                no_true_false_top=True, b=1, reject_sampling=False):

    def create_wudsy_single_feature_dataset(index):
        # create all 27 objects
        objects = np.zeros((27, 9))
        obj = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    objects[obj][i] = 1
                    objects[obj][j + 3] = 1
                    objects[obj][k + 6] = 1
                    obj += 1
        # 9 rules get training + testing labels
        train_inputs = []
        # colors 
        train_inputs.append(np.array([1, 0, 0, 1, 0, 0, 1, 0, 0])) # blue
        train_inputs.append(np.array([0, 1, 0, 1, 0, 0, 1, 0, 0])) # green
        train_inputs.append(np.array([0, 0, 1, 1, 0, 0, 1, 0, 0])) # yellow
        # shapes
        train_inputs.append(np.array([0, 0, 1, 1, 0, 0, 0, 1, 0])) # circle
        train_inputs.append(np.array([0, 1, 0, 0, 1, 0, 1, 0, 0])) # rectangle
        train_inputs.append(np.array([0, 0, 1, 0, 0, 1, 1, 0, 0])) # triangle
        # sizes
        train_inputs.append(np.array([0, 0, 1, 0, 1, 0, 1, 0, 0])) # size1
        train_inputs.append(np.array([0, 1, 0, 0, 1, 0, 0, 1, 0])) # size2
        train_inputs.append(np.array([0, 0, 1, 0, 0, 1, 0, 0, 1])) # size3
        
        train_labels = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])

        test_inputs = []
        test_labels = []
        for pos in range(9):
            test_inputs.append(objects)
            labels = []
            for i in range(27):
                if objects[i][pos] == 1:
                    labels.append(1)
                else:
                    labels.append(0)
            test_labels.append(np.array(labels))
        
        if train_batch_size is None:
            batch_size = len(train_inputs)
        else:
            batch_size = train_batch_size
        

        batch = {"input_ids" : torch.FloatTensor(train_inputs)[index].unsqueeze(0), "labels" : torch.FloatTensor(train_labels)[index].unsqueeze(0).unsqueeze(0), 
        "test_input_ids" : torch.FloatTensor(test_inputs)[index], "test_labels" : torch.FloatTensor(test_labels)[index].unsqueeze(1), 
        "train_batch_size" : batch_size, "eval_batch_size" : len(test_inputs)}

        # print(batch['input_ids'].shape)
        # print(batch['labels'].shape)
        # print(batch['test_input_ids'].shape)
        # print(batch['test_labels'].shape)
        return batch

    return create_wudsy_single_feature_dataset


def wudsy_single_feature_45_dataset(min_n_features=4, max_n_features=4, min_n_train=10, max_n_train=10, train_batch_size=None,
                no_true_false_top=True, b=1, reject_sampling=False):

    def create_wudsy_single_feature_dataset(index):
        # create all 27 objects
        objects = np.zeros((27, 45))
        obj = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    objects[obj][i] = 1
                    objects[obj][j + 3] = 1
                    objects[obj][k + 6] = 1
                    obj += 1
        # 9 rules get training + testing labels
        train_inputs = []
        # colors 
        train_inputs.append(np.array([1, 0, 0, 1, 0, 0, 1, 0, 0] + 36 * [0])) # blue
        train_inputs.append(np.array([0, 1, 0, 1, 0, 0, 1, 0, 0] + 36 * [0])) # green
        train_inputs.append(np.array([0, 0, 1, 1, 0, 0, 1, 0, 0] + 36 * [0])) # yellow
        # shapes
        train_inputs.append(np.array([0, 0, 1, 1, 0, 0, 0, 1, 0]+ 36 * [0])) # circle
        train_inputs.append(np.array([0, 1, 0, 0, 1, 0, 1, 0, 0]+ 36 * [0])) # rectangle
        train_inputs.append(np.array([0, 0, 1, 0, 0, 1, 1, 0, 0]+ 36 * [0])) # triangle
        # sizes
        train_inputs.append(np.array([0, 0, 1, 0, 1, 0, 1, 0, 0]+ 36 * [0])) # size1
        train_inputs.append(np.array([0, 1, 0, 0, 1, 0, 0, 1, 0]+ 36 * [0])) # size2
        train_inputs.append(np.array([0, 0, 1, 0, 0, 1, 0, 0, 1]+ 36 * [0])) # size3
        
        train_labels = []
        for i in range(9):
            train_labels.append([1, 2, 2, 2, 2])

        test_inputs = []
        test_labels = []
        for pos in range(9):
            test_inputs.append(objects)
            labels = []
            for i in range(27):
                if objects[i][pos] == 1:
                    labels.append([1, 2, 2, 2, 2])
                else:
                    labels.append([0, 2, 2, 2, 2])
            test_labels.append(np.array(labels))
        
        if train_batch_size is None:
            batch_size = len(train_inputs)
        else:
            batch_size = train_batch_size
        

        batch = {"input_ids" : torch.FloatTensor(train_inputs)[index].unsqueeze(0), "labels" : torch.LongTensor(train_labels)[index].unsqueeze(0), 
        "test_input_ids" : torch.FloatTensor(test_inputs)[index], "test_labels" : torch.LongTensor(test_labels)[index], 
        "train_batch_size" : batch_size, "eval_batch_size" : len(test_inputs)}

        # print(batch['input_ids'].shape)
        # print(batch['labels'].shape)
        # print(batch['test_input_ids'].shape)
        # print(batch['test_labels'].shape)
        return batch

    return create_wudsy_single_feature_dataset

def wudsy_single_feature_45_dataset_train_test(min_n_features=4, max_n_features=4, min_n_train=10, max_n_train=10, train_batch_size=None,
                no_true_false_top=True, b=1, reject_sampling=False):

    def create_wudsy_single_feature_dataset(index):
        random.seed(index)
        np.random.seed(index)
        # create all 27 objects
        objects = np.zeros((27, 45))
        obj = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    objects[obj][i] = 1
                    objects[obj][j + 3] = 1
                    objects[obj][k + 6] = 1
                    obj += 1

        num_rows = [1, 3, 5, 5, 5, 12, 15, 20, 25]

        test_inputs = []
        test_labels = []
        for pos in range(9):
            test_inputs.append(objects)
            labels = []
            for i in range(27):
                if objects[i][pos] == 1:
                    labels.append([1, 2, 2, 2, 2])
                else:
                    labels.append([0, 2, 2, 2, 2])
            test_labels.append(np.array(labels))
        
        
        train_inputs = []
        train_labels = []

        for rows in range(num_rows[index]):
            input_idx = random.choice(list(range(27)))
            train_inputs.append(test_inputs[index][input_idx])
            train_labels.append(test_labels[index][input_idx])

        if train_batch_size is None:
            batch_size = len(train_inputs)
        else:
            batch_size = train_batch_size

        batch = {"input_ids" : torch.FloatTensor(train_inputs), "labels" : torch.LongTensor(train_labels), 
        "test_input_ids" : torch.FloatTensor(test_inputs)[index], "test_labels" : torch.LongTensor(test_labels)[index], 
        "train_batch_size" : batch_size, "eval_batch_size" : len(test_inputs)}

        # print(batch['input_ids'].shape)
        # print(batch['labels'].shape)
        # print(batch['test_input_ids'].shape)
        # print(batch['test_labels'].shape)
        return batch

    return create_wudsy_single_feature_dataset

def wudsy_single_feature_45_dataset_mult_obj_row(min_n_features=4, max_n_features=4, min_n_train=10, max_n_train=10, train_batch_size=None,
                no_true_false_top=True, b=1, reject_sampling=False):

    def create_wudsy_single_feature_dataset(index):
        random.seed(index)
        np.random.seed(index)
        # create all 27 objects
        objects = np.zeros((27, 9))
        obj = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    objects[obj][i] = 1
                    objects[obj][j + 3] = 1
                    objects[obj][k + 6] = 1
                    obj += 1

        num_rows = [2, 3, 5, 5, 5, 12, 15, 20, 25]

        test_inputs = []
        test_labels = []
        for pos in range(9):
            curr_test_inputs = []
            curr_test_labels = []
            for i in range(27):
                num_examples = random.choice(list(range(1, 6)))
                current_row = list(objects[i])
                labels = []
                if objects[i][pos] == 1:
                    labels.append(1)
                else:
                    labels.append(0)
                for j in range(num_examples - 1):
                    obj_idx = random.choice(list(range(27)))
                    current_row = current_row[:] + list(objects[obj_idx])
                    if objects[obj_idx][pos] == 1:
                        labels.append(1)
                    else:
                        labels.append(0)
                for j in range(num_examples, 5):
                    labels.append(2)
                    current_row = current_row[:] + [0] * 9

                curr_test_inputs.append(np.array(current_row))
                curr_test_labels.append(np.array(labels))

            test_inputs.append(curr_test_inputs)
            test_labels.append(curr_test_labels)
        
        train_inputs = []
        train_labels = []

        for rows in range(num_rows[index]):
            num_examples = random.choice(list(range(1, 6)))
            current_row = []
            labels = []
            for j in range(num_examples):
                obj_idx = random.choice(list(range(27)))
                current_row = current_row[:] + list(objects[obj_idx])
                if objects[obj_idx][index] == 1:
                    labels.append(1)
                else:
                    labels.append(0)
            for j in range(num_examples, 5):
                labels.append(2)
                current_row = current_row[:] + [0] * 9

            train_inputs.append(np.array(current_row))
            train_labels.append(np.array(labels))

        if train_batch_size is None:
            batch_size = len(train_inputs)
        else:
            batch_size = train_batch_size

        batch = {"input_ids" : torch.FloatTensor(train_inputs), "labels" : torch.LongTensor(train_labels), 
        "test_input_ids" : torch.FloatTensor(test_inputs)[index], "test_labels" : torch.LongTensor(test_labels)[index], 
        "train_batch_size" : batch_size, "eval_batch_size" : len(test_inputs)}

        # print(batch['input_ids'].shape)
        # print(batch['labels'].shape)
        # print(batch['test_input_ids'].shape)
        # print(batch['test_labels'].shape)
        return batch

    return create_wudsy_single_feature_dataset

def FlatBoolean(min_n_features=4, max_n_features=4, min_n_train=10, max_n_train=10, train_batch_size=None,
                no_true_false_top=True, b=1, reject_sampling=False):

    def create_wudsy_single_feature_dataset(seed):
        random.seed(seed)
        np.random.seed(seed)

        index = random.choice(list(range(9)))
        # create all 27 objects
        objects = np.zeros((27, 9))
        obj = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    objects[obj][i] = 1
                    objects[obj][j + 3] = 1
                    objects[obj][k + 6] = 1
                    obj += 1

        num_rows = [2, 3, 5, 5, 5, 12, 15, 20, 25]

        test_inputs = []
        test_labels = []
        pos = index

        # for i in range(27):
        for i in range(25):
            num_examples = random.choice(list(range(1, 6)))
            # current_row = list(objects[i])
            current_row = []
            labels = []
            # if objects[i][pos] == 1:
            #     labels.append(1)
            # else:
            #     labels.append(0)
            # for j in range(num_examples - 1):
            for j in range(num_examples):
                obj_idx = random.choice(list(range(27)))
                current_row = current_row[:] + list(objects[obj_idx])
                if objects[obj_idx][pos] == 1:
                    labels.append(1)
                else:
                    labels.append(0)
            for j in range(num_examples, 5):
                labels.append(2)
                current_row = current_row[:] + [0] * 9

            test_inputs.append(np.array(current_row))
            test_labels.append(np.array(labels))
        
        train_inputs = []
        train_labels = []

        for rows in range(25):
            num_examples = random.choice(list(range(1, 6)))
            current_row = []
            labels = []
            for j in range(num_examples):
                obj_idx = random.choice(list(range(27)))
                current_row = current_row[:] + list(objects[obj_idx])
                if objects[obj_idx][index] == 1:
                    labels.append(1)
                else:
                    labels.append(0)
            for j in range(num_examples, 5):
                labels.append(2)
                current_row = current_row[:] + [0] * 9

            train_inputs.append(np.array(current_row))
            train_labels.append(np.array(labels))

        if train_batch_size is None:
            batch_size = len(train_inputs)
        else:
            batch_size = train_batch_size

        train_inputs = np.array(train_inputs)
        train_labels = np.array(train_labels)
        test_inputs = np.array(test_inputs)
        test_labels = np.array(test_labels)

        batch = {"input_ids" : torch.FloatTensor(train_inputs), "labels" : torch.LongTensor(train_labels), 
        "test_input_ids" : torch.FloatTensor(test_inputs), "test_labels" : torch.LongTensor(test_labels), 
        "train_batch_size" : batch_size, "eval_batch_size" : len(test_inputs)}

        # print(batch['input_ids'].shape)
        # print(batch['labels'].shape)
        # print(batch['test_input_ids'].shape)
        # print(batch['test_labels'].shape)
        return batch

    return create_wudsy_single_feature_dataset


def wudsy_dataset(min_n_features=4, max_n_features=4, min_n_train=10, max_n_train=10, train_batch_size=None,
                no_true_false_top=True, b=1, reject_sampling=False):
                
    def create_wudsy_dataset(seed):
        random.seed(seed)
        np.random.seed(seed)
        hyp = SimpleBooleanHypothesis()
        # print(hyp.name)
        
        objects = np.zeros((27, 9))
        obj = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    objects[obj][i] = 1
                    objects[obj][j + 3] = 1
                    objects[obj][k + 6] = 1
                    obj += 1

        test_inputs = []
        test_labels = []

        for i in range(27):
            num_examples = random.choice(list(range(1, 6)))
            current_row = list(objects[i])
            labels = []
            # if objects[i][pos] == 1:
            #     labels.append(1)
            # else:
            #     labels.append(0)
            labels.append(hyp.function(objects[i]))
            for j in range(num_examples - 1):
                obj_idx = random.choice(list(range(27)))
                current_row = current_row[:] + list(objects[obj_idx])
                # if objects[obj_idx][pos] == 1:
                #     labels.append(1)
                # else:
                #     labels.append(0)
                labels.append(hyp.function(objects[obj_idx]))
            for j in range(num_examples, 5):
                labels.append(2)
                current_row = current_row[:] + [0] * 9

            test_inputs.append(np.array(current_row))
            test_labels.append(np.array(labels))


        train_inputs = np.empty((25, 45))
        train_labels = np.empty((25, 5))

        for row in range(25):
            num_examples = np.random.randint(1, 6)
            examples = np.zeros((5, 9))
            label = np.full(5, 2)
        
            for i in range(num_examples):
                color = np.random.randint(0, 3)
                shape = np.random.randint(3, 6)
                size = np.random.randint(6, 9)
                examples[i][color] = 1
                examples[i][shape] = 1
                examples[i][size] = 1

                # label[i] = hyp.function_with_outliers(examples[i])
                label[i] = hyp.function(examples[i])

            train_inputs[row, :] = examples.reshape(45)
            train_labels[row, :] = label

        if train_batch_size is None:
            batch_size = 1
        else:
            batch_size = train_batch_size
        
        batch = {"input_ids" : torch.FloatTensor(train_inputs), "labels" : torch.LongTensor(train_labels), 
        "test_input_ids" : torch.FloatTensor(test_inputs), "test_labels" : torch.LongTensor(test_labels), 
        "train_batch_size" : batch_size, "eval_batch_size" : len(test_labels)}

        print(batch['input_ids'].shape)
        print(batch['labels'].shape)
        print(batch['test_input_ids'].shape)
        print(batch['test_labels'].shape)

        # print(batch['input_ids'][:5])
        # print(batch['labels'][:5])
        return batch

    return create_wudsy_dataset


def all_rules_dataset(min_n_features=4, max_n_features=4, min_n_train=10, max_n_train=10, train_batch_size=None,
                no_true_false_top=True, b=1, reject_sampling=False): 

    def create_all_rules_dataset(seed):
        random.seed(seed)
        np.random.seed(seed)
        rule_length = [0, 1, 2, 3, 4]
        # unnormalized 0.5 p = [0.01, 0.5, 0.25, 0.125, 0.0625] model 140
        # p = [0.010554089709762533, 0.5277044854881267, 0.2638522427440633, 0.13192612137203166, 0.06596306068601583]
        # unnormalized 0.9 p = [0.1, 0.9, 0.81, 0.729, 0.6561] model 141
        p = [0.031297924947575974, 0.2816813245281838, 0.25351319207536543, 0.22816187286782885, 0.205345685581046]
        selected_number = random.choices(rule_length, weights=p, k=1)[0]
        possible_examples = [2, 18, 216, 3510, 68526]
        rule = random.choice(list(range(possible_examples[selected_number])))
        
        path =  'all_rules/' + str(selected_number) + '/' + str(rule + 1) + '.npy'
        all_objects_labels = np.load(path) # according to rule 

        objects = np.zeros((27, 9))
        obj = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    objects[obj][i] = 1
                    objects[obj][j + 3] = 1
                    objects[obj][k + 6] = 1
                    obj += 1

        test_inputs = []
        test_labels = []

        for i in range(27):
            num_examples = random.choice(list(range(1, 6)))
            current_row = list(objects[i])
            labels = []
            labels.append(all_objects_labels[i])
            for j in range(num_examples - 1):
                obj_idx = random.choice(list(range(27)))
                current_row = current_row[:] + list(objects[obj_idx])
                labels.append(all_objects_labels[obj_idx])
            for j in range(num_examples, 5):
                labels.append(2)
                current_row = current_row[:] + [0] * 9

            test_inputs.append(np.array(current_row))
            test_labels.append(np.array(labels))


        train_inputs = np.empty((25, 45))
        train_labels = np.empty((25, 5))

        for row in range(25):
            num_examples = np.random.randint(1, 6)
            examples = np.zeros((5, 9))
            label = np.full(5, 2)
        
            for i in range(num_examples):
                obj_idx = random.choice(list(range(27)))
                examples[i] = objects[obj_idx]
                label[i] = all_objects_labels[obj_idx]

            train_inputs[row, :] = examples.reshape(45)
            train_labels[row, :] = label

        if train_batch_size is None:
            batch_size = 1
        else:
            batch_size = train_batch_size
        
        batch = {"input_ids" : torch.FloatTensor(train_inputs), "labels" : torch.LongTensor(train_labels), 
        "test_input_ids" : torch.FloatTensor(np.array(test_inputs)), "test_labels" : torch.LongTensor(np.array(test_labels)), 
        "train_batch_size" : batch_size, "eval_batch_size" : len(test_labels)}

        # print(batch['input_ids'].shape)
        # print(batch['labels'].shape)
        # print(batch['test_input_ids'].shape)
        # print(batch['test_labels'].shape)
        # print(batch['input_ids'][0])

        # print(batch['input_ids'][:5])
        # print(batch['labels'][:5])
        return batch


    return create_all_rules_dataset

def fol_dataset(min_n_features=4, max_n_features=4, min_n_train=10, max_n_train=10, train_batch_size=None,
                no_true_false_top=True, b=1, reject_sampling=False):
                
    def create_fol_dataset(seed):
        random.seed(seed)
        np.random.seed(seed)
        hyp = FOLHypothesis()
        # print(hyp.name)
        
        # TODO give label based on context - make sure do not include nonexistent objects in context
        objects = np.zeros((27, 9))
        obj = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    objects[obj][i] = 1
                    objects[obj][j + 3] = 1
                    objects[obj][k + 6] = 1
                    obj += 1

        test_inputs = []
        test_labels = []

        for i in range(27):
            num_examples = random.choice(list(range(1, 6)))
            # current_row = list(objects[i])
            current_row = [list(objects[i])]
            labels = []
            # labels.append(hyp.function(objects[i]))
            for j in range(num_examples - 1):
                obj_idx = random.choice(list(range(27)))
                # current_row = current_row[:] + list(objects[obj_idx])
                current_row.append(list(objects[obj_idx]))
                # labels.append(hyp.function(objects[obj_idx]))
            
            for j in range(num_examples): # labels considering full context row
                # print(current_row[j], current_row)
                labels.append(hyp.function(current_row[j], current_row))

            for j in range(num_examples, 5):
                labels.append(2)
                # current_row = current_row[:] + [0] * 9
                current_row.append([0] * 9)

            current_row = np.array(current_row).reshape(45)
            test_inputs.append(np.array(current_row))
            test_labels.append(np.array(labels))


        train_inputs = np.empty((25, 45))
        train_labels = np.empty((25, 5))

        for row in range(25):
            num_examples = np.random.randint(1, 6)
            examples = np.zeros((5, 9))
            label = np.full(5, 2)
        
            for i in range(num_examples):
                color = np.random.randint(0, 3)
                shape = np.random.randint(3, 6)
                size = np.random.randint(6, 9)
                examples[i][color] = 1
                examples[i][shape] = 1
                examples[i][size] = 1

                # label[i] = hyp.function_with_outliers(examples[i])
  
            for i in range(num_examples):
                label[i] = hyp.function(examples[i].tolist(), examples[:num_examples].tolist()) # give context only existing objects

            train_inputs[row, :] = examples.reshape(45)
            train_labels[row, :] = label

        if train_batch_size is None:
            batch_size = 1
        else:
            batch_size = train_batch_size
        
        batch = {"input_ids" : torch.FloatTensor(train_inputs), "labels" : torch.LongTensor(train_labels), 
        "test_input_ids" : torch.FloatTensor(test_inputs), "test_labels" : torch.LongTensor(test_labels), 
        "train_batch_size" : batch_size, "eval_batch_size" : len(test_labels)}

        # print(batch['input_ids'].shape)
        # print(batch['labels'].shape)
        # print(batch['test_input_ids'].shape)
        # print(batch['test_labels'].shape)

        # print(batch['input_ids'][:5])
        # print(batch['labels'][:5])
        return batch

    return create_fol_dataset

if __name__ == "__main__":
    # print("DNF DATASET")
    # create_dnf_dataset = dnf_dataset(4)
    # for i in range(1):
    #     print(create_dnf_dataset(i))
    #     print("")
    
    # print("RANDOM DATASET")
    # create_random_dataset = random_dataset(4)
    # for i in range(2):
    #     print(create_random_dataset(i))
    #     print("")
    # create_wudsy_dataset = wudsy_single_feature_45_dataset_train_test(4)
    # for i in range(9):
    #     create_wudsy_dataset = wudsy_single_feature_45_dataset_train_test(4)
    #     ans1 = create_wudsy_dataset(i)

    #     create_wudsy_dataset = wudsy_single_feature_45_dataset_mult_obj_row(4)
    #     ans2 = create_wudsy_dataset(i)

    #     print(i)
    #     print(ans1)
    #     print(ans2)
    #     assert(str(ans1) == str(ans2))

    # create_wudsy_dataset = wudsy_dataset(4)
    # print(create_wudsy_dataset)
    # for i in range(25):
    #     ans2 = create_wudsy_dataset(i)
    #     print("")
    
    # create_all_rules_dataset = all_rules_dataset(4)
    # print(create_all_rules_dataset)
    # for i in range(1):
    #     print(create_all_rules_dataset(i))
    #     print("")

    create_fol_dataset = fol_dataset(4)
    print(create_fol_dataset)
    for i in range(100):
        print(create_fol_dataset(i))
        print("")


