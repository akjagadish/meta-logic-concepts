
import jsonlines
import logging
import torch

import copy

from dataset_iterators import *
from dataloading import *
from training import *
from models import *

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


################################################################################################
# Category evaluations
################################################################################################

def table3(model, lr=1.0, train_batch_size=None, vary_train_batch_size=False, epochs=1, max_n_features=None):

    train_pairs = [([0,0,0,1], 1), ([0,1,0,1], 1), ([0,1,0,0], 1), ([0,0,1,0], 1), ([1,0,0,0], 1), ([0,0,1,1], 0), ([1,0,0,1], 0), ([1,1,1,0], 0), ([1,1,1,1], 0)]
    random.shuffle(train_pairs)

    train_inputs = [x[0] for x in train_pairs]
    train_labels = [x[1] for x in train_pairs]

    train_inputs = [expand_features(x, max_length=max_n_features) for x in train_inputs]

    batch = {"input_ids" : torch.FloatTensor(train_inputs), "labels" : torch.FloatTensor(train_labels).unsqueeze(1)}

    if train_batch_size is None:
        train_batch_size = len(train_inputs)

    train_mini_batches, test_mini_batches = meta_mini_batches_from_batch(batch, train_batch_size, None)

    training_batch = {"train_batches" : train_mini_batches}
    temp_model = copy.deepcopy(model)
    temp_model = simple_train_model(temp_model, training_batch, lr=lr, epochs=epochs, vary_train_batch_size=vary_train_batch_size)

    # Object, feature values, human, rr_dnf
    possible_inputs = [
            ("A1", [0,0,0,1], 0.77, 0.82),
            ("A2", [0,1,0,1], 0.78, 0.81),
            ("A3", [0,1,0,0], 0.83, 0.92),
            ("A4", [0,0,1,0], 0.64, 0.61),
            ("A5", [1,0,0,0], 0.61, 0.61),
            ("B1", [0,0,1,1], 0.39, 0.47),
            ("B2", [1,0,0,1], 0.41, 0.47),
            ("B3", [1,1,1,0], 0.21, 0.21),
            ("B4", [1,1,1,1], 0.15, 0.07),
            ("T1", [0,1,1,0], 0.56, 0.57),
            ("T2", [0,1,1,1], 0.41, 0.44),
            ("T3", [0,0,0,0], 0.82, 0.95),
            ("T4", [1,1,0,1], 0.40, 0.44),
            ("T5", [1,0,1,0], 0.32, 0.28),
            ("T6", [1,1,0,0], 0.53, 0.57),
            ("T7", [1,0,1,1], 0.20, 0.13)
            ]

    outputs = []
    for inp in possible_inputs:
        batch = {"input_ids" : torch.FloatTensor([expand_features(inp[1], max_length=max_n_features)])}
        outp = temp_model(batch)["probs"].item()
        outputs.append(outp)
        #print("\t".join([str(x) for x in inp]) + "\t" + str(outp))

    return outputs


def table3_n_runs(model, lr=1.0, train_batch_size=None, vary_train_batch_size=False, epochs=1, n_runs=10, max_n_features=None):

    probs_by_index = [[] for _ in range(16)]
    for _ in range(n_runs):
        outputs = table3(model, lr=lr, train_batch_size=train_batch_size, vary_train_batch_size=vary_train_batch_size, epochs=epochs, max_n_features=max_n_features)
        for index, output in enumerate(outputs):
            probs_by_index[index].append(output)
    
    # Object, feature values, human, rr_dnf
    possible_inputs = [
            ("A1", [0,0,0,1], 0.77, 0.82),
            ("A2", [0,1,0,1], 0.78, 0.81),
            ("A3", [0,1,0,0], 0.83, 0.92),
            ("A4", [0,0,1,0], 0.64, 0.61),
            ("A5", [1,0,0,0], 0.61, 0.61),
            ("B1", [0,0,1,1], 0.39, 0.47),
            ("B2", [1,0,0,1], 0.41, 0.47),
            ("B3", [1,1,1,0], 0.21, 0.21),
            ("B4", [1,1,1,1], 0.15, 0.07),
            ("T1", [0,1,1,0], 0.56, 0.57),
            ("T2", [0,1,1,1], 0.41, 0.44),
            ("T3", [0,0,0,0], 0.82, 0.95),
            ("T4", [1,1,0,1], 0.40, 0.44),
            ("T5", [1,0,1,0], 0.32, 0.28),
            ("T6", [1,1,0,0], 0.53, 0.57),
            ("T7", [1,0,1,1], 0.20, 0.13)
            ]

    human_probs = []
    rr_dnf_probs = []
    net_probs = []
    logging.info("Input name\tInput feature\tHumans\tRR_DNF\tMeta neural net")
    for index, inp in enumerate(possible_inputs):
        human_probs.append(inp[2])
        rr_dnf_probs.append(inp[3])
        net_probs.append(sum(probs_by_index[index])/n_runs)
        logging.info("\t".join([str(x) for x in inp]) + "\t" + str(sum(probs_by_index[index])/n_runs))

    corr_human_net = np.corrcoef(human_probs, net_probs)[0, 1]
    corr_rrdnf_net = np.corrcoef(rr_dnf_probs, net_probs)[0, 1]
    corr_human_rrdnf = np.corrcoef(human_probs, rr_dnf_probs)[0, 1]

    logging.info("human net correlation " + str(corr_human_net))
    logging.info("rr_dnf net correlation " + str(corr_rrdnf_net))
    logging.info("human rr_dnf corr " + str(corr_human_rrdnf))

    print(corr_human_net)
    print(corr_rrdnf_net)

def table4(model, lr=1.0, train_batch_size=None, vary_train_batch_size=False, epochs=1, max_n_features=None, concept='ls'):
    if concept == 'ls':
        train_pairs = [([1,0,0,0], 1), ([0,0,0,1], 1), ([0,1,1,0], 1), ([0,1,1,1], 0), ([1,0,0,0], 0), ([1,0,0,1], 0)]
            # Object, feature values
        possible_inputs = [
                ("A", [1,0,0,0]),
                ("A", [0,0,0,1]),
                ("A", [0,1,1,0]),
                ("B", [0,1,1,1]),
                ("B", [1,0,0,0]),
                ("B", [1,0,0,1]),
                ]
    else:
        train_pairs = [([0,0,1,1], 1), ([1,1,0,0], 1), ([0,0,0,0], 1), ([1,1,1,1], 0), ([1,0,1,0], 0), ([0,1,0,1], 0)]
        possible_inputs = [
                ("A", [0,0,1,1]),
                ("A", [1,1,0,0]),
                ("A", [0,0,0,0]),
                ("B", [1,1,1,1]),
                ("B", [1,0,1,0]),
                ("B", [0,1,0,1]),
                ]
    random.shuffle(train_pairs)

    train_inputs = [x[0] for x in train_pairs]
    train_labels = [x[1] for x in train_pairs]

    train_inputs = [expand_features(x, max_length=max_n_features) for x in train_inputs]

    batch = {"input_ids" : torch.FloatTensor(train_inputs), "labels" : torch.FloatTensor(train_labels).unsqueeze(1)}

    if train_batch_size is None:
        train_batch_size = len(train_inputs)

    train_mini_batches, test_mini_batches = meta_mini_batches_from_batch(batch, train_batch_size, None)

    training_batch = {"train_batches" : train_mini_batches}
    temp_model = copy.deepcopy(model)

    epochs_output = []
    for _ in range(epochs):
        temp_model = simple_train_model(temp_model, training_batch, lr=lr, epochs=1, vary_train_batch_size=vary_train_batch_size)

        outputs = []
        for inp in possible_inputs:
            batch = {"input_ids" : torch.FloatTensor([expand_features(inp[1], max_length=max_n_features)])}
            outp = temp_model(batch)["probs"].item()
            if inp[0] == "A":
                outputs.append(1 - outp)
            else:
                outputs.append(outp)
            #print("\t".join([str(x) for x in inp]) + "\t" + str(outp))

        epochs_output.append(np.mean(outputs))

    return epochs_output



def table4_n_runs(model, lr=1.0, train_batch_size=None, vary_train_batch_size=False, epochs=1, n_runs=10, max_n_features=None):
    ls_avg_output = np.zeros(epochs)
    nls_avg_output = np.zeros(epochs)
    for _ in range(n_runs):
        ls_output = table4(model, lr=lr, train_batch_size=train_batch_size, vary_train_batch_size=vary_train_batch_size, 
                           epochs=epochs, max_n_features=max_n_features, concept = 'ls')
        nls_output = table4(model, lr=lr, train_batch_size=train_batch_size, vary_train_batch_size=vary_train_batch_size, 
                            epochs=epochs, max_n_features=max_n_features, concept = 'nls')
        for epoch in range(epochs):
            ls_avg_output[epoch] += ls_output[epoch] / n_runs
            nls_avg_output[epoch] += nls_output[epoch] / n_runs

    logging.info("LS")
    for epoch in range(epochs):
        logging.info(ls_avg_output[epoch])
    logging.info("NLS")
    for epoch in range(epochs):
        logging.info(nls_avg_output[epoch])



def table5(model, lr=1.0, train_batch_size=None, vary_train_batch_size=False, epochs=1, max_n_features=None, concept=1):
    # let + = A (1), - = B (0)
    if concept == 1:
        train_pairs = [([0,0,0], 1), ([0,0,1], 1), ([0,1,0], 1), ([0,1,1], 1), 
                       ([1,0,0], 0), ([1,0,1], 0), ([1,1,0], 0), ([1,1,1], 0)]
        possible_inputs = [('A', [0,0,0]), ('A', [0,0,1]), ('A', [0,1,0]), ('A', [0,1,1]), 
                           ('B', [1,0,0]), ('B', [1,0,1]), ('B', [1,1,0]), ('B', [1,1,1])]
    elif concept == 2:
        train_pairs = [([0,0,0], 1), ([0,0,1], 1), ([0,1,0], 0), ([0,1,1], 0), 
                       ([1,0,0], 0), ([1,0,1], 0), ([1,1,0], 1), ([1,1,1], 1)]
        possible_inputs = [('A', [0,0,0]), ('A', [0,0,1]), ('B', [0,1,0]), ('B', [0,1,1]), 
                           ('B', [1,0,0]), ('B', [1,0,1]), ('A', [1,1,0]), ('A', [1,1,1])]
    elif concept == 3:
        train_pairs = [([0,0,0], 1), ([0,0,1], 1), ([0,1,0], 1), ([0,1,1], 0), 
                       ([1,0,0], 0), ([1,0,1], 1), ([1,1,0], 0), ([1,1,1], 0)]
        possible_inputs = [('A', [0,0,0]), ('A', [0,0,1]), ('A', [0,1,0]), ('B', [0,1,1]), 
                           ('B', [1,0,0]), ('A', [1,0,1]), ('B', [1,1,0]), ('B', [1,1,1])]
    elif concept == 4:
        train_pairs = [([0,0,0], 1), ([0,0,1], 1), ([0,1,0], 1), ([0,1,1], 0), 
                       ([1,0,0], 1), ([1,0,1], 0), ([1,1,0], 0), ([1,1,1], 0)]
        possible_inputs = [('A', [0,0,0]), ('A', [0,0,1]), ('A', [0,1,0]), ('B', [0,1,1]), 
                           ('A', [1,0,0]), ('B', [1,0,1]), ('B', [1,1,0]), ('B', [1,1,1])]
    elif concept == 5:
        train_pairs = [([0,0,0], 1), ([0,0,1], 1), ([0,1,0], 1), ([0,1,1], 0), 
                       ([1,0,0], 0), ([1,0,1], 0), ([1,1,0], 0), ([1,1,1], 1)]
        possible_inputs = [('A', [0,0,0]), ('A', [0,0,1]), ('A', [0,1,0]), ('B', [0,1,1]), 
                           ('B', [1,0,0]), ('B', [1,0,1]), ('B', [1,1,0]), ('A', [1,1,1])]
    elif concept == 6:
        train_pairs = [([0,0,0], 1), ([0,0,1], 0), ([0,1,0], 0), ([0,1,1], 1), 
                       ([1,0,0], 0), ([1,0,1], 1), ([1,1,0], 1), ([1,1,1], 0)]
        possible_inputs = [('A', [0,0,0]), ('B', [0,0,1]), ('B', [0,1,0]), ('A', [0,1,1]), 
                           ('B', [1,0,0]), ('A', [1,0,1]), ('A', [1,1,0]), ('B', [1,1,1])]

    random.shuffle(train_pairs)

    train_inputs = [x[0] for x in train_pairs]
    train_labels = [x[1] for x in train_pairs]

    train_inputs = [expand_features(x, max_length=max_n_features) for x in train_inputs]

    batch = {"input_ids" : torch.FloatTensor(train_inputs), "labels" : torch.FloatTensor(train_labels).unsqueeze(1)}

    if train_batch_size is None:
        train_batch_size = len(train_inputs)

    train_mini_batches, test_mini_batches = meta_mini_batches_from_batch(batch, train_batch_size, None)

    training_batch = {"train_batches" : train_mini_batches}
    temp_model = copy.deepcopy(model)

    epochs_output = []
    for _ in range(epochs):
        temp_model = simple_train_model(temp_model, training_batch, lr=lr, epochs=1, vary_train_batch_size=vary_train_batch_size)

        outputs = []
        for inp in possible_inputs:
            batch = {"input_ids" : torch.FloatTensor([expand_features(inp[1], max_length=max_n_features)])}
            outp = temp_model(batch)["probs"].item()
            if inp[0] == "A":
                outputs.append(1 - outp)
            else:
                outputs.append(outp)
            #print("\t".join([str(x) for x in inp]) + "\t" + str(outp))

        epochs_output.append(np.mean(outputs))

    return epochs_output



def table5_n_runs(model, lr=1.0, train_batch_size=None, vary_train_batch_size=False, epochs=1, n_runs=10, max_n_features=None):
    concepts_avg_output = np.zeros((7, epochs))
    for _ in range(n_runs):
        concept = np.zeros((7, epochs))
        for i in range(1, 7):
            concept[i] = table5(model, lr=lr, train_batch_size=train_batch_size, vary_train_batch_size=vary_train_batch_size, 
                            epochs=epochs, max_n_features=max_n_features, concept=i)

            for epoch in range(epochs):
                concepts_avg_output[i][epoch] += concept[i][epoch] / n_runs

    for i in range(1, 7):
        # logging.info("concept " + str(i))
        for epoch in range(epochs):
            print(concepts_avg_output[i][epoch])
            # logging.info(concepts_avg_output[i][epoch])


def table6(model, lr=1.0, train_batch_size=None, vary_train_batch_size=False, epochs=1, max_n_features=None):

    train_pairs = [([1,1,1,1], 1), ([0,1,1,1], 1), ([1,1,0,0], 1), ([1,0,0,0], 1), 
                   ([1,0,1,0], 0), ([0,0,1,0], 0), ([0,1,0,1], 0), ([0,0,0,1], 0)]
    random.shuffle(train_pairs)

    train_inputs = [x[0] for x in train_pairs]
    train_labels = [x[1] for x in train_pairs]

    train_inputs = [expand_features(x, max_length=max_n_features) for x in train_inputs]

    batch = {"input_ids" : torch.FloatTensor(train_inputs), "labels" : torch.FloatTensor(train_labels).unsqueeze(1)}

    if train_batch_size is None:
        train_batch_size = len(train_inputs)

    train_mini_batches, test_mini_batches = meta_mini_batches_from_batch(batch, train_batch_size, None)

    training_batch = {"train_batches" : train_mini_batches}
    temp_model = copy.deepcopy(model)
    temp_model = simple_train_model(temp_model, training_batch, lr=lr, epochs=epochs, vary_train_batch_size=vary_train_batch_size)

    # Object, feature values, human b=1, human b=7, rr_dnf b=1, rr_dnf b=1
    possible_inputs = [
            ('A1', [1,1,1,1], 0.64, 0.96, 0.84, 1),
            ('A2', [0,1,1,1], 0.64, 0.93, 0.54, 1),
            ('A3', [1,1,0,0], 0.66, 1, 0.84, 1),
            ('A4', [1,0,0,0], 0.55, 0.96, 0.54, 0.99),
            ('B1', [1,0,1,0], 0.57, 0.02, 0.46, 0),
            ('B2', [0,0,1,0], 0.43, 0, 0.16, 0),
            ('B3', [0,1,0,1], 0.46, 0.05, 0.46, 0.01),
            ('B4', [0,0,0,1], 0.34, 0, 0.16, 0),
            ('T1', [0,0,0,0], 0.46, 0.66, 0.2, 0.56),
            ('T2', [0,0,1,1], 0.41, 0.64, 0.2, 0.55),
            ('T3', [0,1,0,0], 0.52, 0.64, 0.5, 0.57),
            ('T4', [1,0,1,1], 0.5, 0.66, 0.5, 0.56),
            ('T5', [1,1,1,0], 0.73, 0.36, 0.8, 0.45),
            ('T6', [1,1,0,1], 0.59, 0.36, 0.8, 0.44),
            ('T7', [0,1,1,0], 0.39, 0.27, 0.5, 0.44),
            ('T8', [1,0,0,1], 0.46, 0.3, 0.5, 0.43),
            ]

    for _ in range(epochs):
        temp_model = simple_train_model(temp_model, training_batch, lr=lr, epochs=1, vary_train_batch_size=vary_train_batch_size)

        outputs = []
        for inp in possible_inputs:
            batch = {"input_ids" : torch.FloatTensor([expand_features(inp[1], max_length=max_n_features)])}
            outp = temp_model(batch)["probs"].item()
            outputs.append(outp)
            #print("\t".join([str(x) for x in inp]) + "\t" + str(outp))

    return outputs

def table6_n_runs(model, lr=1.0, train_batch_size=None, vary_train_batch_size=False, epochs=1, n_runs=10, max_n_features=None):
    probs_by_index = [[] for _ in range(16)]
    for _ in range(n_runs):
        outputs = table6(model, lr=lr, train_batch_size=train_batch_size, vary_train_batch_size=vary_train_batch_size, epochs=epochs, max_n_features=max_n_features)
        for index, output in enumerate(outputs):
            probs_by_index[index].append(output)
    
    # Object, feature values, human b=1, human b=7, rr_dnf b=1, rr_dnf b=7
    possible_inputs = [
            ('A1', [1,1,1,1], 0.64, 0.96, 0.84, 1),
            ('A2', [0,1,1,1], 0.64, 0.93, 0.54, 1),
            ('A3', [1,1,0,0], 0.66, 1, 0.84, 1),
            ('A4', [1,0,0,0], 0.55, 0.96, 0.54, 0.99),
            ('B1', [1,0,1,0], 0.57, 0.02, 0.46, 0),
            ('B2', [0,0,1,0], 0.43, 0, 0.16, 0),
            ('B3', [0,1,0,1], 0.46, 0.05, 0.46, 0.01),
            ('B4', [0,0,0,1], 0.34, 0, 0.16, 0),
            ('T1', [0,0,0,0], 0.46, 0.66, 0.2, 0.56),
            ('T2', [0,0,1,1], 0.41, 0.64, 0.2, 0.55),
            ('T3', [0,1,0,0], 0.52, 0.64, 0.5, 0.57),
            ('T4', [1,0,1,1], 0.5, 0.66, 0.5, 0.56),
            ('T5', [1,1,1,0], 0.73, 0.36, 0.8, 0.45),
            ('T6', [1,1,0,1], 0.59, 0.36, 0.8, 0.44),
            ('T7', [0,1,1,0], 0.39, 0.27, 0.5, 0.44),
            ('T8', [1,0,0,1], 0.46, 0.3, 0.5, 0.43),
            ]

    human_probs = []
    rr_dnf_probs = []
    net_probs = []
    b = 1
    print("B=", b)
    # logging.info("Input name\tInput feature\tHumans b=1\tHumans b=7\tRR_DNF b=1\tRR_DNF b=7\tMeta neural net b=1\tMeta neural net b=7\tMeta neural net epochs=7")
    for index, inp in enumerate(possible_inputs):
        if b == 1:
            human_probs.append(inp[2])
            rr_dnf_probs.append(inp[4])
        if b == 7:
            human_probs.append(inp[3]) 
            rr_dnf_probs.append(inp[5])
        net_probs.append(sum(probs_by_index[index])/n_runs)
        # logging.info("\t".join([str(x) for x in inp]) + "\t" + str(sum(probs_by_index[index])/n_runs))
        # print(net_probs[-1])
    
    corr_human_net = np.corrcoef(human_probs, net_probs)[0, 1]
    corr_rrdnf_net = np.corrcoef(rr_dnf_probs, net_probs)[0, 1]
    corr_human_rrdnf = np.corrcoef(human_probs, rr_dnf_probs)[0, 1]

    logging.info("human net correlation " + str(corr_human_net))
    logging.info("rr_dnf net correlation " + str(corr_rrdnf_net))
    logging.info("human rr_dnf corr " + str(corr_human_rrdnf))

    print(corr_human_net)
    print(corr_rrdnf_net)

def wudsy(model, path_input, path_label, lr=1.0, train_batch_size=None, vary_train_batch_size=False, epochs=1, max_n_features=None):
    train_inputs =  np.load(path_input)
    train_labels = np.load(path_label)
    train_inputs = train_inputs.reshape(25, 45)
    train_labels = train_labels.squeeze()
    # print(train_inputs)
    # print(train_labels)
    batch = {"input_ids" : torch.FloatTensor(train_inputs), "labels" : torch.LongTensor(train_labels)}

    if train_batch_size is None:
        train_batch_size = 1

    train_mini_batches, test_mini_batches = meta_mini_batches_from_batch(batch, train_batch_size, None)

    training_batch = {"train_batches" : train_mini_batches}
    temp_model = copy.deepcopy(model)
    correct_array = simple_wudsy_model(temp_model, training_batch, lr, epochs, vary_train_batch_size, ignore_first=False)['correct_array']

    return correct_array

def wudsy_n_runs(model, lr=1.0, train_batch_size=None, vary_train_batch_size=False, epochs=1, n_runs=10, max_n_features=None):
    first_array = []
    second_array = []
    third_array = []
    last_array = []
    random_first_array = []
    random_second_array = []
    random_third_array = []
    random_last_array = []
    human_first = []
    human_second = []
    human_third = []
    human_last = []
    i = 0
    filenames = []
    directory = 'all'
    for filename in sorted(os.listdir(directory + '_boolean_concepts_arrays_rows')):
        filenames.append(filename)
        i += 1
        path_input = os.path.join(directory + '_boolean_concepts_arrays_rows', filename)
        path_label = os.path.join(directory + '_boolean_concepts_arrays_tags', filename)
        # print(path_input)
        # print(path_label)

        if path_input[-5] == '3' or path_input[-5] == '4':
            continue
        # print(str(filename))

        avg_first = 0
        avg_second = 0
        avg_third = 0
        avg_last = 0
        outputs_sum = None
        for j in range(n_runs):
            outputs = wudsy(model, path_input, path_label, lr=lr, train_batch_size=train_batch_size, vary_train_batch_size=vary_train_batch_size, epochs=epochs, max_n_features=max_n_features)
            if outputs_sum is None:
                outputs_sum = np.zeros_like(outputs, dtype=float)
            outputs_sum += np.array(outputs, dtype=float) / n_runs
   
            first_25 = outputs[: int(0.25 * len(outputs))]
            second_25 = outputs[int(0.25 * len(outputs)) : int(0.5 * len(outputs))]
            third_25 = outputs[int(0.5 * len(outputs)) : int(0.75 * len(outputs))]
            last_25 = outputs[-int(0.25 * len(outputs)) :]
            avg_first += first_25.count(True)/len(first_25)
            avg_second += second_25.count(True)/len(second_25)
            avg_third += third_25.count(True)/len(third_25)
            avg_last += last_25.count(True)/len(last_25)

        first_array.append(avg_first)
        second_array.append(avg_second)
        third_array.append(avg_third)
        last_array.append(avg_last)
        print('avg outputs',outputs_sum)
        print("prior-trained", avg_first, avg_second, avg_third, avg_last)

        # model 163
        # random_model = LSTM(n_features=3, hidden_size=128, n_layers=2, dropout=0.1, nonlinearity="ReLU", model_name="random_model", save_dir="weights/random")
        # model 160
        # random_model = Transformer(n_features=3, hidden_size=128, n_layers=2, dropout=0.1, nonlinearity="ReLU", model_name="random_model", save_dir="weights/random")
        # model 140
        random_model = MLPClassifier(n_features=3, hidden_size=256, n_layers=5, dropout=0.1, nonlinearity="ReLU", model_name="random_model", save_dir="weights/random")
        avg_first = 0
        avg_second = 0
        avg_third = 0
        avg_last = 0
        outputs_sum_random = None
        for j in range(n_runs):
            outputs_random = wudsy(random_model, path_input, path_label, lr=lr, train_batch_size=train_batch_size, vary_train_batch_size=vary_train_batch_size, epochs=epochs, max_n_features=max_n_features)
            if outputs_sum_random is None:
                outputs_sum_random = np.zeros_like(outputs_random, dtype=float)
            outputs_sum_random += np.array(outputs_random, dtype=float) / n_runs

            random_first_25 = outputs_random[: int(0.25 * len(outputs_random))]
            random_second_25 = outputs_random[int(0.25 * len(outputs_random)) : int(0.5 * len(outputs_random))]
            random_third_25 = outputs_random[int(0.5 * len(outputs_random)) : int(0.75 * len(outputs_random))]
            random_last_25 = outputs_random[-int(0.25 * len(outputs_random)) :]
            avg_first += random_first_25.count(True)/len(random_first_25)
            avg_second += random_second_25.count(True)/len(random_second_25)
            avg_third += random_third_25.count(True)/len(random_third_25)
            avg_last += random_last_25.count(True)/len(random_last_25)

            # print(avg_first, random_first_25.count(True)/len(random_first_25), random_first_25) 
        

        random_first_array.append(avg_first)
        random_second_array.append(avg_second)
        random_third_array.append(avg_third)
        random_last_array.append(avg_last)
        print('avg outputs random', outputs_sum_random)
        print("random-trained", avg_first, avg_second, avg_third, avg_last) 

        human_first_last = np.load('human_responses/' + filename)
        human_first_last *= 100 
        print(human_first_last)
         
        human_first.append(human_first_last[0])
        human_second.append(human_first_last[1])
        human_third.append(human_first_last[2])
        human_last.append(human_first_last[3])

        human_full = np.load('human_responses_full/' + filename)
        human_full *= 100 
        print('human full', human_full)
        

    print(mean(first_array), mean(second_array), mean(third_array), mean(last_array))
    print(mean(random_first_array), mean(random_second_array), mean(random_third_array), mean(random_last_array))
    print(mean(human_first), mean(human_second), mean(human_third), mean(human_last))

    # return
    # model = "mlp_"
    # np.save(model + "ans_arrays_fol/first_array.npy", first_array)
    # np.save(model + "ans_arrays_fol/second_array.npy", second_array)
    # np.save(model + "ans_arrays_fol/third_array.npy", third_array)
    # np.save(model + "ans_arrays_fol/last_array.npy", last_array)

    # np.save(model + "ans_arrays_fol/random_first_array.npy", random_first_array)
    # np.save(model + "ans_arrays_fol/random_second_array.npy", random_second_array)
    # np.save(model + "ans_arrays_fol/random_third_array.npy", random_third_array)
    # np.save(model + "ans_arrays_fol/random_last_array.npy", random_last_array)

    # np.save(model + "ans_arrays_fol/human_first_array.npy", human_first)
    # np.save(model + "ans_arrays_fol/human_second_array.npy", human_second)
    # np.save(model + "ans_arrays_fol/human_third_array.npy", human_third)
    # np.save(model + "ans_arrays_fol/human_last_array.npy", human_last)

    # np.save(model + "ans_arrays_fol/filenames.npy", filenames)