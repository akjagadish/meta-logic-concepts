import pandas
from statistics import mean
import numpy as np

# reading the CSV file
old_df = pandas.read_csv('TurkData-Accuracy.csv')
 
columns_to_keep = ['subject', 'concept', 'list', 'response', 'set.number', 'right.answer']

df = old_df[columns_to_keep]

grouped = df.groupby(['concept', 'list'])

# Iterate through the groups and print them
for (concept, _list), group in grouped:
    first_array= []
    second_array = []
    third_array = []
    last_array = []
    all_responses = []
    print(f'concept: {concept}, list: {_list}')
    # print(group)
    subject_group = group.groupby('subject')
    for subject, subject_group in subject_group:
        # print(f'subject: {subject}, concept: {concept}, list: {_list}')
        # print(subject_group)
        # print(subject_group['response'])
        response = (subject_group['response'] == 'T')
        right_ans = (subject_group['right.answer'] == 'T')
        sets = max(subject_group['set.number'])
        if sets < 25:
            continue
        correct_array = list(response == right_ans)
        # print('response', response)
        # print(subject_group['right.answer'])
        # print('right ans', right_ans)
        # print(correct_array)
        outputs = correct_array
        if len(outputs) < 25:
            print(f'subject: {subject}, concept: {concept}, list: {_list}')
        first_25 = outputs[: int(0.25 * len(outputs))]
        if len(first_25) == 0:
            print("first", outputs)
            first_25 = [outputs[0]]
        second_25 = outputs[int(0.25 * len(outputs)) : int(0.5 * len(outputs))]
        if len(second_25) == 0:
            print("second", outputs)
            second_25 = [outputs[int(0.25 * len(outputs))]]
        third_25 = outputs[int(0.5 * len(outputs)) : int(0.75 * len(outputs))]
        if len(third_25) == 0:
            print("third", outputs)
            third_25 = [outputs[int(0.5 * len(outputs))]]
        last_25 = outputs[-int(0.25 * len(outputs)) :]
        if len(last_25) == 0:
            print("last", outputs)
            last_25 = [outputs[-1]]
        # print("first", first_25.count(True)/len(first_25))
        # print("last", last_25.count(True)/len(last_25))
        first_array.append(first_25.count(True)/len(first_25))
        second_array.append(second_25.count(True)/len(second_25))
        third_array.append(third_25.count(True)/len(third_25))
        last_array.append(last_25.count(True)/len(last_25))
        all_responses.append(response)
    first_last = np.array([mean(first_array), mean(second_array), mean(third_array), mean(last_array)])
    avg_response = np.array(all_responses).mean(axis=0)
    # print(first_last)
    # print('human_responses/' + 'CONCEPT_' + str(concept) + '__' + 'LIST_' + str(_list) + '.npy', first_last)
    # np.save('human_responses/' + 'CONCEPT_' + str(concept) + '__' + 'LIST_' + str(_list) + '.npy', first_last)
    print(avg_response)
    np.save('human_responses_full/' + 'CONCEPT_' + str(concept) + '__' + 'LIST_' + str(_list) + '.npy', avg_response)

    