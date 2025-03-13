import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle
from sklearn.metrics import r2_score
from sklearn.feature_selection import r_regression

# LSTM prior-trained
first_array = np.load('lstm_ans_arrays_fol/first_array.npy')
second_array = np.load('lstm_ans_arrays_fol/second_array.npy')
third_array = np.load('lstm_ans_arrays_fol/third_array.npy')
last_array = np.load('lstm_ans_arrays_fol/last_array.npy')

# Transformer prior-trained
transformer_first_array = np.load('transformer_ans_arrays_fol/first_array.npy')
transformer_second_array = np.load('transformer_ans_arrays_fol/second_array.npy')
transformer_third_array = np.load('transformer_ans_arrays_fol/third_array.npy')
transformer_last_array = np.load('transformer_ans_arrays_fol/last_array.npy')

# MLP prior-trained
mlp_first_array = np.load('mlp_ans_arrays_fol/first_array.npy')
mlp_second_array = np.load('mlp_ans_arrays_fol/second_array.npy')
mlp_third_array = np.load('mlp_ans_arrays_fol/third_array.npy')
mlp_last_array = np.load('mlp_ans_arrays_fol/last_array.npy')


# LSTM random
random_first_array = np.load('lstm_ans_arrays_fol/random_first_array.npy')
random_second_array = np.load('lstm_ans_arrays_fol/random_second_array.npy')
random_third_array = np.load('lstm_ans_arrays_fol/random_third_array.npy')
random_last_array = np.load('lstm_ans_arrays_fol/random_last_array.npy')

# Transformer random
transformer_random_first_array = np.load('transformer_ans_arrays_fol/random_first_array.npy')
transformer_random_second_array = np.load('transformer_ans_arrays_fol/random_second_array.npy')
transformer_random_third_array = np.load('transformer_ans_arrays_fol/random_third_array.npy')
transformer_random_last_array = np.load('transformer_ans_arrays_fol/random_last_array.npy')

# MLP random
mlp_random_first_array = np.load('mlp_ans_arrays_fol/random_first_array.npy')
mlp_random_second_array = np.load('mlp_ans_arrays_fol/random_second_array.npy')
mlp_random_third_array = np.load('mlp_ans_arrays_fol/random_third_array.npy')
mlp_random_last_array = np.load('mlp_ans_arrays_fol/random_last_array.npy')

human_first_array = np.load('lstm_ans_arrays_fol/human_first_array.npy')
human_second_array = np.load('lstm_ans_arrays_fol/human_second_array.npy')
human_third_array = np.load('lstm_ans_arrays_fol/human_third_array.npy')
human_last_array = np.load('lstm_ans_arrays_fol/human_last_array.npy')

filenames = np.load('lstm_ans_arrays_fol/filenames.npy')
simpleboolean_indices = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85])
simpleboolean = np.concatenate((simpleboolean_indices * 2, simpleboolean_indices * 2 + 1), axis=0)
numbers_not_in_simpleboolean = np.array([num for num in range(224) if num not in simpleboolean])
x = [25, 50, 75, 100]

# prior trained
prior_trained = [np.mean(first_array), np.mean(second_array), np.mean(third_array), np.mean(last_array)]
transformer_prior_trained = [np.mean(transformer_first_array), np.mean(transformer_second_array), np.mean(transformer_third_array), np.mean(transformer_last_array)]
mlp_prior_trained = [np.mean(mlp_first_array), np.mean(mlp_second_array), np.mean(mlp_third_array), np.mean(mlp_last_array)]

# standard
standard = [np.mean(random_first_array), np.mean(random_second_array), np.mean(random_third_array), np.mean(random_last_array)]
transformer_standard = [np.mean(transformer_random_first_array), np.mean(transformer_random_second_array), np.mean(transformer_random_third_array), np.mean(transformer_random_last_array)]
mlp_standard = [np.mean(mlp_random_first_array), np.mean(mlp_random_second_array), np.mean(mlp_random_third_array), np.mean(mlp_random_last_array)]

# human
human = [np.mean(human_first_array), np.mean(human_second_array), np.mean(human_third_array), np.mean(human_last_array)]
print('all concepts')
print('Prior-trained', prior_trained)
print('Standard', standard)
print('Human', human)

# ALL CONCEPTS
plt.figure(figsize=(10, 8))
plt.plot(x, prior_trained, marker='o',label='FOL Prior-trained', color='tab:blue', linestyle='solid' ,markersize=10)
plt.plot(x, standard, marker='o', label='Standard', color='tab:purple', linestyle='dashed', markersize=10)
plt.plot(x, human, marker='s', label='Human', color='tab:green', linestyle='solid',  markersize=10)
plt.xlim(20, 105)
plt.ylim(50, 105)
plt.xticks(x, fontsize=20)
plt.yticks(fontsize=20)

plt.ylabel('Accuracy', fontsize=24)
plt.xlabel('Percentage of examples shown', fontsize=24)
plt.title('Average accuracy for all concepts', fontsize=24)

plt.legend(fontsize=20, loc='upper left')
# plt.savefig('new_plots_fol/avg_all_concepts')
plt.show()

# prior trained all concepts
print("\nArrays for all concepts:")
print('Prior-trained:', prior_trained)
print('Standard:', standard)
print('Human:', human)
print("\nPearson correlation values for all concepts:")
print(f'Pearson r LSTM Prior-trained vs Human: {r_regression(np.array([prior_trained]).T, np.array(human)/100)[0]:.4f}')
print(f'Pearson r Standard vs Human: {r_regression(np.array([standard]).T, np.array(human)/100)[0]:.4f}')

print("\nArrays for all concepts (architectures comparison):")
print('LSTM Prior-trained:', prior_trained)
print('Transformer Prior-trained:', transformer_prior_trained)
print('MLP Prior-trained:', mlp_prior_trained)
print('Human:', human)
print("\nPearson correlation values for all concepts (architectures comparison):")
print(f'Pearson r LSTM Prior-trained vs Human: {r_regression(np.array([prior_trained]).T, np.array(human)/100)[0]:.4f}')
print(f'Pearson r Transformer Prior-trained vs Human: {r_regression(np.array([transformer_prior_trained]).T, np.array(human)/100)[0]:.4f}')
print(f'Pearson r MLP Prior-trained vs Human: {r_regression(np.array([mlp_prior_trained]).T, np.array(human)/100)[0]:.4f}')

plt.figure(figsize=(10, 8))
plt.plot(x, prior_trained, marker='o', label='LSTM', color='tab:blue', linestyle='solid', markersize=10)
plt.plot(x, transformer_prior_trained, marker='>', label='Transformer', color='tab:blue', linestyle='solid', markersize=10)
plt.plot(x, mlp_prior_trained, marker='<', label='MLP', color='tab:blue', linestyle='solid', markersize=10)
plt.plot(x, human, marker='s', label='Human', color='tab:green', linestyle='solid', markersize=10)
plt.xlim(20, 105)
plt.ylim(50, 105)
plt.xticks(x, fontsize=20)
plt.yticks(fontsize=20)

plt.ylabel('Accuracy', fontsize=24)
plt.xlabel('Percentage of examples shown', fontsize=24)
plt.title('Average accuracy for all concepts', fontsize=24)

plt.legend(fontsize=20, loc='upper left')
# plt.savefig('new_plots_fol/architectures_priortrained_all_concepts.png')
plt.show()

# standard all concepts
print("\nArrays for all concepts (standard architectures):")
print('LSTM Standard:', standard)
print('Transformer Standard:', transformer_standard)
print('MLP Standard:', mlp_standard)
print('Human:', human)
print("\nPearson correlation values for all concepts (standard architectures):")
print(f'Pearson r LSTM Standard vs Human: {r_regression(np.array([standard]).T, np.array(human)/100)[0]:.4f}')
print(f'Pearson r Transformer Standard vs Human: {r_regression(np.array([transformer_standard]).T, np.array(human)/100)[0]:.4f}')
print(f'Pearson r MLP Standard vs Human: {r_regression(np.array([mlp_standard]).T, np.array(human)/100)[0]:.4f}')

plt.figure(figsize=(10, 8))
plt.plot(x, standard, marker='o', label='LSTM', color='tab:purple', linestyle='dashed', markersize=10)
plt.plot(x, transformer_standard, marker='>', label='Transformer', color='tab:purple', linestyle='dashed', markersize=10)
plt.plot(x, mlp_standard, marker='<', label='MLP', color='tab:purple', linestyle='dashed', markersize=10)
plt.plot(x, human, marker='s', label='Human', color='tab:green', linestyle='solid', markersize=10)
plt.xlim(20, 105)
plt.ylim(50, 105)
plt.xticks(x, fontsize=20)
plt.yticks(fontsize=20)

plt.ylabel('Accuracy', fontsize=24)
plt.xlabel('Percentage of examples shown', fontsize=24)
plt.title('Average accuracy for all concepts', fontsize=24)

plt.legend(fontsize=20, loc='upper left')
# plt.savefig('new_plots_fol/architectures_standard_all_concepts.png')
plt.show()


def move_elements_to_back(arr, indices):
    mask = np.ones(arr.shape, dtype=bool)
    mask[indices] = False
    moved_elements = arr[indices]
    remaining_elements = arr[mask]
    result = np.concatenate((remaining_elements, moved_elements))

    return result

indices_to_move = numbers = list(range(18, 38)) + list(range(40, 46))

# prior trained
# LSTM
first_array = move_elements_to_back(first_array, indices_to_move)
second_array = move_elements_to_back(second_array, indices_to_move)
third_array = move_elements_to_back(third_array, indices_to_move)
last_array = move_elements_to_back(last_array, indices_to_move)
# Transformer
transformer_first_array = move_elements_to_back(transformer_first_array, indices_to_move)
transformer_second_array = move_elements_to_back(transformer_second_array, indices_to_move)
transformer_third_array = move_elements_to_back(transformer_third_array, indices_to_move)
transformer_last_array = move_elements_to_back(transformer_last_array, indices_to_move)
# MLP
mlp_first_array = move_elements_to_back(mlp_first_array, indices_to_move)
mlp_second_array = move_elements_to_back(mlp_second_array, indices_to_move)
mlp_third_array = move_elements_to_back(mlp_third_array, indices_to_move)
mlp_last_array = move_elements_to_back(mlp_last_array, indices_to_move)

# standard
# LSTM
random_first_array = move_elements_to_back(random_first_array, indices_to_move)
random_second_array = move_elements_to_back(random_second_array, indices_to_move)
random_third_array = move_elements_to_back(random_third_array, indices_to_move)
random_last_array = move_elements_to_back(random_last_array, indices_to_move)
# Transformer
transformer_random_first_array = move_elements_to_back(transformer_random_first_array, indices_to_move)
transformer_random_second_array = move_elements_to_back(transformer_random_second_array, indices_to_move)
transformer_random_third_array = move_elements_to_back(transformer_random_third_array, indices_to_move)
transformer_random_last_array = move_elements_to_back(transformer_random_last_array, indices_to_move)
# MLP
mlp_random_first_array = move_elements_to_back(mlp_random_first_array, indices_to_move)
mlp_random_second_array = move_elements_to_back(mlp_random_second_array, indices_to_move)
mlp_random_third_array = move_elements_to_back(mlp_random_third_array, indices_to_move)
mlp_random_last_array = move_elements_to_back(mlp_random_last_array, indices_to_move)

human_first_array = move_elements_to_back(human_first_array, indices_to_move)
human_second_array = move_elements_to_back(human_second_array, indices_to_move)
human_third_array = move_elements_to_back(human_third_array, indices_to_move)
human_last_array = move_elements_to_back(human_last_array, indices_to_move)

filenames = move_elements_to_back(filenames, indices_to_move)
# print(filenames)

# SIMPLE BOOLEAN CONCEPTS
# prior-trained
# LSTM
new_first_array = first_array[simpleboolean]
new_second_array = second_array[simpleboolean]
new_third_array = third_array[simpleboolean]
new_last_array = last_array[simpleboolean]
# Transformer
transformer_new_first_array = transformer_first_array[simpleboolean]
transformer_new_second_array = transformer_second_array[simpleboolean]
transformer_new_third_array = transformer_third_array[simpleboolean]
transformer_new_last_array = transformer_last_array[simpleboolean]
# MLP
mlp_new_first_array = mlp_first_array[simpleboolean]
mlp_new_second_array = mlp_second_array[simpleboolean]
mlp_new_third_array = mlp_third_array[simpleboolean]
mlp_new_last_array = mlp_last_array[simpleboolean]

# standard
# LSTM
new_random_first_array = random_first_array[simpleboolean]
new_random_second_array = random_second_array[simpleboolean]
new_random_third_array = random_third_array[simpleboolean]
new_random_last_array = random_last_array[simpleboolean]
# Transformer
transformer_new_random_first_array = transformer_random_first_array[simpleboolean]
transformer_new_random_second_array = transformer_random_second_array[simpleboolean]
transformer_new_random_third_array = transformer_random_third_array[simpleboolean]
transformer_new_random_last_array = transformer_random_last_array[simpleboolean]
# MLP
mlp_new_random_first_array = mlp_random_first_array[simpleboolean]
mlp_new_random_second_array = mlp_random_second_array[simpleboolean]
mlp_new_random_third_array = mlp_random_third_array[simpleboolean]
mlp_new_random_last_array = mlp_random_last_array[simpleboolean]


# human
new_human_first_array = human_first_array[simpleboolean]
new_human_second_array = human_second_array[simpleboolean]
new_human_third_array = human_third_array[simpleboolean]
new_human_last_array = human_last_array[simpleboolean]
new_filenames = filenames[simpleboolean]

print('-----------')
print('simple_boolean')

prior_trained = [np.mean(new_first_array), np.mean(new_second_array), np.mean(new_third_array), np.mean(new_last_array)]
transformer_prior_trained = [np.mean(transformer_new_first_array), np.mean(transformer_new_second_array), np.mean(transformer_new_third_array), np.mean(transformer_new_last_array)]
mlp_prior_trained = [np.mean(mlp_new_first_array), np.mean(mlp_new_second_array), np.mean(mlp_new_third_array), np.mean(mlp_new_last_array)]

standard = [np.mean(new_random_first_array), np.mean(new_random_second_array), np.mean(new_random_third_array), np.mean(new_random_last_array)]
transformer_standard = [np.mean(transformer_new_random_first_array), np.mean(transformer_new_random_second_array), np.mean(transformer_new_random_third_array), np.mean(transformer_new_random_last_array)]
mlp_standard = [np.mean(mlp_new_random_first_array), np.mean(mlp_new_random_second_array), np.mean(mlp_new_random_third_array), np.mean(mlp_new_random_last_array)]
# human
human = [np.mean(new_human_first_array), np.mean(new_human_second_array), np.mean(new_human_third_array), np.mean(new_human_last_array)]
print('Prior-trained', prior_trained)
print('Standard', standard)
print('Human', human)

# prior trained
print("\nArrays for SimpleBoolean concepts:")
print('Prior-trained:', prior_trained)
print('Standard:', standard)
print('Human:', human)
print("\nPearson correlation values for SimpleBoolean concepts:")
print(f'Pearson r LSTM Prior-trained vs Human: {r_regression(np.array([prior_trained]).T, np.array(human)/100)[0]:.4f}')
print(f'Pearson r Standard vs Human: {r_regression(np.array([standard]).T, np.array(human)/100)[0]:.4f}')

print("\nArrays for SimpleBoolean concepts (architectures comparison):")
print('LSTM Prior-trained:', prior_trained)
print('Transformer Prior-trained:', transformer_prior_trained)
print('MLP Prior-trained:', mlp_prior_trained)
print('Human:', human)
print("\nPearson correlation values for SimpleBoolean concepts (architectures comparison):")
print(f'Pearson r LSTM Prior-trained vs Human: {r_regression(np.array([prior_trained]).T, np.array(human)/100)[0]:.4f}')
print(f'Pearson r Transformer Prior-trained vs Human: {r_regression(np.array([transformer_prior_trained]).T, np.array(human)/100)[0]:.4f}')
print(f'Pearson r MLP Prior-trained vs Human: {r_regression(np.array([mlp_prior_trained]).T, np.array(human)/100)[0]:.4f}')


plt.figure(figsize=(10, 8))
plt.plot(x, prior_trained, marker='o',label='FOL Prior-trained', color='tab:blue', linestyle='solid' ,markersize=10)
plt.plot(x, standard, marker='o', label='Standard', color='tab:purple', linestyle='dashed', markersize=10)
plt.plot(x, human, marker='s', label='Human', color='tab:green', linestyle='solid',  markersize=10)
plt.xlim(20, 105)
plt.ylim(50, 105)
plt.xticks(x, fontsize=20)
plt.yticks(fontsize=20)

plt.ylabel('Accuracy', fontsize=24)
plt.xlabel('Percentage of examples shown', fontsize=24)
plt.title('Average accuracy for SimpleBoolean concepts', fontsize=24)

# plt.legend(fontsize=20, loc='upper left')
# plt.savefig('new_plots_fol/avg_simpleboolean_concepts')
plt.show()

# standard
print("\nArrays for SimpleBoolean concepts (standard architectures):")
print('LSTM Standard:', standard)
print('Transformer Standard:', transformer_standard)
print('MLP Standard:', mlp_standard)
print('Human:', human)
print("\nPearson correlation values for SimpleBoolean concepts (standard architectures):")
print(f'Pearson r LSTM Standard vs Human: {r_regression(np.array([standard]).T, np.array(human)/100)[0]:.4f}')
print(f'Pearson r Transformer Standard vs Human: {r_regression(np.array([transformer_standard]).T, np.array(human)/100)[0]:.4f}')
print(f'Pearson r MLP Standard vs Human: {r_regression(np.array([mlp_standard]).T, np.array(human)/100)[0]:.4f}')

plt.figure(figsize=(10, 8))
plt.plot(x, prior_trained, marker='o', label='LSTM', color='tab:blue', linestyle='solid', markersize=10)
plt.plot(x, transformer_prior_trained, marker='>', label='Transformer', color='tab:blue', linestyle='solid', markersize=10)
plt.plot(x, mlp_prior_trained, marker='<', label='MLP', color='tab:blue', linestyle='solid', markersize=10)
plt.plot(x, human, marker='s', label='Human', color='tab:green', linestyle='solid', markersize=10)
plt.xlim(20, 105)
plt.ylim(50, 105)
plt.xticks(x, fontsize=20)
plt.yticks(fontsize=20)

plt.ylabel('Accuracy', fontsize=24)
plt.xlabel('Percentage of examples shown', fontsize=24)
plt.title('Average accuracy for SimpleBoolean concepts', fontsize=24)

# plt.legend(fontsize=20, loc='upper left')
# plt.savefig('new_plots_fol/architectures_priortrained_simpleboolean_concepts.png')
plt.show()

plt.figure(figsize=(10, 8))
plt.plot(x, standard, marker='o', label='LSTM', color='tab:purple', linestyle='dashed', markersize=10)
plt.plot(x, transformer_standard, marker='>', label='Transformer', color='tab:purple', linestyle='dashed', markersize=10)
plt.plot(x, mlp_standard, marker='<', label='MLP', color='tab:purple', linestyle='dashed', markersize=10)
plt.plot(x, human, marker='s', label='Human', color='tab:green', linestyle='solid', markersize=10)
plt.xlim(20, 105)
plt.ylim(50, 105)
plt.xticks(x, fontsize=20)
plt.yticks(fontsize=20)

plt.ylabel('Accuracy', fontsize=24)
plt.xlabel('Percentage of examples shown', fontsize=24)
plt.title('Average accuracy for SimpleBoolean concepts', fontsize=24)

# plt.legend(fontsize=20, loc='upper left')
# plt.savefig('new_plots_fol/architectures_standard_simpleboolean_concepts.png')
plt.show()


# NOT SIMPLE BOOLEAN

# prior-trained
# LSTM
new_first_array = first_array[numbers_not_in_simpleboolean]
new_second_array = second_array[numbers_not_in_simpleboolean]
new_third_array = third_array[numbers_not_in_simpleboolean]
new_last_array = last_array[numbers_not_in_simpleboolean]
# Transformer
transformer_new_first_array = transformer_first_array[numbers_not_in_simpleboolean]
transformer_new_second_array = transformer_second_array[numbers_not_in_simpleboolean]
transformer_new_third_array = transformer_third_array[numbers_not_in_simpleboolean]
transformer_new_last_array = transformer_last_array[numbers_not_in_simpleboolean]
# MLP
mlp_new_first_array = mlp_first_array[numbers_not_in_simpleboolean]
mlp_new_second_array = mlp_second_array[numbers_not_in_simpleboolean]
mlp_new_third_array = mlp_third_array[numbers_not_in_simpleboolean]
mlp_new_last_array = mlp_last_array[numbers_not_in_simpleboolean]

# standard
# LSTM
new_random_first_array = random_first_array[numbers_not_in_simpleboolean]
new_random_second_array = random_second_array[numbers_not_in_simpleboolean]
new_random_third_array = random_third_array[numbers_not_in_simpleboolean]
new_random_last_array = random_last_array[numbers_not_in_simpleboolean]
# Transformer
transformer_new_random_first_array = transformer_random_first_array[numbers_not_in_simpleboolean]
transformer_new_random_second_array = transformer_random_second_array[numbers_not_in_simpleboolean]
transformer_new_random_third_array = transformer_random_third_array[numbers_not_in_simpleboolean]
transformer_new_random_last_array = transformer_random_last_array[numbers_not_in_simpleboolean]
# MLP
mlp_new_random_first_array = mlp_random_first_array[numbers_not_in_simpleboolean]
mlp_new_random_second_array = mlp_random_second_array[numbers_not_in_simpleboolean]
mlp_new_random_third_array = mlp_random_third_array[numbers_not_in_simpleboolean]
mlp_new_random_last_array = mlp_random_last_array[numbers_not_in_simpleboolean]


# human
new_human_first_array = human_first_array[numbers_not_in_simpleboolean]
new_human_second_array = human_second_array[numbers_not_in_simpleboolean]
new_human_third_array = human_third_array[numbers_not_in_simpleboolean]
new_human_last_array = human_last_array[numbers_not_in_simpleboolean]
new_filenames = filenames[numbers_not_in_simpleboolean]
print('-----------')
print('NOT simple_boolean')


prior_trained = [np.mean(new_first_array), np.mean(new_second_array), np.mean(new_third_array), np.mean(new_last_array)]
transformer_prior_trained = [np.mean(transformer_new_first_array), np.mean(transformer_new_second_array), np.mean(transformer_new_third_array), np.mean(transformer_new_last_array)]
mlp_prior_trained = [np.mean(mlp_new_first_array), np.mean(mlp_new_second_array), np.mean(mlp_new_third_array), np.mean(mlp_new_last_array)]
# standard
standard = [np.mean(new_random_first_array), np.mean(new_random_second_array), np.mean(new_random_third_array), np.mean(new_random_last_array)]
transformer_standard = [np.mean(transformer_new_random_first_array), np.mean(transformer_new_random_second_array), np.mean(transformer_new_random_third_array), np.mean(transformer_new_random_last_array)]
mlp_standard = [np.mean(mlp_new_random_first_array), np.mean(mlp_new_random_second_array), np.mean(mlp_new_random_third_array), np.mean(mlp_new_random_last_array)]
# human
human = [np.mean(new_human_first_array), np.mean(new_human_second_array), np.mean(new_human_third_array), np.mean(new_human_last_array)]


print('Prior-trained', prior_trained)
print('Standard', standard)
print('Human', human)

# prior trained
print("\nArrays for NOT SimpleBoolean concepts:")
print('Prior-trained:', prior_trained)
print('Standard:', standard)
print('Human:', human)
print("\nPearson correlation values for NOT SimpleBoolean concepts:")
print(f'Pearson r LSTM Prior-trained vs Human: {r_regression(np.array([prior_trained]).T, np.array(human)/100)[0]:.4f}')
print(f'Pearson r Standard vs Human: {r_regression(np.array([standard]).T, np.array(human)/100)[0]:.4f}')

print("\nArrays for NOT SimpleBoolean concepts (architectures comparison):")
print('LSTM Prior-trained:', prior_trained)
print('Transformer Prior-trained:', transformer_prior_trained)
print('MLP Prior-trained:', mlp_prior_trained)
print('Human:', human)
print("\nPearson correlation values for NOT SimpleBoolean concepts (architectures comparison):")
print(f'Pearson r LSTM Prior-trained vs Human: {r_regression(np.array([prior_trained]).T, np.array(human)/100)[0]:.4f}')
print(f'Pearson r Transformer Prior-trained vs Human: {r_regression(np.array([transformer_prior_trained]).T, np.array(human)/100)[0]:.4f}')
print(f'Pearson r MLP Prior-trained vs Human: {r_regression(np.array([mlp_prior_trained]).T, np.array(human)/100)[0]:.4f}')

print("\nArrays for NOT SimpleBoolean concepts (standard architectures):")
print('LSTM Standard:', standard)
print('Transformer Standard:', transformer_standard)
print('MLP Standard:', mlp_standard)
print('Human:', human)
print("\nPearson correlation values for NOT SimpleBoolean concepts (standard architectures):")
print(f'Pearson r LSTM Standard vs Human: {r_regression(np.array([standard]).T, np.array(human)/100)[0]:.4f}')
print(f'Pearson r Transformer Standard vs Human: {r_regression(np.array([transformer_standard]).T, np.array(human)/100)[0]:.4f}')
print(f'Pearson r MLP Standard vs Human: {r_regression(np.array([mlp_standard]).T, np.array(human)/100)[0]:.4f}')


plt.figure(figsize=(10, 8))
plt.plot(x, prior_trained, marker='o',label='FOL Prior-trained', color='tab:blue', linestyle='solid' ,markersize=10)
plt.plot(x, standard, marker='o', label='Standard', color='tab:purple', linestyle='dashed', markersize=10)
plt.plot(x, human, marker='s', label='Human', color='tab:green', linestyle='solid',  markersize=10)
plt.xlim(20, 105)
plt.ylim(50, 105)
plt.xticks(x, fontsize=20)
plt.yticks(fontsize=20)

plt.ylabel('Accuracy', fontsize=24)
plt.xlabel('Percentage of examples shown', fontsize=24)
plt.title('Average accuracy for NOT SimpleBoolean concepts', fontsize=24)

# plt.legend(fontsize=20, loc='upper left')
# plt.savefig('new_plots_fol/avg_not_simpleboolean_concepts')
plt.show()


plt.figure(figsize=(10, 8))
plt.plot(x, prior_trained, marker='o', label='LSTM', color='tab:blue', linestyle='solid', markersize=10)
plt.plot(x, transformer_prior_trained, marker='>', label='Transformer', color='tab:blue', linestyle='solid', markersize=10)
plt.plot(x, mlp_prior_trained, marker='<', label='MLP', color='tab:blue', linestyle='solid', markersize=10)
plt.plot(x, human, marker='s', label='Human', color='tab:green', linestyle='solid', markersize=10)
plt.xlim(20, 105)
plt.ylim(50, 105)
plt.xticks(x, fontsize=20)
plt.yticks(fontsize=20)

plt.ylabel('Accuracy', fontsize=24)
plt.xlabel('Percentage of examples shown', fontsize=24)
plt.title('Average accuracy for NOT SimpleBoolean concepts', fontsize=24)

# plt.legend(fontsize=20, loc='upper left')
# plt.savefig('new_plots_fol/architectures_priortrained_not_simpleboolean_concepts.png')
plt.show()

plt.figure(figsize=(10, 8))
plt.plot(x, standard, marker='o', label='LSTM', color='tab:purple', linestyle='dashed', markersize=10)
plt.plot(x, transformer_standard, marker='>', label='Transformer', color='tab:purple', linestyle='dashed', markersize=10)
plt.plot(x, mlp_standard, marker='<', label='MLP', color='tab:purple', linestyle='dashed', markersize=10)
plt.plot(x, human, marker='s', label='Human', color='tab:green', linestyle='solid', markersize=10)
plt.xlim(20, 105)
plt.ylim(50, 105)
plt.xticks(x, fontsize=20)
plt.yticks(fontsize=20)

plt.ylabel('Accuracy', fontsize=24)
plt.xlabel('Percentage of examples shown', fontsize=24)
plt.title('Average accuracy for NOT SimpleBoolean concepts', fontsize=24)

# plt.legend(fontsize=20, loc='upper left')
# plt.savefig('new_plots_fol/architectures_standard_not_simpleboolean_concepts.png')
plt.show()
