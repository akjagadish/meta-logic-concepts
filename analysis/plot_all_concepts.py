import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle

fol_first_array = np.load('lstm_ans_arrays_fol/first_array.npy')
fol_second_array = np.load('lstm_ans_arrays_fol/second_array.npy')
fol_third_array = np.load('lstm_ans_arrays_fol/third_array.npy')
fol_last_array = np.load('lstm_ans_arrays_fol/last_array.npy')

simpleboolean_first_array = np.load('lstm_ans_arrays_simpleboolean/first_array.npy')
simpleboolean_second_array = np.load('lstm_ans_arrays_simpleboolean/second_array.npy')
simpleboolean_third_array = np.load('lstm_ans_arrays_simpleboolean/third_array.npy')
simpleboolean_last_array = np.load('lstm_ans_arrays_simpleboolean/last_array.npy')

random_first_array = np.load('lstm_ans_arrays_simpleboolean/random_first_array.npy')
random_second_array = np.load('lstm_ans_arrays_simpleboolean/random_second_array.npy')
random_third_array = np.load('lstm_ans_arrays_simpleboolean/random_third_array.npy')
random_last_array = np.load('lstm_ans_arrays_simpleboolean/random_last_array.npy')

human_first_array = np.load('lstm_ans_arrays_simpleboolean/human_first_array.npy')
human_second_array = np.load('lstm_ans_arrays_simpleboolean/human_second_array.npy')
human_third_array = np.load('lstm_ans_arrays_simpleboolean/human_third_array.npy')
human_last_array = np.load('lstm_ans_arrays_simpleboolean/human_last_array.npy')

filenames = np.load('lstm_ans_arrays_simpleboolean/filenames.npy')


def move_elements_to_back(arr, indices):
    mask = np.ones(arr.shape, dtype=bool)
    mask[indices] = False
    moved_elements = arr[indices]
    remaining_elements = arr[mask]
    result = np.concatenate((remaining_elements, moved_elements))

    return result

indices_to_move = numbers = list(range(18, 38)) + list(range(40, 46))

fol_first_array = move_elements_to_back(fol_first_array, indices_to_move)
fol_second_array = move_elements_to_back(fol_second_array, indices_to_move)
fol_third_array = move_elements_to_back(fol_third_array, indices_to_move)
fol_last_array = move_elements_to_back(fol_last_array, indices_to_move)

simpleboolean_first_array = move_elements_to_back(simpleboolean_first_array, indices_to_move)
simpleboolean_second_array = move_elements_to_back(simpleboolean_second_array, indices_to_move)
simpleboolean_third_array = move_elements_to_back(simpleboolean_third_array, indices_to_move)
simpleboolean_last_array = move_elements_to_back(simpleboolean_last_array, indices_to_move)

random_first_array = move_elements_to_back(random_first_array, indices_to_move)
random_second_array = move_elements_to_back(random_second_array, indices_to_move)
random_third_array = move_elements_to_back(random_third_array, indices_to_move)
random_last_array = move_elements_to_back(random_last_array, indices_to_move)

human_first_array = move_elements_to_back(human_first_array, indices_to_move)
human_second_array = move_elements_to_back(human_second_array, indices_to_move)
human_third_array = move_elements_to_back(human_third_array, indices_to_move)
human_last_array = move_elements_to_back(human_last_array, indices_to_move)

filenames = move_elements_to_back(filenames, indices_to_move)
print(filenames)
print(filenames[0], filenames[1], filenames[2])

simpleboolean = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 23, 24, 25, 
                77, 78, 79, 80, 81, 82, 83, 84, 85, 86]

names = [
'true',
'false',
'blue',
'circles',
'not circles',
'circles or blue',
'circles or triangles',
'blue or green',
'circles and blue',
'circles and [not blue]',
'not [circles and blue]',
'not [circles or blue]',
'not[ [not circles] or blue]',
'circles XOR blue',
'not [circles XOR blue]',
'circles XOR [not blue]',
'everything iff there is a triangle',
'size=3',
'size=2',
'size=1',
'size=3 or 1',
'size= 3 or 2',
'size= 2 or 1',
'size=1 and blue',
'size=1 or blue',
'one of the largest or one of the smallest',
'the unique largest',
'unique largest and blue',
'unique largest or blue',
'one of the largest',
'one of the largest and blue',
'one of the largest or blue',
'there exists a smaller one',
'there exists a smaller blue one',
'same shape as a blue object',
'same shape as a blue or circle object',
'same shape as a blue or green object',
'same shape as a blue object and not blue',
'[same shape as a blue object] or green',
'[same shape as a blue object] and green',
'[same shape as a blue object] and circle',
'same shape as the unique largest',
'same shape as one of the largest',
'same shape as one of the largest and blue',
'same shape as one of the largest or blue',
'same shape as the unique largest but not the largest',
'same shape as one of the largest but not one of the largest',
'unique blue object',
'unique circle',
'the unique element and is [blue or green]',
'the unique element and is [blue and circle]',
'the unique element and is [blue or circle]',
'the unique largest blue one',
'the unique largest blue or green object',
'shape of the unique largest blue one',
'same shape as one of the largest blue ones',
'there exists another object with the same shape',
'there exists another object with the same size',
'[there exists another object with the same shape] or blue',
'[there exists antoher object with the same shape] and blue',
'there exists another object with the same color',
'there does not exist another object with the same shape',
'there exists another object with the same shape and color',
'every other object with the same shape is your color',
'[every other object with the same shape is your color] or blue',
'[every other object with the same shape is your color] or circle',
'every object with your shape is blue',
'every other object with the same shape is your color',
'every other object with the same shape is not your color',
'there exists a blue object with the same shape',
'there exists one object with the same shape, and one with the same color',
'there exists one object with the same shape, and a different one with the same color',
'there exists one object with the same shape which has another object of the same color',
'shares a feature with every other object',
'[there exists another object with the same shape] and blue',
'there exists another object with the same shape, and one with the same color',
'circle implies blue',
'blue implies circle',
'[not blue] implies circle',
'[not blue] implies [not circle]',
'blue implies size=1',
'[circle or triangle] implies blue',
'[circle and blue] or [triangle and green]',
'[circle or blue] or [triangle and green]',
'circle or [blue and triangle]',
'circle or [blue implies triangle]',
'there exists a blue object of the same shape (may be this object)',
'same size as a circle',
'same shape as another object which is blue',
'same shape as another [blue or green] object',
'same shape as a [blue or green] object (potentially itself)',
'the unique object that is blue',
'the unique object that is [blue or circle]',
'the unique object that is [blue or green]',
'the unique object that is [blue and circle]',
'the unique object',
'same size as the unique blue object',
'unique smallest',
'one of the smallest',
'one of the smallest of its shape',
'the unique smallest of its shape',
'exactly one other element is blue',
'exactly one element is blue',
'exactly one other element is the same color',
'same shape as exactly one blue object',
'same shape as exactly one other blue object',
'same shape as exactly one blue object',
'every other object with the same shape is blue',
'every object with the same shape is blue',
'every-other-atleastone object with the same shape is blue',
'every-atleastone object with the same shape is the same color',
'every-other-atleastone object with the same shape is not the same color'
]

print(np.mean(fol_first_array), np.mean(fol_second_array), np.mean(fol_third_array), np.mean(fol_last_array))
print(np.mean(simpleboolean_first_array), np.mean(simpleboolean_second_array), np.mean(simpleboolean_third_array), np.mean(simpleboolean_last_array))
print(np.mean(random_first_array), np.mean(random_second_array), np.mean(random_third_array), np.mean(random_last_array))
print(np.mean(human_first_array), np.mean(human_second_array), np.mean(human_third_array), np.mean(human_last_array))

# set up the grid of plots
fig, axes = plt.subplots(8, 7, figsize=(30, 30), constrained_layout=True)

# Flatten the axes array to make it easier to iterate over
axes = axes.flatten()

print(len(names))
pos = -1 # -1 for first, 55 for last
x = [25, 50, 75, 100]
for j, ax in enumerate(axes):
    i = j * 2# + 0 for first, +112 for last
    pos += 1
    # print(i)
    fol_prior = [(fol_first_array[i] + fol_first_array[i+1]) / 2, (fol_second_array[i] + fol_second_array[i+1]) / 2, 
            (fol_third_array[i] + fol_third_array[i+1]) / 2, (fol_last_array[i] + fol_last_array[i+1]) / 2]

    simpleboolean_prior = [(simpleboolean_first_array[i] + simpleboolean_first_array[i+1]) / 2, (simpleboolean_second_array[i] + simpleboolean_second_array[i+1]) / 2, 
            (simpleboolean_third_array[i] + simpleboolean_third_array[i+1]) / 2, (simpleboolean_last_array[i] + simpleboolean_last_array[i+1]) / 2]

    standard = [(random_first_array[i] + random_first_array[i+1]) / 2, (random_second_array[i] + random_second_array[i+1]) / 2, 
                (random_third_array[i] + random_third_array[i+1]) / 2, (random_last_array[i] + random_last_array[i+1]) / 2]

    human = [(human_first_array[i] + human_first_array[i+1]) / 2, (human_second_array[i] + human_second_array[i+1]) / 2, 
            (human_third_array[i] + human_third_array[i+1]) / 2, (human_last_array[i] + human_last_array[i+1]) / 2]

    print(i, pos, filenames[i])
    print(fol_prior, simpleboolean_prior, standard, human)
    sns.lineplot(x=x, y=simpleboolean_prior, label='SimpleBoolean Prior-trained', marker='o', linestyle='solid', color='tab:orange', ax=ax, markersize=10)
    sns.lineplot(x=x, y=fol_prior, label='FOL Prior-trained', marker='o', linestyle='solid', color='tab:blue', ax=ax, markersize=10)
    sns.lineplot(x=x, y=standard, label='Standard', marker='o', ax=ax, linestyle='dashed', color='tab:purple', markersize=10)
    sns.lineplot(x=x, y=human, label='Human', marker='s', ax=ax, linestyle='solid', color='tab:green', markersize=10)

    # ax.grid(true, linestyle='-', color='lightgray')
    title_box_width = ax.get_position().width
    if len(names[pos]) > 60:
        third = len(names[pos])//3
        while(names[pos][third - 1] != ' '):
            third += 1

        third2 = len(names[pos]) - len(names[pos])//3
        while(names[pos][third2 - 1] != ' '):
            third2 += 1
        ax.set_title(names[pos][ : third] + '\n' + names[pos][third : third2] + 
        '\n' + names[pos][third2 :], fontsize=16)
    elif len(names[pos]) > 36:
        half = len(names[pos])//2
        #print(half, names[pos])
        while(names[pos][half - 1] != ' '):
            half += 1
        ax.set_title(names[pos][ : half] + '\n' + names[pos][half :], fontsize=16)
    else:
        ax.set_title(names[pos], fontsize=16)
    ax.set_xlim(20, 105)
    ax.set_ylim(0, 105)
    if pos >= 49: # + 56 for last 56
        ax.set_xticks(x)
        ax.set_xticklabels(ax.get_xticks(), fontsize=16)
    if i % 7 == 0:
        ax.set_yticks(x)
        ax.set_yticklabels(ax.get_yticks(), fontsize=16)
    # Hide x-axis ticks for all but the last row of plots
    if pos < 49: # + 56 for last 56
        ax.xaxis.set_tick_params(which='both', labelbottom=False)

    # Hide y-axis ticks for all but the first column of plots
    if i % 7 != 0:
        ax.yaxis.set_tick_params(which='both', labelleft=False)

    if (i / 2 + 1) in simpleboolean:
        rect = Rectangle((0, 0), 1, 1, transform=ax.transAxes, color='lightgrey', alpha=0.5)
        ax.add_patch(rect)
    if pos == 55: # 111 for last 56, 55 otherwise
        ax.legend(loc='lower right', fontsize=15)
    else:
        ax.get_legend().remove() 


plt.savefig('new_plots_fol/first_56_concepts.png')
plt.show()

# print('hi')
# print(len(filenames), len(first_array), len(random_first_array), len(human_first_array))
# for i, filename in enumerate(filenames):
#     if i % 2:
#         continue
#     print(filename)
#     plt.figure()
#     # prior-trained
#     plt.plot([25, 50, 75, 100], [(first_array[i] + first_array[i+1]) / 2, 
#                                         (second_array[i] + second_array[i+1]) / 2, 
#                                         (third_array[i] + third_array[i+1]) / 2, 
#                                         (last_array[i] + last_array[i+1]) / 2],  marker='o', label='Prior-trained')
#     # random
#     plt.plot([25, 50, 75, 100], [(random_first_array[i] + random_first_array[i+1]) / 2, 
#                                         (random_second_array[i] + random_second_array[i+1]) / 2, 
#                                         (random_third_array[i] + random_third_array[i+1]) / 2, 
#                                         (random_last_array[i] + random_last_array[i+1]) / 2], marker='v', label='standard')
#     # human
#     plt.plot([25, 50, 75, 100], [(human_first_array[i] + human_first_array[i+1]) / 2, 
#                                         (human_second_array[i] + human_second_array[i+1]) / 2, 
#                                         (human_third_array[i] + human_third_array[i+1]) / 2, 
#                                         (human_last_array[i] + human_last_array[i+1]) / 2],  marker='s', label='Human')

#     plt.xlabel('percentage of examples shown')
#     plt.ylabel('accuracy')
#     plt.legend()
#     plt.ylim(50, 101)
#     plt.xticks([25, 50, 75, 100])  # set x-axis ticks to the specified values
#     plt.xlim(24, 101) 
#     plt.savefig('plots/test_concept1.png')
#     plt.show()

#     break