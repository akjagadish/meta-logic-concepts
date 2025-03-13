from itertools import combinations, permutations, product
import copy
import queue
import numpy as np

objects = np.zeros((27, 9))
obj = 0
for i in range(3):
    for j in range(3):
        for k in range(3):
            objects[obj][i] = 1
            objects[obj][j + 3] = 1
            objects[obj][k + 6] = 1
            obj += 1

leaves = ['blue', 'green', 'yellow', 'circle', 'rectangle', 'triangle', 'size1', 'size2', 'size3']

def generate_combinations_with_repetition(arr, k):
    return list(product(arr, repeat=k))

# combinations_with_repetition = generate_combinations_with_repetition(leaves, 3)
# for combination in combinations_with_repetition:
#     print(combination)

def generate_binary_arrays(n):
    return product([0, 1], repeat=n)

n = 3
binary_arrays = generate_binary_arrays(n)

# for elem in binary_arrays:
#     print(elem)
    
class LogicalExpressionTree:
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

def generate_logical_trees(k):
    if k == 1:
        return [LogicalExpressionTree('leaf')]
    
    result = []
    for i in range(1, k):
        left_subtrees = generate_logical_trees(i)
        right_subtrees = generate_logical_trees(k - i)

        for left in left_subtrees:
            for right in right_subtrees:
                result.append(LogicalExpressionTree('and', left, right))
                result.append(LogicalExpressionTree('or', left, right))
    
    return result

def dfs_print_tree(tree, parent=None, side=None):
    if tree is not None:
        print(f"{parent} --{side}--> {tree.value}")
        if tree.left is not None:
            dfs_print_tree(tree.left, tree.value, 'left')
        if tree.right is not None:
            dfs_print_tree(tree.right, tree.value, 'right')

def dfs_populate_leaves(tree):
    if tree is not None:
        if tree.value == 'leaf':
            tree.value = subset_queue.get()
        if tree.left is not None:
            dfs_populate_leaves(tree.left)
        if tree.right is not None:
            dfs_populate_leaves(tree.right)

def evaluate(tree, obj):
    if tree is not None:
        use_not = nots_queue.get()
        if tree.value == 'or':
            return evaluate(tree.left, obj) or evaluate(tree.right, obj) if not use_not \
                    else not (evaluate(tree.left, obj) or evaluate(tree.right, obj))
        elif tree.value == 'and':
            return evaluate(tree.left, obj) and evaluate(tree.right, obj) if not use_not \
                    else not (evaluate(tree.left, obj) and evaluate(tree.right, obj))
        else: # it's a leaf
            if tree.value == 'blue':
                return obj[0] == 1 if not use_not else obj[0] == 0
            elif tree.value == 'green':
                return obj[1] == 1 if not use_not else obj[1] == 0
            elif tree.value == 'yellow':
                return obj[2] == 1 if not use_not else obj[2] == 0
            elif tree.value == 'circle':
                return obj[3] == 1 if not use_not else obj[3] == 0
            elif tree.value == 'rectangle':
                return obj[4] == 1 if not use_not else obj[4] == 0
            elif tree.value == 'triangle':
                return obj[5] == 1 if not use_not else obj[5] == 0
            elif tree.value == 'size1':
                return obj[6] == 1 if not use_not else obj[6] == 0
            elif tree.value == 'size2':
                return obj[7] == 1 if not use_not else obj[7] == 0
            elif tree.value == 'size3':
                return obj[8] == 1 if not use_not else obj[8] == 0

# we can have a not in front of every node so we generate a binary queue indicating if we have a not 
# we use that in evaluating the tree: assume there is a not which will negate the whole substree

map_labels = {0: [[True] * 27, [False] * 27], 1: [], 2: [], 3: [], 4: []}
string_labels = ['1' * 27, '0' * 27]

for k in range(1, 5):
    trees = generate_logical_trees(k)
    subsets = generate_combinations_with_repetition(leaves, k)
    all_trees = []
    # print(subsets)
    for tree in trees:
        original_tree = copy.deepcopy(tree)
        working_tree = copy.deepcopy(tree)
        # print("init orig")
        # dfs_print_tree(original_tree)
        for subset in subsets:
            subset_queue = queue.Queue()
            subset_queue.queue.extend(subset)
            # print(subset)
            dfs_populate_leaves(working_tree)
            all_trees.append(working_tree)
            # print('populated_tree')
            # dfs_print_tree(working_tree)
            working_tree = copy.deepcopy(original_tree)
            # print("____________")

    # NOTE: tress with k leaves have 2k-1 nodes 
    for tree in all_trees:
        # dfs_print_tree(tree)
        all_binary_queues = generate_binary_arrays(2*k-1)
        # for elem in all_binary_queues:
        #     print(elem)
        for elem in all_binary_queues:
            # evaluate on all 27 objects
            labels_rule = []
            for obj in objects:
                nots_queue = queue.Queue()
                nots_queue.queue.extend(elem)
                label_obj = evaluate(tree, obj)
                labels_rule.append(label_obj)
            # print(labels_rule)
            binary_string = ''.join('1' if x else '0' for x in labels_rule)
            found = 0
            if binary_string not in string_labels:
                string_labels.append(binary_string)
                map_labels[k].append(labels_rule)
                np.save('all_rules/' + str(k) + '/' + str(len(map_labels[k])) + '.npy', labels_rule)
                np.save('all_rules/0/2.npy', [False]*27)
            # for key, val in map_labels.items():
            #     if labels_rule in val:
            #         # print(elem)
            #         # print(labels_rule)
            #         found = 1
            #         break


    print(k, len(map_labels[k]))
# for key, val in map_labels.items():
#     print(key, len(val))


