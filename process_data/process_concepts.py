import os
import numpy as np
# assign directory
directory = 'flat_boolean_concepts'
 
# iterate over files in
# that directory
for filename in os.listdir(directory):
    rows = []
    tags = []
    f = os.path.join(directory, filename)
    file = open(f, 'r')
    print(f)
    if f[-3:] != 'txt':
        continue
    line = file.readline()
    while True:
        line = file.readline()
        if not line:
            break
        start = line.find(")")
        prefix = line[1 : start]
        labels = prefix.split(' ')
        # print(labels)
        # if false make it 0, init as -1 for empty objects
        labels_array = np.full((5, 1), 2)
        for i in range(len(labels)):
            if labels[i] == '#t':
                labels_array[i] = 1
            elif labels[i] == '#f':
                labels_array[i] = 0
        # print(labels_array)

        line = line[start + 1:]
        line = line.strip()
        # print(line)

        items_array = np.zeros((5, 9), dtype=int)
        items = line.split('\t')
        for i in range(len(items)):
            properties = items[i].split(',')   
            print(properties)
            # circle: [1, 0, 0], square= [0, 1, 0], triangle = [0, 0, 1]
            # shape properties[0]   
            if properties[0] == 'circle':
                items_array[i][3] = 1
            elif properties[0] == 'rectangle':
                items_array[i][4] = 1
            elif properties[0] == 'triangle':
                items_array[i][5] = 1 
            # blue: [1, 0, 0], green = [0, 1, 0], yellow = [0, 0, 1] 
            # color properties[1]
            if properties[1] == 'blue':
                items_array[i][0] = 1
            elif properties[1] == 'green':
                items_array[i][1] = 1
            elif properties[0] == 'yellow':
                items_array[i][2] = 1 
            # size1: [1, 0, 0], size2 = [0, 1, 0], size3 = [0, 0, 1] 
            # size properties[2]
            if properties[2] == '1':
                items_array[i][6] = 1
            elif properties[2] == '2':
                items_array[i][7] = 1
            elif properties[2] == '3':
                items_array[i][8] = 1 
        print(items_array)
        rows.append(items_array)
        tags.append(labels_array)
        # print(items)
    # break after the dataset for one concept for test
    rows = np.array(rows)
    tags = np.array(tags)
    rows = rows[:25, :, :]
    tags = tags[:25, :, :]
    print(rows.shape)
    print(tags.shape)
    print(directory + '_arrays_rows/' + filename[:-4] + '.npy')
    np.save(directory + '_arrays_rows/' + filename[:-4] + '.npy', rows)
    np.save(directory + '_arrays_tags/' + filename[:-4] + '.npy', tags)


