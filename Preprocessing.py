import numpy as np

def to_categorical(input):
    #Find how many unique values exist
    uniquevals = []
    for i in input:
        if i not in uniquevals:
            uniquevals.append(i)
    
    #Assign each unique value to its own list
    vals_to_list = {}
    for i, uv in enumerate(uniquevals):
        current_list = []
        for j in range(len(uniquevals)):
            if i == j:
                current_list.append(1)
            else:
                current_list.append(0)

        vals_to_list.update({uv : current_list})

    #Make output 2D array
    output = []
    for i in input:
        output.append(vals_to_list.get(i))

    print('Categories: ' + str(vals_to_list))

    return output

def normalize(input):
    """Puts data into a range from 0 to 1
    Takes 2D array"""
    output = np.array(input, dtype='f')
    for col in range(len(input[0])):
        maxx = max(output[:, col])
        for i, num in enumerate(output[:, col]):
            output[i, col] = float(num / maxx)

    return output