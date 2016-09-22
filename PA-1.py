import math
import constants
import pprint

#List to hold training data
training_data = []

#Tree depth counter
tree_depth_counter = 0

#Input file reader
f = open(constants.TRAINING_DATA, 'r')

#read input file line by line
for line in f:
    features = {}
    features_array = line.split(" ")
    features['class'] = features_array[1] 
    features['a1'] = features_array[2] 
    features['a2'] = features_array[3] 
    features['a3'] = features_array[4]
    features['a4'] = features_array[5]
    features['a5'] = features_array[6]
    features['a6'] = features_array[7]
    features['id'] = features_array[8]
    training_data.append(features)

#calculates entropy of class label.
def entropy(data):
    label_distribution = {}
    for item in data:
        label = item[constants.CLASS_LABEL]
        if label in label_distribution:
            label_distribution[label] += 1
        else:    
            label_distribution[label] = 1

    entropy = 0.0
    for key, value in label_distribution.iteritems():
        probability = float(value)/len(data)
        entropy += (-1 * probability * math.log(probability, 2))        
    return entropy

#calculates joint entropy of class label given a feature.    
def joint_entropy(data, feature):
    #dictionary of feature frequency distribution.
    label_distribution = {}
    for item in data:
        feature_val = item[feature]
        if feature_val in label_distribution:
            label_distribution[feature_val] += 1
        else:    
            label_distribution[feature_val] = 1

    j_entropy = 0.0
    for key, value in label_distribution.iteritems():
        filtered_data = []
        feature_probability = float(value)/len(data)
        for item in data:
            if item[feature] == key:
                filtered_data.append(item)

        j_entropy += feature_probability * entropy(filtered_data)
    return j_entropy

# select the feature with highest information gain to spit the dataset.
def select_feature(data, attributes):
    j_entropy = 1.0
    best_feature = None
    for i in attributes:
        current_entropy = joint_entropy(data, i)
        if current_entropy <= j_entropy:
            j_entropy = current_entropy
            best_feature = i
    return best_feature

# retrieve unique values of the specified feature in given sample.
def distinct_feature_values(data, feature):
    return set(item[feature] for item in data)

# fetch a subset from given sample based on specified feature value
def data_subset(data, feature, value):
    return [item for item in data if str(item[feature]) == str(value)]
    
# find out the dominant value of class label from specified sample
def compute_leaf_node_label(data):
    dominant_label_count = -1
    dominant_label = -1
    label_distribution = {}
    for item in data:
        label = item[constants.CLASS_LABEL]
        if label in label_distribution:
            label_distribution[label] += 1
        else:    
            label_distribution[label] = 1
    
    for key, value in label_distribution.iteritems():
        if value > dominant_label_count:
            dominant_label = key
            dominant_label_count = value
    
    return dominant_label

# create a decision tree    
def decision_tree(data, attributes):
    global tree_depth_counter
    labels = [item[constants.CLASS_LABEL] for item in data]
        
    #or tree_depth_counter >= constants.DEPTH    
    if not data or tree_depth_counter >= constants.DEPTH or len(attributes) <= 1:
        return compute_leaf_node_label(data)
    elif labels.count(labels[0]) == len(data):
        return labels[0]
    else:
        tree_depth_counter += 1
        # Select a feature to split the data on 
        current_node = select_feature(data, attributes)

        # Initialize root node
        tree = {current_node:{}}

        # Recursively create children for every value of chosen feature.
        for val in distinct_feature_values(data, current_node):
            subtree = decision_tree(
                data_subset(data, current_node, val),
                [attr for attr in attributes if attr != current_node])

            tree[current_node][val] = subtree
    return tree

def train():    
    attributes = []
    for i in range(1,7):
        attributes.append('a' + str(i))
    tree = decision_tree(training_data, attributes)
    pprint.pprint(tree)

f.close()

