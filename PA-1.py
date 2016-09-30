import math
import constants
from operator import itemgetter
import matplotlib.pyplot as pyp
import pandas

#Decision tree
tree = None

#List to hold training data
training_data = []

#List to hold test data
test_data = []

#List to hold the test data class labels for evaluation
test_labels = []

# list of attributes for the dataset
attributes = []

# function to create attributes for given dataset
def create_attributes(attribute_list):
    global attributes
    attributes = attribute_list[:]

# function to reset all the values between successive training and testing cycles
def reset():
    global tree
    global training_data
    global test_data
    global test_labels
    global attributes

    tree = None
    training_data = []
    test_data = []
    test_labels = []
    attributes = []

# function to get the training data
def create_training_data(filename):
    global training_data
    #Input file reader
    train = open(filename, 'r')

    #read input file line by line
    for line in train:
        line = line.strip().rstrip()
        training_data.append(line.split(" "))
    train.close()

# function to get the test data
def create_test_data(filename):
    global test_data
    global test_labels

    #Input file reader
    test = open(filename, 'r')

    #read input file line by line
    for line in test:
        line = line.strip().rstrip()
        test_data.append(line.split(" "))    

    test_labels = [item[0] for item in test_data]
    test.close()

# retrieve unique values of the specified feature in given sample.
def distinct_feature_values(data):
    return list(set(data))

# fetch a subset from given sample based on specified feature value
def data_subset(data, feature, value):
    return [item for item in data if str(item[feature]) == str(value)]

# fetch a list of subset indices from given sample based on specified feature value    
def data_subset_indices(data, feature_index, value):
    return [index for index, item in enumerate(data) if str(item[feature_index]) == str(value)]
        
# find out the dominant value of class label from specified sample
def compute_leaf_node_label(data):
    dominant_label_count = -1
    dominant_label = -1
    label_distribution = {}
    for item in data:
        label = item[0]
        if label in label_distribution:
            label_distribution[label] += 1
        else:    
            label_distribution[label] = 1
    
    for key, value in label_distribution.iteritems():
        if value > dominant_label_count:
            dominant_label = key
            dominant_label_count = value
    
    return dominant_label
    
# calculate accuracy and misclassification number on a given test dataset
def calculate_accuracy(test_labels, predicted_labels):
    correct_prediction_count = 0
    for i, val in enumerate(test_labels):
        if val == predicted_labels[i]:
            correct_prediction_count += 1

    accuracy = float(correct_prediction_count) / len(test_labels) * 100
    num_of_misclassification = len(test_labels) - correct_prediction_count
    return accuracy, num_of_misclassification

# generate confusion matrix for the given dataset
def confusion_matrix(test_labels, predicted_labels, result_file):
    tp = tn = fp = fn = 0
    
    for i, val in enumerate(test_labels):
        if predicted_labels[i] != val:
            if (predicted_labels[i] == 1):
                fp += 1
            else:
                fn += 1
        else:
            if (predicted_labels[i] == 1):
                tp += 1
            else:
                tn += 1

    print >> result_file, '\n'            
    print >> result_file, ('{0:{w}}' '{1:{a}{w2}}'.format(' ', ' Model Result ', w=10, w2=50, a='^'))
    print >> result_file, ('{0:{f}{a}{w}} '.format('_', f='_', w=65, a='^'))
    print >> result_file, ('{0:{a}{w2}}' '| {1:{a}{w}} | ' '{2:{a}{w}} | '.format('', 'class = 1', 'class = 0 ', w2=24, w=10, a='^'))
    print >> result_file, ('{0:{f}{a}{w}} '.format('_', f='_', w=65, a='^'))
    print >> result_file, ('{0:{a}{w}} | ' '{1:{a}{w}} | ' '{2:{a}{w}} | ' '{3:{a}{w}} | ' '{4:{a}{w}} | '.format(' ', 'class = 1', 'TP = ' + str(tp), 'FN = ' + str(fn), 'Total = ' + str(fn + tp), w=10, a='^'))
    print >> result_file, ('{0:{a}{w}} ''{1:{f}{a}{w2}}'.format('True Result','_', f='_', w=10, w2=54, a='^'))
    print >> result_file, ('{0:{a}{w}} | ''{1:{a}{w}} | ''{2:{a}{w}} | ''{3:{a}{w}} | ''{4:{a}{w}} | '.format(' ','class = 0', 'FP = ' + str(fp), 'TN = ' + str(tn), 'Total = ' + str(tn + fp), w=10, a='^'))
    print >> result_file, ('{0:{f}{a}{w}} '.format('_', f='_', w=65, a='^'))
    print >> result_file, ('{0:{a}{w2}} ''{1:{a}{w}} | ''{2:{a}{w}} | ''{3:{a}{w}} | '.format(' ', 'Total = ' + str(fp + tp), 'Total = ' + str(tn + fn),str(tn + fn + fp + tp), w2=24, w=10, a='^'))
    print >> result_file, ('{0:{f}{a}{w}} '.format('_', f='_', w=65, a='^'))

# plot learning curve between depth of the tree and accuracy.
def plot_curve():
    result = pandas.read_csv(constants.ACCURACY_PLOT_FILE)
    
    pyp.plot(result[[0]], result[[1]], 'red', linewidth='1', label='Dataset-1')
    pyp.plot(result[[0]], result[[2]], 'green', linewidth='1', label='Dataset-2')
    pyp.plot(result[[0]], result[[3]], 'blue', linewidth='1', label='Dataset-3')
    pyp.plot(result[[0]], result[[4]], 'black', linewidth='3', label='Average Accuracy')

    pyp.ylabel('Accuracy')
    pyp.xlabel('Depth of Tree')
    pyp.legend(loc='upper right')
    pyp.show()

# class representing the decision tree
class DecisionTree:
    # Initialize tree attributes
    label = attribute = value = children = parent = depth = None
    max_depth = constants.MAX_DEPTH
    
    # create a decision tree based on provided parameters
    def __init__(self, data, attributes, value=None, parent=None, max_depth=None, depth=0):
        # fetch the list of class labels for provided data
        labels = [item[0] for item in data]
        
        # initialize the maximum depth of tree as specified by user
        if max_depth is not None:
            self.max_depth = max_depth

        # store current value of the chosen attribute to form a new branch   
        if value is not None:
            self.value = value

        # parent node if present    
        if parent is not None:
            self.parent = parent

        self.depth = depth + 1        
        # If leaf is reached, assign dominant label to it
        if not data or not attributes or self.depth == self.max_depth:
            self.label = compute_leaf_node_label(data)
            return

        # Set first label if all the labels are same
        if labels.count(labels[0]) == len(data):
            self.label = labels[0]
            return

        # Select best feature to split the data
        self.attribute = self.select_feature(data, attributes)
        
        # store index of chosen feature
        attribute_index = attributes.index(self.attribute)
       
        # store feature values for the chosen attribute
        attribute_data = [item[attribute_index] for item in data]
        
        # child node will be further split on other attributes expect its parent
        child_attributes = attributes[:]
        child_attributes.remove(self.attribute)

        self.children = []
        # Create child nodes for every value of parent feature
        for val in distinct_feature_values(attribute_data):            
            child_data = data_subset(data, attribute_index, val)
            self.children.append(DecisionTree(child_data, child_attributes, value=val, parent=self, depth=self.depth, max_depth=self.max_depth))
        return
     
    # represent created tree in a readable format 
    def __str__(self, tabs=0):
        if self.label is not None:
            ret = "\t" * tabs + repr(self.label) + "\n"
        else:    
            ret = "\t" * tabs + repr(self.attribute) + "\n"
        if self.children is not None:
            for child in self.children:
                ret += child.__str__(tabs+1)
        return ret 
        
    # function to find entropy of given sample    
    def entropy(self, data):
        label_distribution = {}
        for item in data:
            label = item[0]
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
    def joint_entropy(self, data, feature):
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

            j_entropy += feature_probability * self.entropy(filtered_data)
        return j_entropy

    # select the feature with highest information gain to spit the dataset.
    def select_feature(self, data, attributes):
        j_entropy = 1.0
        best_feature = None
        for i in attributes:
            if i != 'class':
                current_entropy = self.joint_entropy(data, attributes.index(i))
                if current_entropy <= j_entropy:
                    j_entropy = current_entropy
                    best_feature = i
        return best_feature
        
    # function to test the performance of model on test dataset    
    def classify(self, data):
        # return if no data
        if len(data) == 0:
            return

        # flatten if only one datapoint    
        if len(data) == 1:
            data = data[0]

        # If leaf node then return array of same labels of the size of data
        if self.children is None:
            return [self.label] * len(data)

        # initialize all labels to zero    
        labels = [0] * len(data)

        for child in self.children:
            # Fetch indices of subset of data which align with current attribute value
            filtered_data_indices = data_subset_indices(data, attributes.index(self.attribute), child.value)

            # skip the loop if no data is present for current attribute value
            if not filtered_data_indices:
                continue

            # recursively classify child nodes to further split into smaller groups    
            classified_labels = child.classify(itemgetter(*filtered_data_indices)(data))
            for i in range(0, len(filtered_data_indices)):
                labels[filtered_data_indices[i]] = classified_labels[i]

        return labels    

# function to generate and print required outputs
def report(max_depth, result_file):    
    tree = DecisionTree(training_data, attributes, max_depth=max_depth)
    print >> result_file, '\n#####################################'
    print >> result_file, '#####       Decision Tree      ######'
    print >> result_file, '#####################################'
    print >> result_file, tree

    predicted_labels = tree.classify(test_data)
    accuracy, num_of_misclassifications = calculate_accuracy(test_labels, predicted_labels)

    print >> result_file, '\nAccuracy = ', accuracy
    print >> result_file, 'Misclassifications = ', num_of_misclassifications

    print >> result_file, '\n#####################################'
    print >> result_file, '#####     Confusion Matrix     ######'
    print >> result_file, '#####################################'
    confusion_matrix(test_labels, predicted_labels, result_file)
    
# function used to trigger the program and train the model
def launch():
    for dataset in range(1,4):
        reset()
        result_file = open('output\output_dataset_%s.txt'%(dataset), 'w')

        # attributes for monks dataset
        create_attributes(constants.MONKS_DATASET_ATTR)
        create_training_data('data\monks-%s.train'%(dataset))
        create_test_data('data\monks-%s.test'%(dataset))
    
        for i in range(1, 8):
            print >> result_file, '###############    Depth = %s     ###############' %(i)
            report(i, result_file)
        result_file.close()    

# function used to trigger the program and train the model
def launch_own_data():
        reset()
        result_file = open('output\output_dataset_own.txt', 'w')

        # attributes for own dataset
        create_attributes(constants.OWN_DATASET_ATTR)
        create_training_data('data\\breastcancer-train.csv')
        create_test_data('data\\breastcancer-test.csv')
    
        for i in range(1, 3):
            print >> result_file, '###############    Depth = %s     ###############' %(i)
            report(i, result_file)
        result_file.close()    

# train and test all three datasets
launch()

# train and test all three datasets
launch_own_data()

# Generate learning curve. The CSV file is created first which is needed to generate this graph.
plot_curve()

print 'Output files : \'output_dataset_1.txt\', \'output_dataset_2.txt\', \'output_dataset_3.txt\', \'output_dataset_own.txt\' succefully generated'