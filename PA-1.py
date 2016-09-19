import math

#List to hold training data
training_data = []

#Input file reader
f = open('monks-1.train', 'r')

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
    
def sample_entropy():
    label_distribution = {}
    for item in training_data:
        class_label = item['class']
        if class_label in label_distribution:
            label_distribution[class_label] = label_distribution[class_label] + 1
        else:    
            label_distribution[class_label] = 1
    
    return entropy(label_distribution, len(training_data))
        
#calculates entropy of class label. Takes frequency distribution of class labels in the form dictionary.     
def entropy(label_distribution, data_size):
    entropy = 0.0
    for key, value in label_distribution.iteritems():
        probability = float(value)/data_size
        entropy = entropy + (-1 * probability * math.log(probability, 2))
    return entropy

#calculates joint entropy of class label given a feature.    
def joint_entropy(joint_feature):
    #dictionary of feature values[KEY] and corresponding class label distribution[VALUE].
    joint_label_distribution = {}
    
    #dictionary of class labels frequency distribution.
    label_distribution = {}
    for item in training_data:
        class_label = item['class']
        feature_val = item[joint_feature]
        tmp_label_distribution = {}
        if feature_val in joint_label_distribution:
            tmp_label_distribution = joint_label_distribution[feature_val]
            if class_label in tmp_label_distribution:
                tmp_label_distribution[class_label] = tmp_label_distribution[class_label] + 1
            else:    
                tmp_label_distribution[class_label] = 1
        else:    
            tmp_label_distribution[class_label] = 1
        joint_label_distribution[feature_val] = tmp_label_distribution
            
    j_entropy = 0.0
    data_size = len(training_data)
    for key, label_value in joint_label_distribution.iteritems():
        label_count = 0.0
        for key2, val in label_value.iteritems():
            label_count = label_count + val            
        
        feature_probability = float(label_count)/data_size
        
        tmp_j_entropy = entropy(label_value, label_count)
        print 'ENT=', tmp_j_entropy
        j_entropy = j_entropy + (tmp_j_entropy * feature_probability)
        
    return j_entropy
        

sample_entropy = sample_entropy()
print 'ent = ', sample_entropy

j_entropy = joint_entropy('a2')
print j_entropy
f.close()