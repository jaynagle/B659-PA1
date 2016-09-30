# training datasets
TRAINING_DATA_1 = 'monks-1.train'
TRAINING_DATA_2 = 'monks-2.train'
TRAINING_DATA_3 = 'monks-3.train'

# test datasets
TEST_DATA_1 = 'monks-1.test'
TEST_DATA_2 = 'monks-2.test'
TEST_DATA_3 = 'monks-3.test'

# file used to create learning curve plot
ACCURACY_PLOT_FILE = 'test_results.csv'
CLASS_LABEL = 'class'
MAX_DEPTH = 16

# Attributes for Own dataset
'''
Abbreviations of Features
CT = Clump Thickness
SI = Uniformity of Cell Size
SH = Uniformity of Cell Shape
MA = Marginal Adhesion
SE = Single Epithelial Cell Size
BN = Bare Nuclei
BC = Bland Chromatin
NN = Normal Nucleoli
MI = Mitoses
'''
OWN_DATASET_ATTR = ['class','CT','SI','SH','MA','SE','BN','BC','NN','MI']

# Attributes for Monks dataset
MONKS_DATASET_ATTR = ['class', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6']


