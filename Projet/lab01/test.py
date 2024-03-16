from PointSet import PointSet, FeaturesTypes
from Tree import Tree
from read_write import load_data, write_results
import csv
import sys
import evaluation
import numpy as np

files_debug =\
[
    ['../input_data/debug_data1.csv', '../input_data/debug_data2.csv', '../input_data/debug_data3.csv'],
    ['../input_data/debug_data1.csv', '../input_data/debug_data2.csv', '../input_data/debug_data3.csv'],
    ['../input_data/exo4_debug1.csv', '../input_data/exo4_debug2.csv'],
    ['../input_data/debug_data1.csv', '../input_data/debug_data2.csv', '../input_data/debug_data3.csv'],
    ['../input_data/debug_data1.csv', '../input_data/debug_data2.csv', '../input_data/debug_data3.csv'],
    ['../input_data/cat_debug_data1.csv', '../input_data/cat_debug_data2.csv', '../input_data/cat_debug_data3.csv'],
    ['../input_data/cont_debug_data1.csv','../input_data/cont_debug_data2.csv','../input_data/cont_debug_data3.csv'],
    ['../input_data/cont_debug_data1.csv','../input_data/cont_debug_data2.csv','../input_data/cont_debug_data3.csv'],
    ['../input_data/cont_debug_data2.csv','../input_data/cont_debug_data3.csv',],
    ['../input_data/cont_debug_data4.csv',],
]


tkt = files_debug[5]
''''
results = []
training_proportion = .8
for file in tkt:
    features, labels, types = load_data(file)
    training_nb = int(len(features)*training_proportion)
    current_tree = Tree(features[:training_nb], labels[:training_nb], types, h=2)
    expected_results = labels[training_nb:]
    actual_results = []
    for point_features in features[training_nb:]:
        actual_results += [current_tree.decide(point_features)]
    results += [[evaluation.F1_score(expected_results, actual_results)]]
return results'''

features1, labels1, types1 = load_data(files_debug[5][0])
print(types1)
print(np.array(features1))
print(types1[0] == FeaturesTypes.BOOLEAN)
