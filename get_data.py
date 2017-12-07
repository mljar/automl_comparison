__author__ = 'Piotr Plonski'

import os
import openml
import pandas as pd
import numpy as np

openml.config.apikey = os.environ['OPENML_KEY']

dataset_ids = [3, 24, 31, 38, 44, 179, 715, 718, 720, 722, 723, 727, 728, 734, 735, 737, 740, 741,
               819, 821, 822, 823, 833, 837, 843, 845, 846, 847]


for dataset_id in dataset_ids:
    print 'Get dataset id', dataset_id
    dataset = openml.datasets.get_dataset(dataset_id)
    (X, y, categorical, names) = dataset.get_data(target=dataset.default_target_attribute, \
                                        return_categorical_indicator=True, \
                                        return_attribute_names=True)
    if len(np.unique(y)) != 2:
        print 'Not binary classification'
        continue
    vals = {}
    for i, name in enumerate(names):
        vals[name] = X[:,i]
    vals['target'] = y
    df = pd.DataFrame(vals)
    df.to_csv('./data/{0}.csv'.format(dataset_id), index=False)


# Print table with datasets descriptions for Readme
#print '| Dataset Id | Name | Rows | Columns |'
#print '| - | - | - | - |'
#for dataset_id in dataset_ids:
#    dataset = openml.datasets.get_dataset(dataset_id)
#    (X, y, categorical, names) = dataset.get_data(target=dataset.default_target_attribute,return_categorical_indicator=True, return_attribute_names=True)
#    print '| {0} | {1} | {2} | {3} |'.format(dataset_id, dataset.name, X.shape[0], X.shape[1])
