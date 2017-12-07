__author__ = 'Piotr Plonski'

import os
import numpy as np
import pandas as pd
import sklearn.model_selection
from sklearn.metrics import log_loss

def autosklearn_compute(X_train, y_train, X_test):
    import autosklearn.classification
    automl = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=3600,
                                                                per_run_time_limit=360)
    automl.fit(X_train, y_train)
    response = automl.predict_proba(X_test)[:,1]
    return response

def h2o_compute(X_train, y_train, X_test):
    import h2o
    from h2o.automl import H2OAutoML
    h2o.init(ip='localhost', port='55555', min_mem_size='14g', max_mem_size='15g')
    aml = H2OAutoML(max_runtime_secs = 3600)
    dd = h2o.H2OFrame(pd.concat([X_train, y_train], axis=1))
    dd['target'] = dd['target'].asfactor()
    aml.train(y = 'target', training_frame = dd)
    response = aml.predict(h2o.H2OFrame(X_test))
    return np.array(response[:,2].as_data_frame())

def mljar_compute(X_train, y_train, X_test, dataset_id, seed):
    from mljar import Mljar
    #os.environ['MLJAR_TOKEN'] = '' # set token here or in your env
    if 'MLJAR_TOKEN' not in os.environ:
        raise Exception('Missing MLJAR_TOKEN, please set it.')

    mlj = Mljar('DatasetId_{0}'.format(dataset_id),
                'Seed_{0}'.format(seed),
                metric = 'logloss',
                algorithms = ['xgb', 'lgb', 'rfc', 'etc', 'mlp'],
                tuning_mode = 'Sport',
                create_ensemble  = True,
                single_algorithm_time_limit = 10)

    mlj.fit(X_train, y_train)
    response = mlj.predict(X_test)
    return response

def compute(package, dataset_id, seed):
    try:
        df = pd.read_csv('./data/{0}.csv'.format(dataset_id))
        x_cols = [c for c in df.columns if c != 'target']
        X = df[x_cols]
        y = df['target']

        X_train, X_test, y_train, y_test = \
            sklearn.model_selection.train_test_split(X, y, test_size = 0.3, random_state=seed)

        response = None
        if package == 'auto-sklearn':
            response = autosklearn_compute(X_train, y_train, X_test)
        elif package == 'h2o':
            response = h2o_compute(X_train, y_train, X_test)
        elif package == 'mljar':
            response = mljar_compute(X_train, y_train, X_test, dataset_id, seed)

        # Compute the logloss on test dataset
        ll = log_loss(y_test, response)

        with open('all_results.csv', 'a') as fout:
            fout.write('{0}, {1}, {2}, {3}'.format(package, dataset_id, seed, ll))

    except Exception as e:
        print 'Exception:', str(e)
