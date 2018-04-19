import numpy as np
import _pickle as cPickle
import math
from random import sample
from ensembles_parallel.baseModelClass import BaseModel
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score
from multiprocessing import Pool, cpu_count


def unwrap_self_train(arg, **kwarg):
    return BaggingModel.train_tree(*arg, **kwarg)


__tmp_folder__ = '../../tmp'
__tmp_classifier_pickle__ = __tmp_folder__ + '/my_dumped_classifier.pkl'
__tmp_train_data__ = __tmp_folder__ + '/train_data.csv'
__tmp_subset_prefix__ = __tmp_folder__ + '/subset_'


class BaggingModel:

    def __init__(self, train_data, test_data, x, y):
        self.train_data = train_data
        self.test_data = test_data
        self.x = x
        self.x_size = len(x)
        self.y = y
        self.train_data.to_csv(__tmp_train_data__, sep=',', encoding='utf-8')
        self.train_data_size = len(self.train_data)
        self.baggingModel = []

    def sample_features(self, sample_size):
        x_sample_col_names = sample(self.x, int(math.floor(sample_size*self.x_size)))
        return x_sample_col_names

    def bootstrap_rows(self, sample_size):
        sub_rows = np.random.choice(self.train_data_size, int(math.floor(sample_size*self.train_data_size)))
        return sub_rows

    def train_tree(self, train_parameters):
        model_pickle_name = train_parameters[0]
        i = train_parameters[1]
        feature_sample_percentage = train_parameters[2]
        data_sample_percentage = train_parameters[3]
        train_type = train_parameters[4]

        with open(model_pickle_name, 'rb') as fid:
            model = cPickle.load(fid)

        subset_rows = self.bootstrap_rows(data_sample_percentage)
        sub_file_name = __tmp_subset_prefix__ + str(i) + '.txt'
        f = open(sub_file_name, 'w')
        for sub in subset_rows:
            f.write(str(sub) + '\n')
        f.close()
        features = self.sample_features(feature_sample_percentage)
        print("Training tree - " + str(i))
        iter_model = BaseModel(model_name="model" + str(i),
                               x=features,
                               y=self.y,
                               subset=sub_file_name)
        if train_type == 'parallel':
            iter_model.train(file_name=__tmp_train_data__, model=model)
        else:
            iter_model.train(data=self.train_data, model=model)
        print("finished Training Tree - " + str(i))
        return iter_model

    def train(self, model, number_of_trees, feature_sample_perc, data_sample_perc):
        with open(__tmp_classifier_pickle__, 'wb') as fid:
            cPickle.dump(model, fid)
        for i in range(1, number_of_trees):
            self.baggingModel.append(self.train_tree((__tmp_classifier_pickle__, i,
                                                      feature_sample_perc,
                                                      data_sample_perc,
                                                      'serial')))

    def train_parallel(self, model, number_of_trees, feature_sample_rate, data_sample_rate, number_of_cores=None):
        if number_of_cores is None:
            cores = math.ceil(0.8*cpu_count())
        else:
            cores = number_of_cores
        p = Pool(cores)
        with open(__tmp_classifier_pickle__, 'wb') as fid:
            cPickle.dump(model, fid)
        arg_tuples = [(__tmp_classifier_pickle__, i,
                       feature_sample_rate,
                       data_sample_rate,
                      'parallel') for i in range(1, number_of_trees)]
        models = p.map(unwrap_self_train, zip([self]*len(arg_tuples),arg_tuples))
        self.baggingModel = models

    def predict(self, x):
        model_predictions = np.array([model.predict(x) for model in self.baggingModel])
        prediction_count = np.apply_along_axis(lambda p: Counter(p), 0, model_predictions)
        return {'prediction_count': prediction_count,
                'predictions': np.array([max(pred, key=lambda key:pred[key]) for pred in prediction_count])}

    def get_metrics(self):
        test_predictions = self.predict(self.test_data)['predictions']
        test_actual = self.test_data[self.y]
        return {'accuracy': accuracy_score(test_actual, test_predictions)*100,
                'f1_score': f1_score(test_actual, test_predictions)*100}

    def get_oob(self):
        oob_predictions = np.array([model.predict_for_oob(self.train_data) for model in self.baggingModel])
        oob_prediction_count = np.apply_along_axis(lambda x: Counter([i for i in x if i is not None]),
                                                   0,
                                                   oob_predictions)[0]
        max_prediction_count = np.array([max(prediction, key=lambda key:prediction[key])
                                         for prediction in oob_prediction_count if prediction != Counter()])
        actual_prediction = self.train_data[self.y]
        error = 1 - accuracy_score(actual_prediction, max_prediction_count)
        return error




