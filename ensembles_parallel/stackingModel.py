from sklearn.model_selection import KFold
from ensembles_parallel.baseModelClass import BaseModel
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import multiprocessing
from ensembles_parallel.subPool import SubPool
import numpy as np
from math import ceil, sqrt

__tmp_folder__ = '../../tmp'
__tmp_classifier_pickle__ = __tmp_folder__ + '/my_dumped_classifier.pkl'
__tmp_train_data__ = __tmp_folder__ + '/train_data.csv'
__tmp_subset_prefix__ = __tmp_folder__ + '/stack_subset_'


def unwrap_train_and_predict(arg, **kwarg):
    return StackingModel.train_and_predict_fold(*arg, **kwarg)


def unwrap_train_model(arg, **kwarg):
    return StackingModel.train_model(*arg, **kwarg)


class StackingModel:

    def __init__(self, train_data, test_data, x, y):
        self.train_data = train_data
        self.test_data = test_data
        self.x = x
        self.y = y
        self.models = {}
        self.fold_models = {}
        self.combiner_input = None

    @staticmethod
    def read_subset_file(subset_file):
        sub_file = open(subset_file, 'r')
        subset = [int(line.strip()) for line in sub_file.readlines()]
        sub_file.close()
        return subset

    @staticmethod
    def write_subset_to_file(i, fold):
        sub_file_train_name = __tmp_subset_prefix__ + '_train_' + str(i) + '.txt'
        sub_file_test_name = __tmp_subset_prefix__ + '_test_' + str(i) + '.txt'
        f_train = open(sub_file_train_name, 'w')
        for row in fold[0]:
            f_train.write(str(row) + '\n')
        f_train.close()
        f_test = open(sub_file_test_name, 'w')
        for row in fold[1]:
            f_test.write(str(row) + '\n')
        f_test.close()
        return sub_file_train_name, sub_file_test_name

    def train_fold(self, i, train_subset, model):
        iter_model = BaseModel('model_'+str(i),
                               x=self.x,
                               y=self.y,
                               subset=train_subset)
        print("training iter model: "+iter_model.model_name)
        iter_model.train(model=model,
                         data=self.train_data)
        print("Training the iter_model "+iter_model.model_name+" is finished")
        return iter_model

    def predict_fold(self, i, test_subset, iter_model):
        data_to_be_predicted = self.train_data.loc[test_subset]
        x_predictions = pd.Series(iter_model.predict(data_to_be_predicted),
                                  index=test_subset,
                                  name='x_'+str(i))
        return x_predictions

    def train_and_predict_fold(self, train_parameters):
        model_id = train_parameters[0]
        fold_id = train_parameters[1]
        fold = train_parameters[2]
        model = train_parameters[3]

        if isinstance(fold[1], np.ndarray) or isinstance(fold[1], list):
            test_subset = fold[1]
        else:
            test_subset = self.read_subset_file(subset_file=fold[1])

        iter_model = self.train_fold(i=fold_id,
                                     train_subset=fold[0],
                                     model=model)
        x_predictions = self.predict_fold(i=model_id,
                                          test_subset=test_subset,
                                          iter_model=iter_model)

        return iter_model, x_predictions

    def train_model(self, model_parameters):
        model_id = model_parameters[0]
        final_folds = model_parameters[1]
        model = model_parameters[2]
        cores = model_parameters[3]

        p = multiprocessing.Pool(cores)
        arg_tuples = [(model_id, i, fold, model) for i, fold in enumerate(final_folds)]

        model_and_folds = p.map(unwrap_train_and_predict, zip([self] * len(arg_tuples), arg_tuples))
        p.close()
        p.join()

        fold_models, fold_frames = zip(*model_and_folds)

        predicted_df = pd.concat(fold_frames)

        delivered_model = BaseModel('delivered_model_' + str(model_id),
                                    x=self.x,
                                    y=self.y)
        print("Training delivery models: " + delivered_model.model_name)
        delivered_model.train(model=model,
                              data=self.train_data)
        print("Training on " + delivered_model.model_name + " is finished")
        return predicted_df, model_id, fold_models, delivered_model

    def train_combiner_model(self, x_frames, combiner):
        x_frames.append(self.train_data[self.y])
        self.combiner_input = pd.concat(x_frames, axis=1)
        combiner_features = [col_name for col_name in self.combiner_input.columns.values if 'x_' in col_name]

        combiner_model = BaseModel('combiner_model',
                                   x=combiner_features,
                                   y=self.y,
                                   subset=None)
        print("training combiner model: " + combiner_model.model_name)
        combiner_model.train(combiner, data=self.combiner_input)

        self.models['combiner_model'] = combiner_model

    def train(self, model_list, combiner, n_folds=3):
        # Creating Folds
        k_folds = KFold(n_splits=n_folds)
        final_folds = list(k_folds.split(self.train_data))

        frames = []
        for m, model in enumerate(model_list):
            model_and_folds = [self.train_and_predict_fold((m, i, fold, model)) for i, fold in enumerate(final_folds)]
            fold_models, fold_frames = zip(*model_and_folds)

            frames.append(pd.concat(fold_frames))
            self.fold_models['classifier_' + str(m)] = fold_models

            delivered_model = BaseModel('delivered_model_'+str(m),
                                        x=self.x,
                                        y=self.y)
            print("Training delivery models: "+delivered_model.model_name)
            delivered_model.train(model=model,
                                  data=self.train_data)
            print("Training of "+delivered_model.model_name+" is finished")
            self.models[delivered_model.model_name] = delivered_model

        self.train_combiner_model(frames, combiner)

    def train_parallel(self, model_list, combiner, n_folds=3, number_of_cores=None):
        # Creating Folds
        k_folds = KFold(n_splits=n_folds)
        final_folds = [self.write_subset_to_file(i, fold) for i, fold in enumerate(k_folds.split(self.train_data))]

        if number_of_cores is None:
            cores = ceil(0.8*sqrt(multiprocessing.cpu_count()))
        else:
            cores = ceil(sqrt(number_of_cores))
        arg_tuples = [(m, final_folds, model, cores) for m, model in enumerate(model_list)]
        p = SubPool(cores)
        models_and_frames = p.map(unwrap_train_model, zip([self]*len(arg_tuples), arg_tuples))

        p.close()
        p.join()

        frames = []

        for df, model_id, fold_model, delivered_model in models_and_frames:
            frames.append(df)
            self.fold_models['classifier_' + str(model_id)] = fold_model
            self.models[delivered_model.model_name] = delivered_model

        self.train_combiner_model(frames, combiner)

    def predict(self, x):
        predictions = dict([('x_'+model.split('_')[2], self.models[model].predict(x))
                            for model in self.models if model != 'combiner_model'])
        combiner_input = pd.DataFrame(predictions)

        final_predictions = self.models['combiner_model'].predict(combiner_input)
        return final_predictions

    def get_metrics(self):
        test_predictions = self.predict(self.test_data)
        test_actual = self.test_data[self.y]
        return {'accuracy': accuracy_score(test_actual, test_predictions) * 100,
                'f1_score': f1_score(test_actual, test_predictions) * 100}









