from sklearn.model_selection import KFold
from ensembles_parallel.baseModelClass import BaseModel
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

__tmp_folder__ = '../../tmp'
__tmp_classifier_pickle__ = __tmp_folder__ + '/my_dumped_classifier.pkl'
__tmp_train_data__ = __tmp_folder__ + '/train_data.csv'
__tmp_subset_prefix__ = __tmp_folder__ + '/stack_subset_'


class StackingModel:

    def __init__(self, train_data, test_data, x, y):
        self.train_data = train_data
        self.test_data = test_data
        self.x = x
        self.y = y
        self.models = {}
        self.fold_models = {}
        self.combiner_input = None

    def train(self, model_list, combiner, n_folds=3):
        # Creating Folds
        k_folds = KFold(n_splits=n_folds)
        final_folds = []

        for i, fold in enumerate(k_folds.split(self.train_data)):
            sub_file_name = __tmp_subset_prefix__ + str(i) + '.txt'
            f = open(sub_file_name, 'w')
            for row in fold[0]:
                f.write(str(row)+'\n')
            f.close()
            final_folds.append((sub_file_name, fold[1]))

        # print(final_folds)

        frames = []
        for m, model in enumerate(model_list):
            fold_models = []
            fold_frames = []
            for i, fold in enumerate(final_folds):
                # Training on a single fold
                iter_model = BaseModel('model_'+str(i),
                                       x=self.x,
                                       y=self.y,
                                       subset=fold[0])
                print("training iter model: "+iter_model.model_name)
                iter_model.train(model=model,
                                 data=self.train_data)
                fold_models.append(iter_model)

                # y_subset = self.train_data[self.y].loc[fold[1]]
                data_to_be_predicted = self.train_data.loc[fold[1]]
                x_predictions = pd.Series(iter_model.predict(data_to_be_predicted),
                                          index=fold[1],
                                          name='x_'+str(m))
                fold_frames.append(x_predictions)

            frames.append(pd.concat(fold_frames))
            self.fold_models['classifier_' + str(m)] = fold_models

            delivered_model = BaseModel('delivered_model_'+str(m),
                                        x=self.x,
                                        y=self.y)
            print("Training delivery models: "+delivered_model.model_name)
            delivered_model.train(model=model,
                                  data=self.train_data)
            self.models[delivered_model.model_name] = delivered_model

        frames.append(self.train_data[self.y])

        self.combiner_input = pd.concat(frames, axis=1)
        combiner_features = [col_name for col_name in self.combiner_input.columns.values if 'x_' in col_name]

        combiner_model = BaseModel('combiner_model',
                                   x=combiner_features,
                                   y=self.y,
                                   subset=None)
        print("training combiner model: "+combiner_model.model_name)
        combiner_model.train(combiner, data=self.combiner_input)

        self.models['combiner_model'] = combiner_model

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









