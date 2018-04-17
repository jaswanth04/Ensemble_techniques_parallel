import pandas as pd
import numpy as np


class BaseModel:

    def __init__(self, model_name, x, y, subset=None):
        self.model_name = model_name
        self.x = x
        self.y = y
        self.subset_file = subset
        self.model = None

    def read_subset_file(self):
        sub_file = open(self.subset_file, 'r')
        subset = [int(line.strip()) for line in sub_file.readlines()]
        sub_file.close()
        return subset

    def train(self, model, data=None, file_name=None):
        if data is None:
            data = pd.read_csv(file_name)

        if self.subset_file is not None:
            if isinstance(self.subset_file, np.ndarray) or isinstance(self.subset_file, list):
                data_subset = data.loc[self.subset_file].dropna()
            else:
                data_subset = data.loc[self.read_subset_file()].dropna()
        else:
            data_subset = data

        x = data_subset[data_subset.columns.intersection(self.x)]
        y = data_subset[self.y]

        self.model = model
        self.model.fit(x, y)

    def predict(self,x):
        x_featurised = x[x.columns.intersection(self.x)]
        return self.model.predict(x_featurised)

    def predict_for_oob(self, train_data = None, file_name = None):
        if train_data is None:
            train_data = pd.read_csv(file_name)

        subset_rows = self.read_subset_file()
        predictions = []

        for i,row in train_data.iterrows():
            if i in subset_rows:
                predictions.append(None)
            else:
                predictions.append(self.model.predict(row[self.x].reshape(1,-1))[0])

        return np.array(predictions)
