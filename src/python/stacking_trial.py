import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from ensembles_parallel.stackingModel import StackingModel

if __name__ == '__main__':
    data = pd.read_csv('../../data/fake_data_1.csv')
    print("Read the data successfully")

    x_flags = [col_name for col_name in data.columns.values if 'x_' in col_name]
    y_flag = 'y'
    print("Extracted the columns")

    trial = data[1:50]

    data_na_removed = data.dropna(axis=0, how='any')

    clf1 = KNeighborsClassifier(n_neighbors=1)
    clf2 = RandomForestClassifier(random_state=1)
    clf3 = GaussianNB()
    lr = LogisticRegression()

    classifier_list = [clf1, clf2, clf3]

    msk = np.random.rand(len(data)) < 0.7
    # train_data, test_data = train_test_split(data_na_removed, test_size=0.3, random_state=100)

    train_data = data[msk].reset_index(drop=True)
    test_data = data[~msk].reset_index(drop=True)

    stacker = StackingModel(train_data=train_data,
                            test_data=test_data,
                            x=x_flags,
                            y=y_flag)

    stacker.train_parallel(model_list=classifier_list, combiner=lr, n_folds=5)

    print(stacker.models)

    print(stacker.predict(trial))

    print(stacker.get_metrics())

