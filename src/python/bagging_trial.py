import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from ensembles_parallel.baggingModel import BaggingModel

if __name__ == '__main__':

    data = pd.read_csv('../../data/fake_data_1.csv')
    print("Read the data successfully")

    x_flags = [col_name for col_name in data.columns.values if 'x_' in col_name]
    y_flag = 'y'
    print("Extracted the columns")
    clf_gini = DecisionTreeClassifier(criterion="gini", random_state=100)

    train_data, test_data = train_test_split(data, test_size=0.3, random_state=100)

    print("Model training -- started")
    # BaggingModel
    rfModel = BaggingModel(train_data=train_data,
                           test_data=test_data,
                           x=x_flags,
                           y=y_flag)
    rfModel.train_parallel(clf_gini, 70, 0.4, 0.4)

    trial = data[1:10]

    print(rfModel.predict(trial))
    print(rfModel.get_metrics())

