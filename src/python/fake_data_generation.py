from sklearn.datasets import make_classification
import pandas as pd


if __name__ == '__main__':

    data = make_classification(n_samples=6000, n_features=200, n_classes=2)
    cols = ['x_' + str(i) for i in range(0, 200)]

    df = pd.DataFrame(data[0], columns=cols)
    df['y'] = data[1]

    df.to_csv('../../data/fake_data_1.csv', sep=',', encoding='utf-8')

    print('Successfully generated fake data')