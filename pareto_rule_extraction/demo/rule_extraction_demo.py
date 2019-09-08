from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import pandas as pd


def get_data(class_regr='class'):
    df = pd.read_csv('winequality-white.csv', ';')
    if class_regr == 'class':
        y = df['quality'] > 5
    else:
        y = df['quality']
    X = df.drop('quality', axis=1)
    return X, y


def classification_example():
    X, y = get_data(class_regr='class')
    pass


def regression_example():
    X, y = get_data(class_regr='regr')
    pass


if __name__ == '__main__':
    classification_example()
    regression_example()
