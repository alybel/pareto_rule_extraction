from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import pandas as pd

def get_data(class_regr = 'class'):
    df = pd.read_csv('winequality-whites.csv', ';')
    if class_regr == 'class':
        df['high_quality'] = df['quality'] > 5


def classifiction_example():
    pass

def run_regression_example():
    pass


if __name__ == '__main__':
    run_classification_example()
    run_regression_example()