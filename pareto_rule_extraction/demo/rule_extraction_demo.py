from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import pandas as pd
from pareto_rule_extraction import rule_extractor
import os

#pd.set_option('max_columns', 999)


def get_data(class_regr='class'):
    df = demodata()
    if class_regr == 'class':
        y = df['quality'] > 5
    else:
        y = df['quality']
    X = df.drop('quality', axis=1)
    return X, y


def demodata():
    fp = os.path.dirname(__file__)
    return pd.read_csv(os.path.join(fp, 'winequality-white.csv'), ';')


def classification_example():
    X, y = get_data(class_regr='class')
    clf = RandomForestClassifier(n_estimators=100, max_depth=2, min_samples_leaf=.1)
    clf.fit(X, y)
    clf.predict()
    rex = rule_extractor.RuleExtractor(clf, feature_names=X.columns, debug=1)
    rc = rex._get_rule_counts()
    print(rc)
    stats = rex.extract_rule_statistics(top_n=10)
    print(stats)
    print(rex._predict_sample(X.iloc[-1]))
    print(rex.rules_summary())


def regression_example():
    X, y = get_data(class_regr='regr')
    clf = RandomForestRegressor(n_estimators=100, max_depth=2, min_samples_leaf=.1)
    clf.fit(X, y)
    clf.predict()
    rex = rule_extractor.RuleExtractor(clf, feature_names=X.columns, debug=1)
    rex._extract_rules()
    rc = rex._get_rule_counts()
    print(rc)
    stats = rex.extract_rule_statistics(top_n=10)
    print(stats)
    print(rex._predict_sample(X.iloc[-1]))
    print(rex.rules_summary())

if __name__ == '__main__':
    print('running regression example')
    regression_example()

    print('running classification example')
    classification_example()
