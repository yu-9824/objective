from sklearn.metrics import r2_score as R2
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

import numpy as np

from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor


class Objective:
    def __init__(self, clf, X, y, random_state = None, cv = 5):
        self.clf = clf
        self.X = X
        self.y = y
        self.cv = int(cv)
        if type(self.clf) == type(RandomForestRegressor):
            self.fixed_params = {'random_state' : random_state}
        elif type(self.clf) == type(XGBRegressor):
            self.fixed_params = {'random_state' : random_state, 'silent' : True, 'objective' : 'reg:squarederror'}

    def __call__(self, trial):
        if type(self.clf) == type(RandomForestRegressor):
            params = {
                'min_samples_split' : trial.suggest_int('min_samples_split', 2, 16),
                'max_depth' : trial.suggest_int('max_depth', 10, 500),
                'n_estimators' : trial.suggest_int('n_estimators', 10, 500),
            }
        elif type(self.clf) == type(XGBRegressor):
            params = {
                'n_estimators' : trial.suggest_int('n_estimators', 0, 1000),
                'max_depth' : trial.suggest_int('max_depth', 1, 20),
                'min_child_weight' : trial.suggest_int('min_child_weight', 1, 20),
                'subsample' : trial.suggest_discrete_uniform('subsample', 0.5, 0.9, 0.1),
                'colsample_bytree' : trial.suggest_discrete_uniform('colsample_bytree', 0.5, 0.9, 0.1),
            }
        clf = self.clf(**params, **self.fixed_params)

        # ***** スコア算出法 *****
        if self.cv:  # CV
            scores = cross_val_score(clf, self.X, self.y, scoring = 'r2', cv = 5)
            score = - np.average(scores)
        else:   # もう一回分割
            X_train1, X_train2, y_train1, y_train2 = train_test_split(self.X, self.y, test_size = 0.2, random_state = random_state)
            clf.fit(X_train1, y_train1)
            y_pred_on_train2 = clf.predict(X_train2)
            score = - R2(y_train2, y_pred_on_train2)
        return score
