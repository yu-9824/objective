from sklearn.metrics import r2_score as R2
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

import numpy as np

from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from ngboost import NGBRegressor
from sklearn.tree import DecisionTreeRegressor


class Objective:
    def __init__(self, clf, X, y, random_state = None, cv = 5):
        try:
            clf()
            self.clf = clf
        except:
            self.clf = type(clf)
        self.X = X
        self.y = y
        self.cv = int(cv)
        self.random_state = random_state

        if self.clf == RandomForestRegressor:    # clfには原則モデルのクラスを入れる想定だが，もしインスタンスを入れてしまった場合もできるようにするため．　不具合があったらどちらかのみにする．
            self.fixed_params = {'random_state' : self.random_state}
        elif self.clf == XGBRegressor:
            self.fixed_params = {'random_state' : self.random_state, 'silent' : True, 'objective' : 'reg:squarederror'}
        elif self.clf == GradientBoostingRegressor:
            self.fixed_params = {'random_state' : self.random_state, 'n_iter_no_change' : 5}
        elif self.clf == NGBRegressor:
            self.fixed_params = {'random_state': self.random_state}
        elif self.clf == SVR:
            self.fixed_params = {'gamma' : 'auto'}

    def __call__(self, trial):
        if self.clf == RandomForestRegressor:
            self.params = {
                'min_samples_split' : trial.suggest_int('min_samples_split', 2, 16),
                'max_depth' : trial.suggest_int('max_depth', 10, 500),
                'n_estimators' : trial.suggest_int('n_estimators', 10, 500)
            }
        elif self.clf == XGBRegressor:
            self.params = {
                'n_estimators' : trial.suggest_int('n_estimators', 0, 1000),
                'max_depth' : trial.suggest_int('max_depth', 1, 20),
                'min_child_weight' : trial.suggest_int('min_child_weight', 1, 20),
                'subsample' : trial.suggest_discrete_uniform('subsample', 0.5, 0.9, 0.1),
                'colsample_bytree' : trial.suggest_discrete_uniform('colsample_bytree', 0.5, 0.9, 0.1)
            }
        elif self.clf == GradientBoostingRegressor:
            self.params = {
                'max_depth' : trial.suggest_int('max_depth', 2, 500),
                'n_estimators' : trial.suggest_int('n_estimators', 10, 500),
                'learning_rate' : trial.suggest_uniform('learning_rate', 0.05, 0.5),
                'min_samples_split' : trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf' : trial.suggest_int('min_samples_leaf', 1, 10)
            }
        elif self.clf == NGBRegressor:
            self.sub_params = {
                'max_depth' : trial.suggest_int('max_depth', 2, 10),
                'min_samples_split' : trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf' : trial.suggest_int('min_samples_leaf', 1, 10)
            }
            self.params = {
                'n_estimators' : trial.suggest_int('n_estimators', 100, 1000),
                'learning_rate' : trial.suggest_uniform('learning_rate', 0.005, 0.05),
                'Base' : DecisionTreeRegressor(**self.sub_params)
            }
        elif self.clf == SVR:
            self.params = {
                'C' : trial.suggest_loguniform('C', 1e0, 1e2),
                'epsilon' : trial.suggest_loguniform('epsilon', 1e-1, 1e1)
            }
        clf = self.clf(**self.params, **self.fixed_params)

        # ***** スコア算出法 *****
        if self.cv:  # CV
            scores = cross_val_score(clf, self.X, self.y, scoring = 'r2', cv = 5)
            score = - np.average(scores)
        else:   # もう一回分割
            X_train1, X_train2, y_train1, y_train2 = train_test_split(self.X, self.y, test_size = 0.2, random_state = self.random_state)
            clf.fit(X_train1, y_train1)
            y_pred_on_train2 = clf.predict(X_train2)
            score = - R2(y_train2, y_pred_on_train2)
        return score

if __name__ == '__main__':
    from sklearn.datasets import load_boston
    boston = load_boston()
    X = np.array(boston.data)
    y = np.array(boston.target)

    objective = Objective(RandomForestRegressor(), X, y, random_state = 334, cv = 0)

    import optuna
    study = optuna.create_study()
    study.optimize(objective, n_trials = 1)
