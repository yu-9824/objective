## インストール方法

### pipの場合

~~~
pip install git+https://github.com/yu-9824/objective
~~~

### anacondaの場合
~~~
conda install pip
~~~
のあと，
~~~
pip install git+https://github.com/yu-9824/objective
~~~


## 使い方
exampleフォルダを参照．


## INPUT
clf : 使う機械学習モデルのクラスを代入 (e.g. RandomForestRegressor)

X : ハイパーパラメータ探索に使用する特徴量．pd.DataFrameでもnp.arrayでもOK．

y : 目的変数．pd.Seriesでもnp.arrayでもOK．

random_state : default(None), int．再現性のため．

cv : default(5)．交差検証 (Cross Validation) の回数．cv = 0のとき，8 : 2で分割したときのテストスコアを返す．

scoring : [scikit-learnの'scoring-parameter'](https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter) に従う．

現在使えるのは，'r2\_score', 'neg\_mean\_squared\_error', 'neg\_mean\_absolute\_error' だけ．

特にr2\_scoreのときは他の二つと違って最大化方向が最適方法であり，デフォルトの最小化ではダメなので，

~~~python
study = optuna.create_study(direction = 'maximize')
~~~

とする必要がある．

## OUTPUT
None (何もreturnされない)
~~~python
best_model = objective.clf(**objective.fixed_params, **study.best_params)
~~~

で最適なハイパーパラメータ の機械学習モデルを得ることができる．

例外として，NGBRegressorのとき，

~~~python
from sklearn.tree import DecisionTreeRegressor

params = dict()
sub_params = dict()
for k, v in study.best_params.items():
    if k in objective.params:
	params[k] = v
    else:
	sub_params[k] = v
params['Base'] = DecisionTreeRegressor(**sub_params)
best_model = objective.clf(**params, **objective.fixed_params)
~~~
で最適なハイパーパラメータ の機械学習モデルを得ることができる．
