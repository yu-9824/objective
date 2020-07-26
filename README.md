## INPUT
clf : 使う機械学習モデルのクラスを代入 (e.g. RandomForestRegressor)

X : ハイパーパラメータ探索に使用する特徴量．pd.DataFrameでもnp.arrayでもOK．

y : 目的変数．pd.Seriesでもnp.arrayでもOK．

random_state : default(None), int．再現性のため．

cv : default(5)．交差検証 (Cross Validation) の回数．cv = 0のとき，8 : 2で分割したときのスコア．

## OUTPUT
None (何もreturnされない)
~~~python

objective.clf(**objective.fixed_params, **study.best_params)

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
objective.clf(**params, **objective.fixed_params)
~~~
で最適なハイパーパラメータ の機械学習モデルを得ることができる．
