```python3

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.metrics import roc_auc_score

from xgboost import XGBClassifier

df = pd.read_csv('/content/X_train.csv', encoding='cp949')
df_y = pd.read_csv('/content/y_train.csv')
test = pd.read_csv('/content/X_test.csv', encoding='cp949')

# 환불금액
df.info()

df.head()

df['환불금액'].fillna(0,inplace=True)
test['환불금액'].fillna(0, inplace=True)

robust_data = df[['총구매액','최대구매액','환불금액','내점일수','내점당구매건수','주말방문비율','구매주기']]
robust = RobustScaler()
robust_result = robust.fit_transform(robust_data)
robust_result = pd.DataFrame(robust_result, columns=['총구매액','최대구매액','환불금액','내점일수','내점당구매건수','주말방문비율','구매주기'])

robust_test = test[['총구매액','최대구매액','환불금액','내점일수','내점당구매건수','주말방문비율','구매주기']]
robust = RobustScaler()
robust_result_test = robust.fit_transform(robust_test)
robust_result_test = pd.DataFrame(robust_result_test, columns=['총구매액','최대구매액','환불금액','내점일수','내점당구매건수','주말방문비율','구매주기'])

df.drop(['cust_id','총구매액','최대구매액','환불금액','내점일수','내점당구매건수','주말방문비율','구매주기'], axis=1, inplace=True)
train = pd.concat([df, robust_result], axis=1)

id = test['cust_id']
test.drop(['cust_id','총구매액','최대구매액','환불금액','내점일수','내점당구매건수','주말방문비율','구매주기'], axis=1, inplace=True)
test = pd.concat([test, robust_result], axis=1)

train  = pd.get_dummies(train)
test = pd.get_dummies(test)

print(set(train) - set(test))

train.drop(['주구매상품_소형가전'], axis=1, inplace=True)

df_y.drop('cust_id', axis=1, inplace=True)

x_train, x_test, y_train, y_test = train_test_split(train, df_y, train_size = 0.75, random_state=44)

x_train

xgb = XGBClassifier(n_estimators= 300,)
xgb.fit(x_train, y_train,
        eval_set = [(x_test, y_test)],
        early_stopping_rounds = 100,
        eval_metric='auc')

pred = xgb.predict(x_test)
roc_auc_score(pred, y_test)

preds = xgb.predict_proba(test)
predict = pd.DataFrame(preds)

sub = pd.DataFrame()
sub['cust_id'] = id
sub['sex'] = predict[0]

sub.to_csv('1500620_정태민.csv', index=False)

```
