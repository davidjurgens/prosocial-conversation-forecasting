import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import joblib
import sys
from pathlib import Path
sys.path.append('/home/jiajunb/prosocial-conversations')
from models import XGBOOST_FEATURES, EIGENMETRICS

ROOT_DIR = Path('/shared/0/projects/prosocial/data/finalized/')
train_df = pd.read_csv(ROOT_DIR / 'data_cache/lr_or_xgboost/train.tsv', sep='\t', usecols=XGBOOST_FEATURES + EIGENMETRICS)

train_X = train_df[XGBOOST_FEATURES].values
train_y = train_df[EIGENMETRICS].values.reshape(-1)
dummy_clf = DummyRegressor(strategy="mean")
dummy_clf.fit(train_X, train_y)

# on training set
train_preds = dummy_clf.predict(train_X)
print(f'R^2 on training set: {r2_score(train_y, train_preds)}')
print(f'MSELoss on training set: {mean_squared_error(train_preds, train_y)}')

output_path = ROOT_DIR / 'model_checkpoints/dummy'
output_path.mkdir(exist_ok=True, parents=True)
joblib.dump(dummy_clf, output_path / 'dummy.model.buffer')

test_df = pd.read_csv(ROOT_DIR / 'data_cache/lr_or_xgboost/test.tsv', sep='\t', usecols=XGBOOST_FEATURES + EIGENMETRICS)
test_X = test_df[XGBOOST_FEATURES].values
test_y = test_df[EIGENMETRICS].values

# on test set
test_preds = dummy_clf.predict(test_X)
print(f'R^2 on training set: {r2_score(test_y, test_preds)}')
print(f'MSELoss on training set: {mean_squared_error(test_y, test_preds)}')