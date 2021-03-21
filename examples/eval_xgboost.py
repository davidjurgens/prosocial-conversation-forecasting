from pathlib import Path
import pandas as pd
from models.xgboost.XgboostSolver import XgboostSolver
from models import EIGENMETRICS
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import joblib


def main():
    model = XgboostSolver.from_pretrained(Path('/shared/0/projects/prosocial/data/finalized/xgboost/1e-3_50000') / 'model.tar.path')
    print('start inference')
    # pred = XgboostSolver.infer(model, Path('/shared/0/projects/prosocial/data/finalized/test_features.tsv'), best_ntree_limit=6856)

if __name__ == '__main__':
    main()
