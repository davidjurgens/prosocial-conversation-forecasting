import sys
sys.path.append('/home/jiajunb/prosocial-conversations')

from models.xgboost.XgboostSolver import XgboostSolver
from models.fusedModel.FusedDataset import FusedDataset
from models import ANNOTATION_XGBOOST_META_FEATURES, ANNOTATION_XGBOOST_LABELS_HEADER
from pathlib import Path
import pandas as pd
from examples.utils import get_scores
import xgboost as xgb
import torch

ROOT_DIR = Path('/shared/0/projects/prosocial/data/finalized/')
pretrained_system_dir = ROOT_DIR / 'model_checkpoints/xgboost/run4/'
solver = XgboostSolver.from_pretrained(pretrained_system_dir=pretrained_system_dir)

data = pd.read_csv(ROOT_DIR / 'data_cache/annotation_xgboost/test.tsv', sep='\t')

header_l, header_r = FusedDataset.assemble_columns_headers(ANNOTATION_XGBOOST_META_FEATURES)
dtest_l = xgb.DMatrix(data[header_l].values)
dtest_r = xgb.DMatrix(data[header_r].values)
logits_l = torch.tensor(solver.infer(dmatx=dtest_l), dtype=torch.long)
logits_r = torch.tensor(solver.infer(dmatx=dtest_r), dtype=torch.long)

gold = torch.tensor(data[ANNOTATION_XGBOOST_LABELS_HEADER].values, dtype=torch.long)

pred = torch.zeros_like(gold, dtype=torch.long)
pred[logits_l > logits_r] = 0
pred[logits_l < logits_r] = 1
r = torch.rand(pred[logits_l == logits_r].shape)
pred[logits_l == logits_r] = (r > 0.5).to(torch.long)

summary = get_scores(pred, gold)
print(f'The performance summary: \n{summary}')

torch.save({'preds': pred,
            'golds': gold,
            'summary': summary},
           (pretrained_system_dir / f'annotation.pred.test'))
print(f"saved predictions at {(pretrained_system_dir / f'annotation.pred.test')}")

