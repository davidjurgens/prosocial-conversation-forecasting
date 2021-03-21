import sys
sys.path.append('/home/jiajunb/prosocial-conversations')

from models.albert.EigenmetricRegressionSolver import EigenmetricRegressionSolver
from models.albert.AlbertForEigenmetricRegression import AlbertForEigenmetricRegression
from models.fusedModel.FusedDataset import FusedDataset
import torch
import json
from examples.utils import get_scores
from pathlib import Path

ROOT_DIR = Path("/shared/0/projects/prosocial/data/finalized")
# checkpoint_path = ROOT_DIR / 'model_checkpoints/albert/run2/best_checkpoint_ep254710.pth'

ckp_name = 'best_checkpoint_ep6749'
ckp_dir = 'run1_freeze'

checkpoint_dir = ROOT_DIR / f'model_checkpoints/albert/{ckp_dir}'
solver = EigenmetricRegressionSolver.from_pretrained(
    model_constructor=AlbertForEigenmetricRegression,
    pretrained_system_name_or_path=checkpoint_dir / f'{ckp_name}.pth'
)

with open(ROOT_DIR / 'dataframes/subreddit_mappings.json', 'r') as istream:
    subreddit_map = json.load(istream)

data_dir = Path(ROOT_DIR / 'data_cache/fused_albert/cached.test.annotation.albert.tensors.dict')
annotation_dataset = FusedDataset.from_cached_dataset(data_dir)

logits_l = solver.infer(dataset=annotation_dataset.comment_dataset_l).squeeze(dim=1)
logits_r = solver.infer(dataset=annotation_dataset.comment_dataset_r).squeeze(dim=1)

pred = torch.zeros_like(annotation_dataset.labels, dtype=torch.long)

pred[logits_l > logits_r] = 0
pred[logits_l < logits_r] = 1
r = torch.rand(pred[logits_l == logits_r].shape)
pred[logits_l == logits_r] = (r > 0.5).to(torch.long)

print(f'length of equal logits: {len(pred)}')
summary = get_scores(pred, annotation_dataset.labels)
print(f'The performance summary: \n{summary}')

torch.save({'preds': pred,
            'golds': annotation_dataset.labels,
            'summary': summary},
           (checkpoint_dir / f'{ckp_name}.annotation.pred.test'))
print(f"saved predictions at {(checkpoint_dir / f'{ckp_name}.annotation.pred.test')}")
