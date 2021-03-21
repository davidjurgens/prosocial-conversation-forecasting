import torch
from pathlib import Path


def get_scores(preds: torch.tensor, golds: torch.tensor) -> dict:
    summary = dict()
    summary['accuracy'] = float((preds == golds).sum()) / len(golds)
    summary['recall_l'] = float(((golds == 0) & (preds == 0)).sum().item()) / (golds == 0).sum().item()
    summary['recall_r'] = float(((golds == 1) & (preds == 1)).sum().item()) / (golds == 1).sum().item()
    summary['f1_score'] = 2. * (summary['recall_l'] * summary['recall_r']) / (summary['recall_l'] + summary['recall_r'])
    return summary
