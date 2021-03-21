import sys
sys.path.append('/home/jiajunb/prosocial-conversations')

from models.fusedModel.FusedModelSolver import FusedModelSolver
from models.fusedModel.FusedPredictor import FusedPredictor
from pathlib import Path
import torch

ROOT_DIR = Path('/shared/0/projects/prosocial/data/finalized')


def main():
    solver_path = ROOT_DIR / 'model_checkpoints/annotation_albert/run1/best_checkpoint_ep320.pth'
    solver = FusedModelSolver.from_pretrained(
        model_constructor=FusedPredictor,
        pretrained_system_name_or_path=solver_path
    )

    test_data_dir = Path('/shared/0/projects/prosocial/data/finalized/data_cache/fused_albert')
    test_dataloader = solver.get_test_dataloader(test_data_dir, 800)
    mean_loss, metrics_scores, preds, golds = solver.validate(test_dataloader)
    print("Scores on test set: ")
    print(str(metrics_scores))
    torch.save({'preds': preds,
                'golds': golds,
                'score': str(metrics_scores)},
               ROOT_DIR / 'model_checkpoints/annotation_albert/run1/best_checkpoint_ep320.test.preds.golds.pth.tar')


if __name__ == '__main__':
    main()
s