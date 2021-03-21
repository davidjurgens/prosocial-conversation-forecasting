import sys
sys.path.append('/home/jiajunb/prosocial-conversations')

from models.albert.AlbertForEigenmetricRegression import AlbertForEigenmetricRegression
from models.albert.EigenmetricRegressionSolver import EigenmetricRegressionSolver
from pathlib import Path
import torch

ROOT_DIR = Path('/shared/0/projects/prosocial/data/finalized')


def main():
    solver_path = ROOT_DIR / 'model_checkpoints/albert/run1_freeze/best_checkpoint_ep6749.pth'
    solver = EigenmetricRegressionSolver.from_pretrained(
        model_constructor=AlbertForEigenmetricRegression,
        pretrained_system_name_or_path=solver_path
    )
    test_data_dir = Path('/shared/0/projects/prosocial/data/finalized/data_cache/albert')
    test_dataloader = solver.get_test_dataloader(test_data_dir, 890)
    mean_loss, metrics_scores, preds, golds = solver.validate(test_dataloader)
    print("Scores on test set: ")
    print(str(metrics_scores))
    torch.save({'preds': preds,
                'golds': golds},
               ROOT_DIR / 'model_checkpoints/albert/run1_freeze/best_checkpoint_ep6749.test.preds.golds.pth.tar')


if __name__ == '__main__':
    main()
