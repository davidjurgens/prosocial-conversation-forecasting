import sys
sys.path.append('/home/jiajunb/prosocial-conversations')

from pathlib import Path
from models.fusedModel.FusedPredictor import FusedPredictor
from models.fusedModel.FusedModelSolver import FusedModelSolver
from models.albert.AlbertForEigenmetricRegression import AlbertForEigenmetricRegression
import torch


def main():
    args = FusedModelSolver.get_fused_model_arguments()
    ROOT_DIR = Path('/shared/0/projects/prosocial/data/finalized')
    solver_path = ROOT_DIR / 'model_checkpoints/annotation_albert/run1/last_checkpoint_ep796.pth'
    model = FusedPredictor.from_scratch(
        model_constructor=AlbertForEigenmetricRegression,
        pretrained_feature_extractor_name_or_path=args.pretrained_feature_extractor_name_or_path,
        freeze_feature_extractor=args.freeze_feature_extractor,
    )

    ckp = torch.load(solver_path)
    model.load_state_dict(ckp['state_dict'])
    model.eval()
    solver = FusedModelSolver.from_scratch(
        model,
        args.input_dir,
        args.output_dir,
        args.learning_rate,
        args.n_epoch,
        args.per_gpu_batch_size,
        args.weight_decay,
        args.seed
    )
    test_data_dir = Path('/shared/0/projects/prosocial/data/finalized/data_cache/fused_albert')
    test_dataloader = solver.get_test_dataloader(test_data_dir, 800)
    mean_loss, metrics_scores, preds, golds = solver.validate(test_dataloader)
    print("Scores on test set: ")
    print(str(metrics_scores))
    torch.save({'preds': preds,
                'golds': golds,
                'score': str(metrics_scores)},
               ROOT_DIR / 'model_checkpoints/annotation_albert/run1/last_checkpoint_ep796.test.preds.golds.pth.tar')


if __name__ == '__main__':
    main()
