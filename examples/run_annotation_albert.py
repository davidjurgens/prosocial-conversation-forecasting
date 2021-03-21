import sys
sys.path.append('/home/jiajunb/prosocial-conversations')

from pathlib import Path
from models.fusedModel.FusedPredictor import FusedPredictor
from models.fusedModel.FusedModelSolver import FusedModelSolver
from models.albert.AlbertForEigenmetricRegression import AlbertForEigenmetricRegression


def main():
    args = FusedModelSolver.get_fused_model_arguments()
    model = FusedPredictor.from_scratch(
        model_constructor=AlbertForEigenmetricRegression,
        pretrained_feature_extractor_name_or_path=args.pretrained_feature_extractor_name_or_path,
        freeze_feature_extractor=args.freeze_feature_extractor,
    )
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
    solver.fit(num_eval_per_epoch=args.num_eval_per_epoch)


if __name__ == '__main__':
    main()
